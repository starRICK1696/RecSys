"""Reproducible KT-2 suite. Run with python -m RecSys.kt2_experiments --help."""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.stats import rankdata
from .data import split_interactions, inject_noise
from .engine import evaluate
from .models import EASE, SVD_AE
from .kt2_models import ResearchAE, GenreCosine


def load_movielens(directory, max_users=0, max_items=0):
    ratings = np.array([list(map(int, line.split("::"))) for line in
                        (Path(directory)/"ratings.dat").read_text().splitlines()], dtype=int)
    ratings = ratings[ratings[:, 2] >= 4]
    ids, counts = np.unique(ratings[:, 1], return_counts=True)
    if max_items:
        ids = np.sort(ids[np.argsort(-counts, kind="stable")[:max_items]])
    ratings = ratings[np.isin(ratings[:, 1], ids)]
    users, counts = np.unique(ratings[:, 0], return_counts=True)
    users = users[counts >= 22]  # same eligible population for 5/10/20 + val/test
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(1729).choice(users, max_users, replace=False))
    ratings = ratings[np.isin(ratings[:, 0], users)]
    rows, cols = np.searchsorted(users, ratings[:, 0]), np.searchsorted(ids, ratings[:, 1])
    matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), len(ids)))
    matrix.data.fill(1)
    movies = {}
    for line in (Path(directory)/"movies.dat").read_text(encoding="latin-1").splitlines():
        item, _, genre = line.split("::")
        movies[int(item)] = genre.split("|")
    genres = sorted({g for item in ids for g in movies[int(item)]})
    features = np.array([[float(g in movies[int(item)]) for g in genres] for item in ids])
    return matrix, features, {"raw_user_ids": users.tolist(), "raw_item_ids": ids.tolist(), "genres": genres}


def detection(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    p, n = labels.sum(), (~labels).sum()
    if not p or not n:
        return {"auroc": None, "ap": None, "prevalence": float(labels.mean())}
    auc = (rankdata(scores)[labels].sum()-p*(p+1)/2)/(p*n)
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]; precision = tp/(ends+1)
    ap = np.sum(np.diff(np.r_[0, tp])/p * precision)
    return {"auroc": float(auc), "ap": float(ap), "prevalence": float(p/len(labels))}


def shorten(matrix, length, seed):
    result = matrix.copy()
    rng = np.random.default_rng(seed)
    for u in range(matrix.shape[0]):
        lo, hi = result.indptr[u:u+2]
        order = rng.permutation(hi-lo)
        result.data[lo+order[length:]] = 0
    result.eliminate_zeros()
    return result


def terciles(values):
    bounds = np.quantile(values, [1/3, 2/3])
    return np.searchsorted(bounds, values, side="right")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    matrix, features, mapping = load_movielens(args.data, args.max_users, args.max_items)
    if min(matrix.shape) < 2:
        raise ValueError("not enough eligible data")
    manifest = {"dataset": "MovieLens-1M", "shape": list(matrix.shape), "nnz": matrix.nnz,
        "subset": bool(args.max_users or args.max_items), "selection": "ratings>=4; most popular items if capped; random eligible users seed 1729; >=22 positives",
        "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "protocol": "random leave-one-out validation/test; noise after split, train only; exclude heldouts from additions; mask original train UNION observed noisy train",
        "tuning": "per variant clean validation NDCG@20; 3 trials per lambda, alpha=[.005,.01,.05], trim paired with [.05,.1,.2]; freeze across noise levels; test never used for selection",
        "limitations": "original events are not verified clean; subset results exploratory; no transfer claim beyond this linear decoder",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        **mapping}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, diagnostics, groups, tuning = [], [], [], []
    methods = ["unit", "trim", "gaussian", "student", "cauchy"]
    def record(model, train, ds, meta, clean=None):
        start = perf_counter(); model.fit(train); duration = perf_counter()-start
        mask = ds.train.maximum(train)
        result = evaluate(model, train, ds.test, seen=mask)
        row = {**meta, "ndcg": result.ndcg, "recall": result.recall, "train_seconds": duration,
               "inference_seconds": result.inference_seconds, "users": result.users_evaluated}
        if hasattr(model, "confidence_"):
            rr, cc = train.nonzero()
            original = ds.train if clean is None else clean
            labels = np.asarray(original[rr, cc]).ravel() == 0
            row.update(detection(labels, 1-model.confidence_))
            if meta["suite"] == "robustness" and meta["noise_type"] == "noisy_1" and meta["noise"] == .1:
                popularity = np.asarray(original.sum(0)).ravel()
                history = np.asarray(original.sum(1)).ravel()
                for axis, values in [("popularity", terciles(popularity)[cc]), ("history", terciles(history)[rr])]:
                    for group in range(3):
                        for added in [False, True]:
                            sel = (values == group)&(labels == added)
                            if sel.any():
                                w = model.confidence_[sel]
                                diagnostics.append({**meta, "axis": axis, "group": group, "added": added,
                                    "count": int(sel.sum()), "mean_weight": float(w.mean()), "q10": float(np.quantile(w,.1)),
                                    "q50": float(np.median(w)), "q90": float(np.quantile(w,.9)), "low_weight_fraction": float((w<.5).mean())})
                # Preserve actual distributions, not just aggregate means.
                np.savez_compressed(out/f'weights_{meta["model"]}_{meta["seed"]}.npz', weights=model.confidence_,
                    added=labels, popularity=popularity[cc], history=history[rr], errors=model.errors_,
                    popularity_group=terciles(popularity)[cc], history_group=terciles(history)[rr])
        if meta["suite"] == "robustness" and meta["noise_type"] == "noisy_1" and meta["noise"] == .1:
            for axis, value in [("history", np.asarray(ds.train.sum(1)).ravel()),
                                ("popularity", np.asarray(ds.test@np.asarray(ds.train.sum(0)).ravel()).ravel())]:
                membership = terciles(value)
                for group in range(3):
                    heldout = sp.diags((membership == group).astype(float))@ds.test
                    if heldout.nnz:
                        metric = evaluate(model, train, heldout.tocsr(), seen=mask)
                        groups.append({**meta, "axis": axis, "group": group, "ndcg": metric.ndcg, "users": metric.users_evaluated})
        rows.append(row)
        write_csv(out/"metrics.csv", rows); write_csv(out/"diagnostics.csv", diagnostics); write_csv(out/"groups.csv", groups)
        print(f'{len(rows):03} {meta} NDCG={result.ndcg:.4f}', flush=True)
        return model
    for seed in args.seeds:
        ds = split_interactions(matrix, seed=seed, name="ml1m")
        selected = {}
        def tune(name, method, f=None, rank=None, train=None):
            train = ds.train if train is None else train
            candidates = []
            for lam in args.lambdas:
                for alpha, candidate_trim in zip([.005, .01, .05], [.05, .1, .2]):
                    trim = candidate_trim if method == "trim" else .1
                    kw = dict(rank=rank or args.rank, regularization=lam, method=method, outer=args.outer,
                              trim=trim, zero_weight=alpha, features=f, seed=seed)
                    model = ResearchAE(**kw).fit(train)
                    metric = evaluate(model, train, ds.validation, seen=ds.train)
                    tuning.append({"seed": seed, "variant": name, "lambda": lam, "trim": trim, "zero_weight": alpha, "ndcg": metric.ndcg})
                    candidates.append((metric.ndcg, kw))
            selected[name] = max(candidates, key=lambda x:x[0])[1]
            write_csv(out/"tuning.csv", tuning)
            return selected[name]
        if "robustness" in args.suites or "stability" in args.suites or "crossfit" in args.suites:
            for method in methods:
                tune(method, method)
        if "robustness" in args.suites:
            ease_lam = max(args.lambdas, key=lambda lam: evaluate(EASE(lam).fit(ds.train), ds.train, ds.validation).ndcg)
            for noise_type in ["noisy_1", "noisy_0"]:
                for level in args.noise_levels:
                    noisy = inject_noise(ds.train, level, noise_type=noise_type, exclude=ds.validation+ds.test, seed=seed)
                    for method in methods+["EASE", "SVD_AE"]:
                        model = EASE(ease_lam) if method == "EASE" else SVD_AE(args.rank, seed) if method == "SVD_AE" else ResearchAE(**selected[method])
                        record(model, noisy, ds, dict(suite="robustness", model=method, seed=seed, noise_type=noise_type, noise=level))
        if "hybrid" in args.suites:
            shuffled = features[np.random.default_rng(seed+312).permutation(len(features))]
            for length in [5,10,20]:
                short = shorten(ds.train, length, seed)
                variants = [("CF unit","unit",None,args.rank), ("CF weights","gaussian",None,args.rank),
                    ("Hybrid unit","unit",features,args.rank), ("Hybrid weights","gaussian",features,args.rank),
                    ("Shuffled genres","gaussian",shuffled,args.rank), ("CF matched dimension","gaussian",None,args.rank+features.shape[1])]
                for name, method, f, rank in variants:
                    config = tune(f"{name} history={length}", method, f, rank, short)
                    record(ResearchAE(**config), short, ds, dict(suite="hybrid", model=name, seed=seed, history=length))
                record(GenreCosine(features), short, ds, dict(suite="hybrid", model="Genre cosine", seed=seed, history=length))
        noisy = inject_noise(ds.train, .1, noise_type="noisy_1", exclude=ds.validation+ds.test, seed=seed)
        if "stability" in args.suites:
            for method in ["gaussian","student","cauchy"]:
                for init, mix_seed in [("quantile",0),("random",11),("random",29)]:
                    for outer in [0,1,2,5,10]:
                        config = {**selected[method], "outer": outer, "initialization": init, "mixture_seed": mix_seed}
                        record(ResearchAE(**config), noisy, ds, dict(suite="stability",model=method,seed=seed,noise=.1,outer=outer,initialization=init,mixture_seed=mix_seed))
        if "crossfit" in args.suites:
            for method in ["gaussian","student","cauchy"]:
                for folds in [0,3]:
                    config = {**selected[method], "crossfit": folds}
                    record(ResearchAE(**config), noisy, ds, dict(suite="crossfit",model=method,seed=seed,noise=.1,folds=folds))
    manifest["completed_runs"] = len(rows)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("output/kt2/full"))
    p.add_argument("--max-users",type=int,default=0); p.add_argument("--max-items",type=int,default=0)
    p.add_argument("--rank",type=int,default=16); p.add_argument("--outer",type=int,default=2)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--lambdas",type=float,nargs="+",default=[1.,10.,100.])
    p.add_argument("--noise-levels",type=float,nargs="+",default=[0,.05,.1,.15,.2])
    p.add_argument("--suites",nargs="+",choices=["robustness","hybrid","stability","crossfit"],default=["robustness","hybrid","stability","crossfit"])
    return p

if __name__ == "__main__":
    run(parser().parse_args())
