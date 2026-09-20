"""Validation-only hyperparameter search for ProbAE on synthetic data.

The objective is mean Recall@20 across clean, 10% Noisy 1, and 10% Noisy 0
training matrices.  Test metrics are computed only after model selection.
This is a bounded, reproducible search, not a claim of a global optimum.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from .data import InteractionDataset, inject_noise, load_dataset
from .engine import EvaluationResult, evaluate
from .models import ProbAE


@dataclass(frozen=True, slots=True)
class TrialConfig:
    rank: int
    regularization: float
    zero_weight: float
    gmm_components: int
    signal_components: int
    em_iterations: int

    def create_model(self, *, seed: int) -> ProbAE:
        return ProbAE(
            rank=self.rank,
            regularization=self.regularization,
            zero_weight=self.zero_weight,
            gmm_components=self.gmm_components,
            signal_components=self.signal_components,
            em_iterations=self.em_iterations,
            seed=seed,
        )


def candidate_configs(n_trials: int, *, seed: int) -> list[TrialConfig]:
    """Stratified random sample so each K in {1,2,3,4} is compared."""
    if n_trials < 4:
        raise ValueError("n_trials must be at least 4 to cover all GMM sizes")
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[TrialConfig]] = {}
    baseline = TrialConfig(64, 10.0, 0.01, 2, 1, 5)
    for components in (1, 2, 3, 4):
        signal_options = (1,) if components <= 2 else (1, 2)
        em_options = (0,) if components == 1 else (2, 5)
        options = [
            TrialConfig(rank, regularization, zero_weight, components, signal, em)
            for rank in (16, 32, 64)
            for regularization in (0.1, 1.0, 10.0, 100.0)
            for zero_weight in (0.001, 0.01, 0.05)
            for signal in signal_options
            for em in em_options
        ]
        quota = n_trials // 4 + int(components <= n_trials % 4)
        if quota > len(options):
            raise ValueError("n_trials exceeds the finite candidate space")
        selected: list[TrialConfig] = []
        if components == 2:
            selected.append(baseline)
        for index in rng.permutation(len(options)):
            if len(selected) >= quota:
                break
            candidate = options[int(index)]
            if candidate not in selected:
                selected.append(candidate)
        buckets[components] = selected

    interleaved: list[TrialConfig] = []
    for position in range(max(map(len, buckets.values()))):
        for components in (1, 2, 3, 4):
            if position < len(buckets[components]):
                interleaved.append(buckets[components][position])
    return interleaved


def _scenarios(
    dataset: InteractionDataset, *, noise_ratio: float, seed: int
) -> dict[str, sp.csr_matrix]:
    exclude = dataset.validation + dataset.test
    return {
        "clean": dataset.train,
        "noisy_1": inject_noise(
            dataset.train, noise_ratio, noise_type="noisy_1", exclude=exclude, seed=seed
        ),
        "noisy_0": inject_noise(
            dataset.train, noise_ratio, noise_type="noisy_0", exclude=exclude, seed=seed
        ),
    }


def _evaluate_config(
    config: TrialConfig,
    scenarios: dict[str, sp.csr_matrix],
    heldout: sp.csr_matrix,
    *,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, EvaluationResult], float]:
    results: dict[str, EvaluationResult] = {}
    started = perf_counter()
    for name, train in scenarios.items():
        model = config.create_model(seed=seed)
        model.fit(train)
        results[name] = evaluate(model, train, heldout, k=20, batch_size=batch_size)
    return results, perf_counter() - started


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-ratio", type=float, default=0.10)
    parser.add_argument("--n-users", type=int, default=2_000)
    parser.add_argument("--n-items", type=int, default=1_000)
    parser.add_argument("--interactions-per-user", type=int, default=30)
    parser.add_argument("--n-topics", type=int, default=24)
    parser.add_argument("--user-concentration", type=float, default=0.3)
    parser.add_argument("--item-concentration", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output", type=Path, default=Path("RecSys/results/probae_tuning_dirichlet.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("RecSys/results/probae_tuning_dirichlet.json")
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 < args.noise_ratio <= 1:
        raise ValueError("noise_ratio must be in (0, 1]")
    dataset = load_dataset(
        format="synthetic",
        seed=args.seed,
        n_users=args.n_users,
        n_items=args.n_items,
        interactions_per_user=args.interactions_per_user,
        n_topics=args.n_topics,
        user_concentration=args.user_concentration,
        item_concentration=args.item_concentration,
    )
    scenarios = _scenarios(dataset, noise_ratio=args.noise_ratio, seed=args.seed)
    configs = candidate_configs(args.trials, seed=args.seed)
    trial_rows: list[dict[str, Any]] = []
    best_config: TrialConfig | None = None
    best_key = (-float("inf"), -float("inf"), -float("inf"))

    for number, config in enumerate(configs, start=1):
        metrics, seconds = _evaluate_config(
            config, scenarios, dataset.validation, seed=args.seed, batch_size=args.batch_size
        )
        mean_recall = float(np.mean([value.recall for value in metrics.values()]))
        mean_ndcg = float(np.mean([value.ndcg for value in metrics.values()]))
        record: dict[str, Any] = {
            "trial": number,
            **asdict(config),
            "validation_mean_recall_at_20": mean_recall,
            "validation_mean_ndcg_at_20": mean_ndcg,
            "validation_clean_recall_at_20": metrics["clean"].recall,
            "validation_noisy_1_recall_at_20": metrics["noisy_1"].recall,
            "validation_noisy_0_recall_at_20": metrics["noisy_0"].recall,
            "trial_seconds": seconds,
        }
        trial_rows.append(record)
        _write_csv(args.output, trial_rows)
        key = (mean_recall, mean_ndcg, -seconds)
        if key > best_key:
            best_key = key
            best_config = config
        print(
            f"{number:02d}/{len(configs)} K={config.gmm_components} "
            f"rank={config.rank} recall={mean_recall:.4f} ndcg={mean_ndcg:.4f} "
            f"time={seconds:.1f}s",
            flush=True,
        )

    if best_config is None:
        raise RuntimeError("candidate search produced no models")
    test_metrics, test_seconds = _evaluate_config(
        best_config, scenarios, dataset.test, seed=args.seed, batch_size=args.batch_size
    )
    summary: dict[str, Any] = {
        "selection_protocol": "mean validation Recall@20 across clean, Noisy 1, Noisy 0; tie-break by NDCG@20",
        "search_is_global_optimum": False,
        "seed": args.seed,
        "synthetic_shape": [dataset.n_users, dataset.n_items],
        "synthetic_parameters": {
            "interactions_per_user": args.interactions_per_user,
            "n_topics": args.n_topics,
            "user_concentration": args.user_concentration,
            "item_concentration": args.item_concentration,
        },
        "train_nnz": dataset.train.nnz,
        "noise_ratio": args.noise_ratio,
        "trials": len(configs),
        "search_space": {
            "gmm_components": [1, 2, 3, 4],
            "signal_components": "1 for K<=2; 1 or 2 for K>=3",
            "rank": [16, 32, 64],
            "regularization": [0.1, 1.0, 10.0, 100.0],
            "zero_weight": [0.001, 0.01, 0.05],
            "em_iterations": "0 for K=1; 2 or 5 for K>=2",
        },
        "fixed_parameters": {
            "encoder": "random_projection",
            "min_confidence": 0.02,
            "gmm_iterations": 30,
            "zero_diagonal": True,
        },
        "best_config": asdict(best_config),
        "best_validation_mean_recall_at_20": best_key[0],
        "best_validation_mean_ndcg_at_20": best_key[1],
        "test_by_scenario": {
            name: {"recall_at_20": value.recall, "ndcg_at_20": value.ndcg}
            for name, value in test_metrics.items()
        },
        "test_evaluation_seconds": test_seconds,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(build_parser().parse_args())
