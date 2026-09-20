"""Run robustness experiments for ProbAE, SVD-AE, EASE, and MultVAE."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Sequence, cast

import numpy as np
import torch

from .data import NoiseType, inject_noise, load_dataset
from .engine import RunResult, fit_and_evaluate
from .models import EASE, MultVAE, ProbAE, Recommender, SVD_AE


def _float_grid(value: str) -> list[float]:
    values = [float(part) for part in value.split(",")]
    if not values or any(level < 0.0 or level > 1.0 for level in values):
        raise argparse.ArgumentTypeError("noise levels must be comma-separated values in [0,1]")
    return values


def _string_grid(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--format", choices=["synthetic", "npz", "csv", "tsv", "ml1m", "yelp"], default="synthetic"
    )
    parser.add_argument("--output", type=Path, default=Path("results/robustness.csv"))
    parser.add_argument("--noise-levels", type=_float_grid, default=_float_grid("0,0.05,0.10,0.15,0.20"))
    parser.add_argument("--noise-types", type=_string_grid, default=_string_grid("noisy_1,noisy_0"))
    parser.add_argument("--models", type=_string_grid, default=_string_grid("ProbAE,SVD_AE,EASE,MultVAE"))
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=100.0)
    parser.add_argument("--prob-regularization", type=float, default=10.0)
    parser.add_argument("--em-iterations", type=int, default=5)
    parser.add_argument("--gmm-components", type=int, default=2)
    parser.add_argument("--signal-components", type=int, default=1)
    parser.add_argument("--gmm-iterations", type=int, default=30)
    parser.add_argument("--zero-weight", type=float, default=0.01)
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=2_000)
    parser.add_argument("--n-items", type=int, default=1_000)
    parser.add_argument("--interactions-per-user", type=int, default=30)
    parser.add_argument("--n-topics", type=int, default=24)
    parser.add_argument("--user-concentration", type=float, default=0.3)
    parser.add_argument("--item-concentration", type=float, default=0.05)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    return parser


def _factories(args: argparse.Namespace) -> dict[str, Callable[[], Recommender]]:
    return {
        "ProbAE": lambda: ProbAE(
            rank=args.rank,
            regularization=args.prob_regularization,
            em_iterations=args.em_iterations,
            gmm_components=args.gmm_components,
            signal_components=args.signal_components,
            gmm_iterations=args.gmm_iterations,
            zero_weight=args.zero_weight,
            seed=args.seed,
        ),
        "SVD_AE": lambda: SVD_AE(rank=args.rank, seed=args.seed),
        "EASE": lambda: EASE(regularization=args.regularization),
        "MultVAE": lambda: MultVAE(
            latent_dim=args.rank,
            epochs=args.vae_epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
    }


def _write_results(path: Path, results: Sequence[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = [result.as_dict() for result in results]
    if not rows:
        return
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> list[RunResult]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_dataset(
        args.data,
        format=args.format,
        seed=args.seed,
        n_users=args.n_users,
        n_items=args.n_items,
        interactions_per_user=args.interactions_per_user,
        n_topics=args.n_topics,
        user_concentration=args.user_concentration,
        item_concentration=args.item_concentration,
        min_rating=args.min_rating,
    )
    factories = _factories(args)
    unknown = set(args.models) - factories.keys()
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    unknown_noise = set(args.noise_types) - {"noisy_1", "noisy_0", "mixed"}
    if unknown_noise:
        raise ValueError(f"unknown noise types: {sorted(unknown_noise)}")

    print(
        json.dumps(
            {
                "dataset": dataset.name,
                "shape": dataset.train.shape,
                "train_nnz": dataset.train.nnz,
                "validation_nnz": dataset.validation.nnz,
                "test_nnz": dataset.test.nnz,
            }
        )
    )
    results: list[RunResult] = []
    for noise_type_string in args.noise_types:
        noise_type = cast(NoiseType, noise_type_string)
        for level in args.noise_levels:
            noisy_train = inject_noise(
                dataset.train,
                level,
                noise_type=noise_type,
                exclude=dataset.validation + dataset.test,
                seed=args.seed,
            )
            for model_name in args.models:
                print(f"running {model_name}: {noise_type}={level:.3f}", flush=True)
                result = fit_and_evaluate(
                    factories[model_name](),
                    noisy_train,
                    dataset.test,
                    dataset=dataset.name,
                    model_name=model_name,
                    noise_type=noise_type,
                    noise_ratio=level,
                    k=args.top_k,
                    batch_size=args.batch_size,
                )
                results.append(result)
                _write_results(args.output, results)
                print(json.dumps(result.as_dict()), flush=True)
    return results


if __name__ == "__main__":
    run(build_parser().parse_args())
