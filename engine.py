"""Training timers, full-ranking Recall/NDCG, and experiment utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from .models import Recommender


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    recall: float
    ndcg: float
    users_evaluated: int
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class RunResult:
    dataset: str
    model: str
    noise_type: str
    noise_ratio: float
    train_seconds: float
    inference_seconds: float
    recall_at_20: float
    ndcg_at_20: float
    users_evaluated: int
    train_nnz: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mask_seen(scores: npt.NDArray[np.float32], seen: sp.csr_matrix) -> None:
    for row in range(seen.shape[0]):
        start, end = seen.indptr[row : row + 2]
        scores[row, seen.indices[start:end]] = -np.inf


def evaluate(
    model: Recommender,
    train: sp.csr_matrix,
    heldout: sp.csr_matrix,
    *,
    k: int = 20,
    batch_size: int = 512,
) -> EvaluationResult:
    """Evaluate full-catalog ranking while masking observed training items."""
    if train.shape != heldout.shape:
        raise ValueError("train and heldout shapes differ")
    if k <= 0 or batch_size <= 0:
        raise ValueError("k and batch_size must be positive")
    top_k = min(k, train.shape[1])
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2, dtype=np.float64))
    recall_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0
    started = perf_counter()

    for start in range(0, train.shape[0], batch_size):
        end = min(start + batch_size, train.shape[0])
        users = np.arange(start, end, dtype=np.int64)
        scores = np.asarray(model.score(users, train), dtype=np.float32)
        if scores.shape != (users.size, train.shape[1]):
            raise ValueError(f"model returned scores with invalid shape {scores.shape}")
        _mask_seen(scores, train[start:end])
        candidates = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        ranking = np.take_along_axis(candidates, order, axis=1)

        target_batch = heldout[start:end]
        for local_user in range(users.size):
            left, right = target_batch.indptr[local_user : local_user + 2]
            relevant = target_batch.indices[left:right]
            if relevant.size == 0:
                continue
            hits = np.isin(ranking[local_user], relevant, assume_unique=False)
            recall_sum += float(hits.sum()) / relevant.size
            dcg = float(np.dot(hits, discounts))
            ideal = float(discounts[: min(relevant.size, top_k)].sum())
            ndcg_sum += dcg / ideal
            evaluated += 1

    elapsed = perf_counter() - started
    if evaluated == 0:
        raise ValueError("heldout matrix has no evaluable users")
    return EvaluationResult(
        recall=recall_sum / evaluated,
        ndcg=ndcg_sum / evaluated,
        users_evaluated=evaluated,
        inference_seconds=elapsed,
    )


def fit_and_evaluate(
    model: Recommender,
    train: sp.csr_matrix,
    test: sp.csr_matrix,
    *,
    dataset: str,
    model_name: str,
    noise_type: str,
    noise_ratio: float,
    k: int = 20,
    batch_size: int = 512,
) -> RunResult:
    started = perf_counter()
    model.fit(train)
    train_seconds = perf_counter() - started
    metrics = evaluate(model, train, test, k=k, batch_size=batch_size)
    return RunResult(
        dataset=dataset,
        model=model_name,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        train_seconds=train_seconds,
        inference_seconds=metrics.inference_seconds,
        recall_at_20=metrics.recall,
        ndcg_at_20=metrics.ndcg,
        users_evaluated=metrics.users_evaluated,
        train_nnz=train.nnz,
    )
