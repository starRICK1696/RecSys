"""Sparse interaction loading, splitting, synthetic generation, and noise.

All public functions return canonical binary CSR matrices.  Noise is injected
only into the training matrix; validation/test interactions must remain clean.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

NoiseType = Literal["noisy_1", "noisy_0", "mixed"]
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def _binary_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = matrix.tocsr().astype(np.float32, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.data.fill(1.0)
    result.sort_indices()
    return result


@dataclass(frozen=True, slots=True)
class InteractionDataset:
    """Clean implicit-feedback train/validation/test split."""

    train: sp.csr_matrix
    validation: sp.csr_matrix
    test: sp.csr_matrix
    name: str = "dataset"

    def __post_init__(self) -> None:
        shape = self.train.shape
        if self.validation.shape != shape or self.test.shape != shape:
            raise ValueError("train, validation, and test must have equal shapes")
        if len(shape) != 2:
            raise ValueError("interaction matrices must be two-dimensional")

    @property
    def n_users(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.train.shape[1])


def split_interactions(
    interactions: sp.spmatrix,
    *,
    validation_per_user: int = 1,
    test_per_user: int = 1,
    seed: int = 42,
    name: str = "dataset",
) -> InteractionDataset:
    """Make a reproducible leave-N-out split, keeping at least one train item."""
    if validation_per_user < 0 or test_per_user < 0:
        raise ValueError("holdout counts must be non-negative")
    matrix = _binary_csr(interactions)
    rng = np.random.default_rng(seed)
    train_rows: list[int] = []
    train_cols: list[int] = []
    val_rows: list[int] = []
    val_cols: list[int] = []
    test_rows: list[int] = []
    test_cols: list[int] = []

    for user in range(matrix.shape[0]):
        items = matrix.indices[matrix.indptr[user] : matrix.indptr[user + 1]].copy()
        rng.shuffle(items)
        available = max(0, items.size - 1)
        n_test = min(test_per_user, available)
        available -= n_test
        n_val = min(validation_per_user, available)
        test_items = items[:n_test]
        val_items = items[n_test : n_test + n_val]
        train_items = items[n_test + n_val :]
        train_rows.extend([user] * train_items.size)
        train_cols.extend(train_items.tolist())
        val_rows.extend([user] * val_items.size)
        val_cols.extend(val_items.tolist())
        test_rows.extend([user] * test_items.size)
        test_cols.extend(test_items.tolist())

    shape = matrix.shape

    def build(rows: Sequence[int], cols: Sequence[int]) -> sp.csr_matrix:
        values = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    return InteractionDataset(
        train=build(train_rows, train_cols),
        validation=build(val_rows, val_cols),
        test=build(test_rows, test_cols),
        name=name,
    )


def generate_synthetic_graph(
    *,
    n_users: int = 2_000,
    n_items: int = 1_000,
    interactions_per_user: int = 30,
    n_topics: int = 24,
    user_concentration: float = 0.3,
    item_concentration: float = 0.05,
    seed: int = 42,
) -> sp.csr_matrix:
    """Sample implicit interactions from a Dirichlet-Categorical topic model.

    Each user has a Dirichlet-distributed preference over topics.  Each topic
    has a Dirichlet-distributed probability vector over items.  An interaction
    samples a topic from the user's distribution and then an item from that
    topic's distribution.  Repeated user-item draws are rejected, yielding
    exactly ``interactions_per_user`` distinct positives per user.  No latent
    score, dot product, or dense user-item probability matrix is computed.

    Smaller concentrations create more specialized users/topics.  This is a
    synthetic generative model, not a claim to reproduce ML-1M/Yelp statistics.
    """
    if min(n_users, n_items, interactions_per_user, n_topics) <= 0:
        raise ValueError("all graph dimensions must be positive")
    if interactions_per_user > n_items:
        raise ValueError("interactions_per_user cannot exceed n_items")
    if (
        not np.isfinite(user_concentration)
        or not np.isfinite(item_concentration)
        or user_concentration <= 0
        or item_concentration <= 0
    ):
        raise ValueError("Dirichlet concentrations must be positive")

    rng = np.random.default_rng(seed)
    topic_item_probabilities = rng.dirichlet(
        np.full(n_items, item_concentration, dtype=np.float64), size=n_topics
    )
    user_topic_probabilities = rng.dirichlet(
        np.full(n_topics, user_concentration, dtype=np.float64), size=n_users
    )
    topic_item_cdf = np.cumsum(topic_item_probabilities, axis=1)
    user_topic_cdf = np.cumsum(user_topic_probabilities, axis=1)
    topic_item_cdf[:, -1] = 1.0
    user_topic_cdf[:, -1] = 1.0

    rows = np.repeat(np.arange(n_users, dtype=np.int64), interactions_per_user)
    cols = np.empty(n_users * interactions_per_user, dtype=np.int64)
    for user in range(n_users):
        start = user * interactions_per_user
        seen = np.zeros(n_items, dtype=bool)
        chosen = 0
        proposals_seen = 0

        while chosen < interactions_per_user:
            # Draw several independent topic->item pairs at once.  Rejecting
            # duplicates implements sampling from the mixture without repeats.
            batch_size = max(32, 2 * (interactions_per_user - chosen))
            topics = np.searchsorted(
                user_topic_cdf[user], rng.random(batch_size), side="right"
            )
            proposed_items = np.empty(batch_size, dtype=np.int64)
            for topic in np.unique(topics):
                positions = np.flatnonzero(topics == topic)
                proposed_items[positions] = np.searchsorted(
                    topic_item_cdf[topic], rng.random(positions.size), side="right"
                )

            for item in proposed_items:
                if not seen[item]:
                    seen[item] = True
                    cols[start + chosen] = item
                    chosen += 1
                    if chosen == interactions_per_user:
                        break
            proposals_seen += batch_size

            if proposals_seen > max(1_000, 20 * n_items) and chosen < interactions_per_user:
                # Dense requests can have very low rejection acceptance.  The
                # exact conditional mixture below finishes without stalling.
                while chosen < interactions_per_user:
                    available = np.flatnonzero(~seen)
                    remaining_mass = topic_item_probabilities[:, available].sum(axis=1)
                    topic_weights = user_topic_probabilities[user] * remaining_mass
                    topic_weights /= topic_weights.sum()
                    topic = int(rng.choice(n_topics, p=topic_weights))
                    item_weights = topic_item_probabilities[topic, available]
                    item_weights /= item_weights.sum()
                    chosen_item = int(rng.choice(available, p=item_weights))
                    seen[chosen_item] = True
                    cols[start + chosen] = chosen_item
                    chosen += 1

    return sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )


def _iter_delimited(
    path: Path,
    *,
    delimiter: str,
    user_column: str,
    item_column: str,
    rating_column: str | None,
    min_rating: float,
) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        if delimiter == "::":
            header = stream.readline().strip().split(delimiter)
            has_header = user_column in header and item_column in header
            if not has_header:
                stream.seek(0)
                header = [user_column, item_column, rating_column or "rating", "timestamp"]
            for line in stream:
                values = line.rstrip("\n").split(delimiter)
                row = dict(zip(header, values))
                if rating_column is None or float(row[rating_column]) >= min_rating:
                    yield row[user_column], row[item_column]
            return
        reader = csv.DictReader(stream, delimiter=delimiter)
        for row in reader:
            if rating_column is None or float(row[rating_column]) >= min_rating:
                yield row[user_column], row[item_column]


def _iter_yelp_json(path: Path, min_rating: float) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if float(row.get("stars", 5.0)) >= min_rating:
                yield str(row["user_id"]), str(row["business_id"])


def load_interaction_file(
    path: str | Path,
    *,
    format: Literal["csv", "tsv", "ml1m", "yelp"] = "csv",
    user_column: str = "user_id",
    item_column: str = "item_id",
    rating_column: str | None = "rating",
    min_rating: float = 0.0,
) -> sp.csr_matrix:
    """Load CSV/TSV, MovieLens ``ratings.dat``, or Yelp review JSONL."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if format == "yelp":
        pairs = _iter_yelp_json(source, min_rating)
    else:
        delimiter = {"csv": ",", "tsv": "\t", "ml1m": "::"}[format]
        if format == "ml1m":
            user_column, item_column, rating_column = "user_id", "item_id", "rating"
        pairs = _iter_delimited(
            source,
            delimiter=delimiter,
            user_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
            min_rating=min_rating,
        )

    user_map: dict[str, int] = {}
    item_map: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for raw_user, raw_item in pairs:
        rows.append(user_map.setdefault(raw_user, len(user_map)))
        cols.append(item_map.setdefault(raw_item, len(item_map)))
    if not rows:
        raise ValueError(f"no interactions passed the filter in {source}")
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_map), len(item_map)),
    )
    return _binary_csr(matrix)


def load_dataset(
    path: str | Path | None = None,
    *,
    format: Literal["synthetic", "npz", "csv", "tsv", "ml1m", "yelp"] = "synthetic",
    seed: int = 42,
    n_users: int = 2_000,
    n_items: int = 1_000,
    interactions_per_user: int = 30,
    n_topics: int = 24,
    user_concentration: float = 0.3,
    item_concentration: float = 0.05,
    min_rating: float = 0.0,
) -> InteractionDataset:
    """Unified loader used by both scripts and notebooks."""
    if format == "synthetic":
        matrix = generate_synthetic_graph(
            n_users=n_users,
            n_items=n_items,
            interactions_per_user=interactions_per_user,
            n_topics=n_topics,
            user_concentration=user_concentration,
            item_concentration=item_concentration,
            seed=seed,
        )
        name = "synthetic_dirichlet"
    elif format == "npz":
        if path is None:
            raise ValueError("path is required for npz data")
        matrix = _binary_csr(sp.load_npz(path))
        name = Path(path).stem
    else:
        if path is None:
            raise ValueError(f"path is required for {format} data")
        matrix = load_interaction_file(path, format=format, min_rating=min_rating)
        name = Path(path).stem
    return split_interactions(matrix, seed=seed, name=name)


def _sample_unobserved(
    matrix: sp.csr_matrix,
    count: int,
    rng: np.random.Generator,
    *,
    forbidden: sp.csr_matrix | None = None,
) -> tuple[IntArray, IntArray]:
    n_users, n_items = matrix.shape
    universe = n_users * n_items
    blocked_matrix = matrix if forbidden is None else forbidden
    blocked = np.sort(
        np.repeat(np.arange(n_users, dtype=np.int64), np.diff(blocked_matrix.indptr))
        * n_items
        + blocked_matrix.indices.astype(np.int64, copy=False)
    )
    available = universe - blocked.size
    if count > available:
        raise ValueError(f"cannot draw {count} zeros; only {available} are available")
    selected: set[int] = set()
    while len(selected) < count:
        needed = count - len(selected)
        proposals = rng.integers(0, universe, size=max(64, needed * 2), dtype=np.int64)
        positions = np.searchsorted(blocked, proposals)
        positions = np.minimum(positions, blocked.size - 1) if blocked.size else positions
        is_blocked = np.zeros(proposals.size, dtype=bool)
        if blocked.size:
            is_blocked = blocked[positions] == proposals
        selected.update(int(value) for value in proposals[~is_blocked])
    linear = np.fromiter(selected, dtype=np.int64, count=len(selected))[:count]
    return linear // n_items, linear % n_items


def inject_noise(
    interactions: sp.spmatrix,
    noise_ratio: float,
    *,
    noise_type: NoiseType = "mixed",
    exclude: sp.spmatrix | None = None,
    seed: int = 42,
) -> sp.csr_matrix:
    """Inject SVD-AE-style false positives (Noisy 1) or false negatives.

    ``noise_ratio`` is measured against the number of original nonzeros.  For
    ``mixed``, half of that budget is assigned to each noise type.  False
    positives are sampled only from zeros in the *original* matrix and, if
    provided, from outside ``exclude`` (typically validation + test).
    """
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError("noise_ratio must lie in [0, 1]")
    if noise_type not in {"noisy_1", "noisy_0", "mixed"}:
        raise ValueError(f"unknown noise type: {noise_type}")
    clean = _binary_csr(interactions)
    blocked = clean
    if exclude is not None:
        excluded = _binary_csr(exclude)
        if excluded.shape != clean.shape:
            raise ValueError("exclude must have the same shape as interactions")
        blocked = _binary_csr(clean + excluded)
    if noise_ratio == 0.0 or clean.nnz == 0:
        return clean.copy()
    rng = np.random.default_rng(seed)
    budget = min(clean.nnz, int(round(noise_ratio * clean.nnz)))
    n_remove = budget if noise_type == "noisy_0" else (budget // 2 if noise_type == "mixed" else 0)
    n_add = budget if noise_type == "noisy_1" else (budget - n_remove if noise_type == "mixed" else 0)

    noisy = clean.copy()
    if n_remove:
        removed = rng.choice(noisy.nnz, size=n_remove, replace=False)
        noisy.data[removed] = 0.0
        noisy.eliminate_zeros()
    if n_add:
        rows, cols = _sample_unobserved(noisy, n_add, rng, forbidden=blocked)
        additions = sp.csr_matrix(
            (np.ones(n_add, dtype=np.float32), (rows, cols)), shape=clean.shape
        )
        noisy = _binary_csr(noisy + additions)
    return noisy
