"""Prob-AE and three collaborative-filtering baselines.

ProbAE uses a fixed linear encoder and an exact confidence-weighted ridge
decoder.  For arbitrary element-wise weights, the decoder has one closed-form
system per target item; there is no mathematically valid single shared inverse.
The low-dimensional encoder keeps those systems tractable on sparse datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import scipy.sparse as sp
from scipy.special import logsumexp
from scipy.sparse.linalg import svds
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Self

FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


def _as_binary_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = matrix.tocsr().astype(np.float32, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.data.fill(1.0)
    result.sort_indices()
    return result


class Recommender(Protocol):
    def fit(self, interactions: sp.csr_matrix) -> Self: ...

    def score(self, users: npt.NDArray[np.int64], interactions: sp.csr_matrix) -> FloatArray: ...


class EASE:
    """Embarrassingly Shallow Autoencoder with an exact zero diagonal."""

    def __init__(self, regularization: float = 300.0, dtype: npt.DTypeLike = np.float64) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = float(regularization)
        self.dtype = np.dtype(dtype)
        self.coef_: FloatArray | None = None

    def fit(self, interactions: sp.csr_matrix) -> Self:
        matrix = _as_binary_csr(interactions).astype(self.dtype)
        gram = (matrix.T @ matrix).toarray()
        gram.flat[:: gram.shape[0] + 1] += self.regularization
        factor = la.cho_factor(gram, lower=True, check_finite=False)
        precision = la.cho_solve(factor, np.eye(gram.shape[0], dtype=self.dtype), check_finite=False)
        diagonal = np.diag(precision)
        coefficients = -precision / diagonal[np.newaxis, :]
        np.fill_diagonal(coefficients, 0.0)
        self.coef_ = coefficients.astype(np.float32, copy=False)
        return self

    def score(self, users: npt.NDArray[np.int64], interactions: sp.csr_matrix) -> FloatArray:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before score")
        return np.asarray(interactions[users] @ self.coef_, dtype=np.float32)


class SVD_AE:
    """SVD-AE from Hong et al. (IJCAI 2024), evaluated without dense R-hat.

    If ``R_tilde = Q S V^T``, Eq. 15 simplifies to ``R_hat = Q Q^T R``.
    """

    def __init__(self, rank: int = 64, seed: int = 42) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.seed = seed
        self.user_factors_: FloatArray | None = None
        self.projected_items_: FloatArray | None = None

    def fit(self, interactions: sp.csr_matrix) -> Self:
        matrix = _as_binary_csr(interactions)
        max_rank = min(matrix.shape) - 1
        if max_rank < 1:
            raise ValueError("SVD_AE needs at least two users and two items")
        rank = min(self.rank, max_rank)
        user_degree = np.asarray(matrix.sum(axis=1)).ravel()
        item_degree = np.asarray(matrix.sum(axis=0)).ravel()
        user_scale = np.zeros_like(user_degree, dtype=np.float32)
        item_scale = np.zeros_like(item_degree, dtype=np.float32)
        np.power(user_degree, -0.5, out=user_scale, where=user_degree > 0)
        np.power(item_degree, -0.5, out=item_scale, where=item_degree > 0)
        normalized = sp.diags(user_scale) @ matrix @ sp.diags(item_scale)
        try:
            left, singular, _ = svds(
                normalized.astype(np.float64), k=rank, which="LM", rng=np.random.default_rng(self.seed)
            )
        except TypeError:  # SciPy < 1.15 uses random_state.
            left, singular, _ = svds(
                normalized.astype(np.float64), k=rank, which="LM", random_state=self.seed
            )
        order = np.argsort(singular)[::-1]
        left = left[:, order].astype(np.float32, copy=False)
        self.user_factors_ = left
        self.projected_items_ = np.asarray(left.T @ matrix, dtype=np.float32)
        return self

    def score(self, users: npt.NDArray[np.int64], interactions: sp.csr_matrix) -> FloatArray:
        del interactions
        if self.user_factors_ is None or self.projected_items_ is None:
            raise RuntimeError("fit must be called before score")
        return np.asarray(self.user_factors_[users] @ self.projected_items_, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class GaussianMixtureState:
    means: tuple[float, ...]
    variances: tuple[float, ...]
    mixing: tuple[float, ...]
    signal_components: int
    mean_confidence: float


def signal_responsibilities(
    errors: npt.ArrayLike,
    *,
    n_components: int = 2,
    signal_components: int = 1,
    iterations: int = 30,
    min_confidence: float = 0.02,
) -> tuple[FloatArray, GaussianMixtureState]:
    """Fit a K-Gaussian mixture to log errors and return signal posteriors.

    The ``signal_components`` components with the smallest fitted means are
    treated as genuine interactions.  K=1 is an unweighted ridge ablation.
    """
    if n_components < 1 or not 1 <= signal_components <= n_components:
        raise ValueError("require n_components >= 1 and 1 <= signal_components <= n_components")
    if iterations < 1 or not 0.0 < min_confidence <= 1.0:
        raise ValueError("iterations must be positive and min_confidence in (0, 1]")
    raw = np.asarray(errors, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("errors must be a non-empty vector")
    if n_components > raw.size:
        raise ValueError("n_components cannot exceed the number of observations")
    if np.any(raw < 0) or not np.all(np.isfinite(raw)):
        raise ValueError("errors must be finite and non-negative")
    values = np.log(np.maximum(raw, 1e-8))
    variance = max(float(np.var(values)), 1e-4)
    if n_components == 1 or variance <= 1e-4:
        confidence = np.ones(values.size, dtype=np.float32)
        means = np.full(n_components, float(values.mean()), dtype=np.float64)
        variances = np.full(n_components, variance, dtype=np.float64)
        mixing = np.full(n_components, 1.0 / n_components, dtype=np.float64)
    else:
        quantiles = (np.arange(n_components, dtype=np.float64) + 0.5) / n_components
        means = np.asarray(np.quantile(values, quantiles), dtype=np.float64)
        variances = np.full(n_components, variance, dtype=np.float64)
        mixing = np.full(n_components, 1.0 / n_components, dtype=np.float64)

        def posterior() -> Float64Array:
            log_prob = (
                np.log(np.maximum(mixing, 1e-12))[None, :]
                - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
                - 0.5 * (values[:, None] - means[None, :]) ** 2 / variances[None, :]
            )
            return np.asarray(np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True)))

        for _ in range(iterations):
            responsibilities = posterior()
            mass = np.maximum(responsibilities.sum(axis=0), 1e-8)
            means = (responsibilities * values[:, None]).sum(axis=0) / mass
            variances = np.maximum(
                (
                    responsibilities * (values[:, None] - means[None, :]) ** 2
                ).sum(axis=0) / mass,
                1e-4,
            )
            mixing = np.maximum(mass / values.size, 1e-8)
            mixing /= mixing.sum()

        responsibilities = posterior()
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        mixing = mixing[order]
        confidence = np.clip(
            responsibilities[:, order[:signal_components]].sum(axis=1),
            min_confidence,
            1.0,
        ).astype(np.float32)
    state = GaussianMixtureState(
        means=tuple(float(value) for value in means),
        variances=tuple(float(value) for value in variances),
        mixing=tuple(float(value) for value in mixing),
        signal_components=signal_components,
        mean_confidence=float(confidence.mean()),
    )
    return confidence, state


class ProbAE:
    """EM probabilistic-weighting autoencoder with a closed-form M-step.

    The default fixed random projection is a computational sketch, not a
    spectral noise filter.  Set ``encoder='svd'`` for an ablation with a
    data-dependent encoder.  Missing entries carry ``zero_weight``; without it,
    positive-only implicit reconstruction has a degenerate all-positive fit.
    """

    def __init__(
        self,
        rank: int = 64,
        regularization: float = 10.0,
        em_iterations: int = 5,
        zero_weight: float = 0.01,
        min_confidence: float = 0.02,
        gmm_components: int = 2,
        signal_components: int = 1,
        gmm_iterations: int = 30,
        encoder: Literal["random_projection", "svd"] = "random_projection",
        zero_diagonal: bool = True,
        seed: int = 42,
        tolerance: float = 1e-3,
    ) -> None:
        if rank <= 0 or regularization <= 0 or em_iterations < 0:
            raise ValueError("rank/regularization must be positive and EM iterations non-negative")
        if not 0.0 < zero_weight < 1.0:
            raise ValueError("zero_weight must lie in (0, 1)")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in (0, 1]")
        if gmm_components < 1 or not 1 <= signal_components <= gmm_components:
            raise ValueError("require gmm_components >= 1 and 1 <= signal_components <= gmm_components")
        if gmm_iterations < 1:
            raise ValueError("gmm_iterations must be positive")
        self.rank = rank
        self.regularization = regularization
        self.em_iterations = em_iterations
        self.zero_weight = zero_weight
        self.min_confidence = min_confidence
        self.gmm_components = gmm_components
        self.signal_components = signal_components
        self.gmm_iterations = gmm_iterations
        self.encoder = encoder
        self.zero_diagonal = zero_diagonal
        self.seed = seed
        self.tolerance = tolerance
        self.projection_: FloatArray | None = None
        self.design_: FloatArray | None = None
        self.decoder_: FloatArray | None = None
        self.confidence_: FloatArray | None = None
        self.history_: list[GaussianMixtureState] = []

    def _make_design(self, matrix: sp.csr_matrix) -> FloatArray:
        rank = min(self.rank, min(matrix.shape) - 1)
        if rank < 1:
            raise ValueError("ProbAE needs at least two users and two items")
        if self.encoder == "svd":
            try:
                left, singular, right_t = svds(
                    matrix.astype(np.float64), k=rank, which="LM", rng=np.random.default_rng(self.seed)
                )
            except TypeError:
                left, singular, right_t = svds(
                    matrix.astype(np.float64), k=rank, which="LM", random_state=self.seed
                )
            order = np.argsort(singular)[::-1]
            features = left[:, order] * singular[order][None, :]
            projection = right_t[order].T
        else:
            rng = np.random.default_rng(self.seed)
            projection = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32), size=(matrix.shape[1], rank)
            ) / np.sqrt(rank)
            features = matrix @ projection
        features = np.asarray(features, dtype=np.float32)
        scale = np.sqrt(np.maximum(np.mean(features * features, axis=0), 1e-8))
        features /= scale[None, :]
        self.projection_ = np.asarray(projection / scale[None, :], dtype=np.float32)
        return np.column_stack(
            [features, np.ones(features.shape[0], dtype=np.float32)]
        ).astype(np.float32, copy=False)

    def _m_step(
        self,
        matrix: sp.csr_matrix,
        confidence: FloatArray,
        design: FloatArray,
    ) -> FloatArray:
        confidence_csr = sp.csr_matrix(
            (confidence, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
        )
        matrix_csc = matrix.tocsc()
        confidence_csc = confidence_csr.tocsc()
        dimension = design.shape[1]
        base = self.zero_weight * (design.T @ design)
        base.flat[:: dimension + 1] += self.regularization
        decoder = np.empty((dimension, matrix.shape[1]), dtype=np.float32)

        for item in range(matrix.shape[1]):
            start, end = matrix_csc.indptr[item : item + 2]
            users = matrix_csc.indices[start:end]
            weights = confidence_csc.data[start:end].astype(np.float32, copy=False)
            if users.size:
                local = design[users]
                delta = weights - self.zero_weight
                system = base + local.T @ (delta[:, None] * local)
                target = local.T @ weights
            else:
                system = base
                target = np.zeros(dimension, dtype=np.float32)
            try:
                factor = la.cho_factor(system, lower=True, check_finite=False)
                if self.zero_diagonal:
                    if self.projection_ is None:
                        raise RuntimeError("encoder projection was not initialized")
                    constraint = np.r_[self.projection_[item], np.float32(0.0)]
                    solved = la.cho_solve(
                        factor, np.column_stack([target, constraint]), check_finite=False
                    )
                    solution = solved[:, 0]
                    direction = solved[:, 1]
                    denominator = float(constraint @ direction)
                    if denominator > 1e-12:
                        solution -= direction * float(constraint @ solution) / denominator
                else:
                    solution = la.cho_solve(factor, target, check_finite=False)
            except la.LinAlgError:
                solution = la.solve(system, target, assume_a="sym", check_finite=False)
            decoder[:, item] = solution.astype(np.float32, copy=False)
        return decoder

    @staticmethod
    def _observed_scores(
        matrix: sp.csr_matrix, design: FloatArray, decoder: FloatArray, chunk_size: int = 1_000_000
    ) -> FloatArray:
        rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
        result = np.empty(matrix.nnz, dtype=np.float32)
        for start in range(0, matrix.nnz, chunk_size):
            end = min(start + chunk_size, matrix.nnz)
            result[start:end] = np.einsum(
                "ij,ji->i",
                design[rows[start:end]],
                decoder[:, matrix.indices[start:end]],
                optimize=True,
            )
        return result

    def fit(self, interactions: sp.csr_matrix) -> Self:
        matrix = _as_binary_csr(interactions)
        design = self._make_design(matrix)
        confidence = np.ones(matrix.nnz, dtype=np.float32)
        decoder = self._m_step(matrix, confidence, design)
        self.history_ = []
        for _ in range(self.em_iterations):
            observed = self._observed_scores(matrix, design, decoder)
            squared_error = np.square(1.0 - observed).astype(np.float64)
            new_confidence, state = signal_responsibilities(
                squared_error,
                n_components=self.gmm_components,
                signal_components=self.signal_components,
                iterations=self.gmm_iterations,
                min_confidence=self.min_confidence,
            )
            self.history_.append(state)
            change = float(np.mean(np.abs(new_confidence - confidence)))
            confidence = new_confidence
            decoder = self._m_step(matrix, confidence, design)
            if change < self.tolerance:
                break
        self.design_ = design
        self.decoder_ = decoder
        self.confidence_ = confidence
        return self

    def _design_for(self, users: npt.NDArray[np.int64]) -> FloatArray:
        if self.design_ is None:
            raise RuntimeError("fit must be called before score")
        # Normal evaluation addresses the training users and reuses their exact
        # standardized representation.  This also avoids train/test leakage.
        return self.design_[users]

    def score(self, users: npt.NDArray[np.int64], interactions: sp.csr_matrix) -> FloatArray:
        del interactions
        if self.decoder_ is None:
            raise RuntimeError("fit must be called before score")
        design = self._design_for(users)
        return np.asarray(design @ self.decoder_, dtype=np.float32)


class _SparseRows(Dataset[Tensor]):
    def __init__(self, matrix: sp.csr_matrix) -> None:
        self.matrix = matrix

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> Tensor:
        return torch.from_numpy(self.matrix[index].toarray().ravel().astype(np.float32))


class _MultVAENetwork(nn.Module):
    def __init__(self, n_items: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(nn.Linear(n_items, hidden_dim), nn.Tanh())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_items)

    def forward(self, values: Tensor, *, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        normalized = nn.functional.normalize(values, p=2, dim=1)
        hidden = self.encoder(self.dropout(normalized))
        mean = self.mu(hidden)
        log_variance = self.log_variance(hidden).clamp(-10.0, 10.0)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return self.decoder(latent), mean, log_variance


class MultVAE:
    """Mult-VAE baseline with multinomial likelihood and KL annealing."""

    def __init__(
        self,
        hidden_dim: int = 200,
        latent_dim: int = 64,
        dropout: float = 0.5,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        anneal_steps: int = 20_000,
        max_kl_weight: float = 0.2,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.anneal_steps = anneal_steps
        self.max_kl_weight = max_kl_weight
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        self.network_: _MultVAENetwork | None = None
        self.loss_history_: list[float] = []

    def fit(self, interactions: sp.csr_matrix) -> Self:
        matrix = _as_binary_csr(interactions)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        network = _MultVAENetwork(
            matrix.shape[1], self.hidden_dim, self.latent_dim, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            _SparseRows(matrix), batch_size=self.batch_size, shuffle=True, generator=generator
        )
        updates = 0
        self.loss_history_ = []
        for _ in range(self.epochs):
            network.train()
            total_loss = 0.0
            examples = 0
            for batch in loader:
                values = batch.to(self.device, non_blocking=True)
                logits, mean, log_variance = network(values, sample=True)
                negative_log_likelihood = -(
                    nn.functional.log_softmax(logits, dim=1) * values
                ).sum(dim=1).mean()
                kl = -0.5 * torch.sum(
                    1.0 + log_variance - mean.square() - log_variance.exp(), dim=1
                ).mean()
                anneal = min(self.max_kl_weight, updates / max(1, self.anneal_steps))
                loss = negative_log_likelihood + anneal * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * values.shape[0]
                examples += values.shape[0]
                updates += 1
            self.loss_history_.append(total_loss / max(1, examples))
        self.network_ = network
        return self

    def score(self, users: npt.NDArray[np.int64], interactions: sp.csr_matrix) -> FloatArray:
        if self.network_ is None:
            raise RuntimeError("fit must be called before score")
        values = torch.from_numpy(
            interactions[users].toarray().astype(np.float32, copy=False)
        ).to(self.device)
        self.network_.eval()
        with torch.inference_mode():
            logits, _, _ = self.network_(values, sample=False)
        result = cast(npt.NDArray[np.float32], logits.cpu().numpy())
        return result.astype(np.float32, copy=False)
