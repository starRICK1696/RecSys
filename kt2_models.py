"""Controlled error-weighting experiments; target-excluded linear decoder.

Heavy tails describe log squared errors, not the decoder residual likelihood.
The outer reweighting loop is not claimed to maximize a joint EM objective.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln, logsumexp
from scipy.linalg import solve


def mixture_weights(errors, family="gaussian", initialization="quantile", seed=0,
                    iterations=40, floor=0.02):
    if family not in {"gaussian", "student", "cauchy"}:
        raise ValueError("unknown mixture family")
    if initialization not in {"quantile", "random"}:
        raise ValueError("unknown initialization")
    errors = np.asarray(errors, dtype=float)
    if errors.size < 2 or np.any(errors < 0) or not np.isfinite(errors).all():
        raise ValueError("need at least two finite nonnegative errors")
    x = np.log(errors + 1e-8)
    rng = np.random.default_rng(seed)
    mu = np.quantile(x, [.25, .75]) if initialization == "quantile" else rng.choice(x, 2, replace=False)
    scale2 = np.full(2, max(np.var(x), 1e-4))
    pi = np.full(2, .5)
    nu = 3. if family == "student" else 1.
    def expectation():
        d2 = (x[:, None] - mu)**2 / scale2
        if family == "gaussian":
            lp = -.5 * (np.log(2*np.pi*scale2) + d2)
        else:
            lp = (gammaln((nu+1)/2)-gammaln(nu/2)
                  -.5*np.log(nu*np.pi*scale2)-(nu+1)/2*np.log1p(d2/nu))
        lp += np.log(np.maximum(pi, 1e-12))
        return np.exp(lp-logsumexp(lp, axis=1, keepdims=True)), d2
    for _ in range(iterations):
        resp, d2 = expectation()
        mass = np.maximum(resp.sum(0), 1e-8)
        latent = np.ones_like(resp) if family == "gaussian" else (nu+1)/(nu+d2)
        effective = resp * latent
        mu = (effective*x[:, None]).sum(0)/np.maximum(effective.sum(0), 1e-8)
        scale2 = np.maximum((effective*(x[:, None]-mu)**2).sum(0)/mass, 1e-4)
        pi = mass/mass.sum()
    resp, _ = expectation()
    signal = int(np.argmin(mu))
    return np.clip(resp[:, signal], floor, 1), {"location": mu.tolist(), "scale2": scale2.tolist(), "mixing": pi.tolist()}


class ResearchAE:
    """Exact per-item ridge, with the target removed from all input features.

    Cross-fitting removes held-out edges from features AND from the loss
    (they are not converted into negative training examples).
    """
    def __init__(self, rank=16, regularization=10., zero_weight=.01, method="gaussian",
                 outer=2, trim=.1, features=None, seed=42, mixture_seed=0,
                 initialization="quantile", crossfit=0):
        if method not in {"unit", "trim", "gaussian", "student", "cauchy"}:
            raise ValueError("unknown method")
        if rank < 1 or regularization <= 0 or not 0 < zero_weight < 1 or outer < 0 or not 0 <= trim < 1 or crossfit == 1 or crossfit < 0:
            raise ValueError("invalid model parameters")
        self.rank, self.regularization, self.zero_weight = rank, regularization, zero_weight
        self.method, self.outer, self.trim = method, outer, trim
        self.features, self.seed, self.mixture_seed = features, seed, mixture_seed
        self.initialization, self.crossfit = initialization, crossfit

    def _prepare(self, matrix):
        self.matrix_ = matrix.tocsr().astype(float)
        self.matrix_.sort_indices()
        self.projection_ = np.random.default_rng(self.seed).choice([-1., 1.], (matrix.shape[1], self.rank))/np.sqrt(self.rank)
        self.degree_ = np.asarray(matrix.sum(1)).ravel()
        blocks = [np.asarray(matrix @ self.projection_)]
        if self.features is not None:
            self.genre_sum_ = np.asarray(matrix @ self.features)
            blocks.append(self.genre_sum_/np.maximum(self.degree_[:, None], 1))
        blocks.append(np.ones((matrix.shape[0], 1)))
        self.design_ = np.column_stack(blocks)

    def _target_design(self, item):
        z = self.design_.copy()
        users = self.matrix_[:, item].nonzero()[0]
        z[users, :self.rank] -= self.projection_[item]
        if self.features is not None:
            z[users, self.rank:-1] = (self.genre_sum_[users]-self.features[item])/np.maximum(self.degree_[users, None]-1, 1)
        return z

    def _solve(self, weights, excluded=None):
        m = self.matrix_
        wc = sp.csr_matrix((weights, m.indices, m.indptr), shape=m.shape).tocsc()
        ec = None if excluded is None else excluded.tocsc()
        beta = np.zeros((self.design_.shape[1], m.shape[1]))
        for j in range(m.shape[1]):
            z = self._target_design(j)
            lo, hi = wc.indptr[j:j+2]; users = wc.indices[lo:hi]; w = wc.data[lo:hi]
            a = self.zero_weight*(z.T@z) + z[users].T@((w-self.zero_weight)[:, None]*z[users])
            if ec is not None:
                ignored = ec.indices[ec.indptr[j]:ec.indptr[j+1]]
                a -= self.zero_weight*(z[ignored].T@z[ignored])
            a.flat[::a.shape[0]+1] += self.regularization
            beta[:, j] = solve(a, z[users].T@w, assume_a="pos", check_finite=False)
        return beta

    def pair_scores(self, rows, cols):
        z = self.design_[rows].copy()
        present = np.asarray(self.matrix_[rows, cols]).ravel()
        z[:, :self.rank] -= present[:, None]*self.projection_[cols]
        if self.features is not None:
            z[:, self.rank:-1] = (self.genre_sum_[rows]-present[:, None]*self.features[cols])/np.maximum(self.degree_[rows, None]-present[:, None], 1)
        return np.einsum("ij,ji->i", z, self.decoder_[:, cols])

    def fit(self, matrix, excluded=None):
        self._prepare(matrix)
        rows, cols = self.matrix_.nonzero()
        weights = np.ones(len(rows)); self.history_ = []
        self.decoder_ = self._solve(weights, excluded)
        folds = np.random.default_rng(self.seed+991).permutation(len(rows)) % max(self.crossfit, 1)
        for iteration in range(self.outer if self.method != "unit" else 0):
            scores = self.pair_scores(rows, cols)
            if self.crossfit:
                for fold in range(self.crossfit):
                    mask = folds == fold
                    omitted = sp.csr_matrix((np.ones(mask.sum()), (rows[mask], cols[mask])), shape=matrix.shape)
                    auxiliary = ResearchAE(rank=self.rank, regularization=self.regularization,
                        zero_weight=self.zero_weight, method=self.method, outer=iteration,
                        trim=self.trim, features=self.features, seed=self.seed,
                        mixture_seed=self.mixture_seed, initialization=self.initialization)
                    auxiliary.fit(self.matrix_-omitted, excluded=omitted)
                    scores[mask] = auxiliary.pair_scores(rows[mask], cols[mask])
            errors = (1-scores)**2
            if self.method == "trim":
                weights = (errors <= np.quantile(errors, 1-self.trim)).astype(float)
                state = {}
            else:
                weights, state = mixture_weights(errors, self.method, self.initialization, self.mixture_seed)
            self.history_.append({"iteration": iteration+1, "mean_weight": float(weights.mean()), **state})
            self.decoder_ = self._solve(weights, excluded)
        self.confidence_ = weights
        self.errors_ = (1-self.pair_scores(rows, cols))**2
        return self

    def score(self, users, interactions):
        return np.asarray(self.design_[users]@self.decoder_, dtype=np.float32)


class GenreCosine:
    def __init__(self, features):
        self.features = features
    def fit(self, matrix):
        h = np.asarray(matrix@self.features)
        self.h = h/np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        self.f = self.features/np.maximum(np.linalg.norm(self.features, axis=1, keepdims=True), 1e-12)
        return self
    def score(self, users, interactions):
        return np.asarray(self.h[users]@self.f.T, dtype=np.float32)
