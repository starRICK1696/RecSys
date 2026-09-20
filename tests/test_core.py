from __future__ import annotations

import numpy as np
import pytest
from RecSys.data import generate_synthetic_graph, inject_noise, split_interactions
from RecSys.engine import evaluate
from RecSys.models import EASE, MultVAE, ProbAE, Recommender, SVD_AE, signal_responsibilities


def test_noise_counts_and_determinism() -> None:
    matrix = generate_synthetic_graph(
        n_users=20, n_items=30, interactions_per_user=5, n_topics=4, seed=7
    )
    added = inject_noise(matrix, 0.2, noise_type="noisy_1", seed=3)
    removed = inject_noise(matrix, 0.2, noise_type="noisy_0", seed=3)
    assert added.nnz == matrix.nnz + round(0.2 * matrix.nnz)
    assert removed.nnz == matrix.nnz - round(0.2 * matrix.nnz)
    assert (added != inject_noise(matrix, 0.2, noise_type="noisy_1", seed=3)).nnz == 0


def test_noise_excludes_heldout_pairs() -> None:
    matrix = generate_synthetic_graph(
        n_users=20, n_items=30, interactions_per_user=5, n_topics=4, seed=7
    )
    data = split_interactions(matrix, seed=7)
    noisy = inject_noise(
        data.train,
        0.5,
        noise_type="noisy_1",
        exclude=data.validation + data.test,
        seed=7,
    )
    assert noisy.multiply(data.validation + data.test).nnz == 0


def test_gmm_assigns_more_confidence_to_small_errors() -> None:
    errors = np.r_[np.full(100, 0.01), np.full(100, 1.0)]
    confidence, _ = signal_responsibilities(errors)
    assert float(confidence[:100].mean()) > float(confidence[100:].mean())


@pytest.mark.parametrize("components,signal", [(1, 1), (2, 1), (3, 1), (4, 2)])
def test_arbitrary_gmm_components(components: int, signal: int) -> None:
    errors = np.r_[np.linspace(0.01, 0.03, 30), np.linspace(0.3, 0.5, 30), np.linspace(1, 2, 30)]
    confidence, state = signal_responsibilities(
        errors, n_components=components, signal_components=signal
    )
    assert confidence.shape == errors.shape
    assert len(state.means) == components
    assert np.all(np.diff(state.means) >= 0)
    assert np.all((confidence >= 0.02) & (confidence <= 1))
    if components == 1:
        assert np.all(confidence == 1)


def test_four_component_probae_fit() -> None:
    matrix = generate_synthetic_graph(
        n_users=24, n_items=32, interactions_per_user=6, n_topics=4, seed=5
    )
    model = ProbAE(
        rank=5, regularization=2, em_iterations=2,
        gmm_components=4, signal_components=2, seed=5,
    ).fit(matrix)
    assert model.confidence_ is not None
    assert model.confidence_.shape == (matrix.nnz,)
    assert len(model.history_[-1].means) == 4


def test_invalid_gmm_configuration() -> None:
    with pytest.raises(ValueError, match="signal_components"):
        ProbAE(gmm_components=2, signal_components=3)


def test_closed_form_models_shapes() -> None:
    matrix = generate_synthetic_graph(
        n_users=25, n_items=35, interactions_per_user=7, n_topics=4, seed=2
    )
    data = split_interactions(matrix, seed=2)
    users = np.arange(5, dtype=np.int64)
    models: list[Recommender] = [
        EASE(regularization=5.0),
        SVD_AE(rank=4, seed=2),
        ProbAE(rank=5, regularization=2.0, em_iterations=2, seed=2),
    ]
    for model in models:
        model.fit(data.train)
        scores = model.score(users, data.train)
        assert scores.shape == (5, data.n_items)
        assert np.isfinite(scores).all()

    prob = models[-1]
    assert isinstance(prob, ProbAE)
    assert prob.projection_ is not None and prob.decoder_ is not None
    effective_decoder = prob.projection_ @ prob.decoder_[:-1]
    assert np.max(np.abs(np.diag(effective_decoder))) < 1e-5


def test_multvae_and_evaluator_smoke() -> None:
    matrix = generate_synthetic_graph(
        n_users=24, n_items=32, interactions_per_user=6, n_topics=4, seed=4
    )
    data = split_interactions(matrix, seed=4)
    model = MultVAE(
        hidden_dim=12, latent_dim=4, epochs=1, batch_size=8, device="cpu", seed=4
    ).fit(data.train)
    result = evaluate(model, data.train, data.test, k=5, batch_size=8)
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.ndcg <= 1.0


def test_distribution_generator_is_reproducible_and_has_unique_edges() -> None:
    first = generate_synthetic_graph(
        n_users=12, n_items=18, interactions_per_user=15, n_topics=3,
        user_concentration=0.2, item_concentration=0.05, seed=17,
    )
    second = generate_synthetic_graph(
        n_users=12, n_items=18, interactions_per_user=15, n_topics=3,
        user_concentration=0.2, item_concentration=0.05, seed=17,
    )
    assert first.nnz == 12 * 15
    assert np.array_equal(np.diff(first.indptr), np.full(12, 15))
    assert (first != second).nnz == 0
    with pytest.raises(ValueError, match="concentrations"):
        generate_synthetic_graph(
            n_items=10, interactions_per_user=5, item_concentration=0
        )
