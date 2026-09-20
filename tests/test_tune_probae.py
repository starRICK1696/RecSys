from __future__ import annotations

from RecSys.tune_probae import candidate_configs


def test_candidate_search_is_stratified_and_bounded() -> None:
    for trial_count in (4, 5, 32):
        candidates = candidate_configs(trial_count, seed=42)
        assert len(candidates) == trial_count
        assert len(set(candidates)) == trial_count
        assert {item.gmm_components for item in candidates} == {1, 2, 3, 4}
