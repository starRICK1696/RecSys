from __future__ import annotations

import csv
from pathlib import Path

import pytest

from RecSys.main import build_parser, run
from RecSys.plot_robustness import load_results, plot_robustness


def test_plot_robustness_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "robustness.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dataset", "model", "noise_type", "noise_ratio",
                "recall_at_20", "ndcg_at_20", "train_seconds",
            ],
        )
        writer.writeheader()
        for noise_type in ("noisy_1", "noisy_0"):
            for ratio in (0.0, 0.1):
                writer.writerow(
                    {
                        "dataset": "toy", "model": "ProbAE", "noise_type": noise_type,
                        "noise_ratio": ratio, "recall_at_20": 0.5 - ratio,
                        "ndcg_at_20": 0.3 - ratio, "train_seconds": 1.2,
                    }
                )
    assert len(load_results(csv_path)) == 4
    figure = plot_robustness(csv_path)
    assert len(figure.axes) == 6
    assert figure.axes[0].get_ylabel() == "Recall@20"
    output = tmp_path / "plot.png"
    figure.savefig(output)
    assert output.stat().st_size > 0


def test_plot_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("dataset,model\ntoy,ProbAE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отсутствуют столбцы"):
        load_results(csv_path)


def test_custom_top_k_from_sweep_to_plot(tmp_path: Path) -> None:
    csv_path = tmp_path / "robustness_at_3.csv"
    args = build_parser().parse_args([
        "--output", str(csv_path), "--top-k", "3", "--models", "EASE",
        "--n-users", "12", "--n-items", "24", "--interactions-per-user", "4",
        "--noise-types", "noisy_1", "--noise-levels", "0",
    ])
    result, = run(args)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    assert float(row["recall_at_3"]) == result.recall
    assert float(row["ndcg_at_3"]) == result.ndcg
    assert {"recall_at_20", "ndcg_at_20"}.isdisjoint(row)

    figure = plot_robustness(csv_path, include_time=False)
    assert [axis.get_ylabel() for axis in figure.axes] == ["Recall@3", "NDCG@3"]
