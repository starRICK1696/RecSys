"""Plot recommendation quality and training time from robustness.csv.

Examples:
    python -m RecSys.plot_robustness
    python -m RecSys.plot_robustness --input RecSys/results/robustness.csv \
        --output RecSys/results/robustness.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Works on headless servers and in CI.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np


@dataclass(frozen=True, slots=True)
class ResultRow:
    dataset: str
    model: str
    noise_type: str
    noise_ratio: float
    recall_at_20: float
    ndcg_at_20: float
    train_seconds: float


REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "noise_type",
    "noise_ratio",
    "recall_at_20",
    "ndcg_at_20",
    "train_seconds",
}
MODEL_ORDER = ("ProbAE", "SVD_AE", "EASE", "MultVAE")
MODEL_STYLES = {
    "ProbAE": ("#1864ab", "o"),
    "SVD_AE": ("#e8590c", "s"),
    "EASE": ("#2b8a3e", "^"),
    "MultVAE": ("#9c36b5", "D"),
}
NOISE_LABELS = {
    "noisy_1": "Noisy 1 · ложные взаимодействия",
    "noisy_0": "Noisy 0 · скрытые взаимодействия",
    "mixed": "Смешанный шум",
}


def load_results(path: str | Path) -> list[ResultRow]:
    """Read and validate the CSV produced by ``RecSys.main``."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV с результатами не найден: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}")

        rows: list[ResultRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = ResultRow(
                    dataset=str(raw["dataset"]),
                    model=str(raw["model"]),
                    noise_type=str(raw["noise_type"]),
                    noise_ratio=float(raw["noise_ratio"]),
                    recall_at_20=float(raw["recall_at_20"]),
                    ndcg_at_20=float(raw["ndcg_at_20"]),
                    train_seconds=float(raw["train_seconds"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Некорректные значения в строке {line_number}") from exc
            values = (
                row.noise_ratio,
                row.recall_at_20,
                row.ndcg_at_20,
                row.train_seconds,
            )
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"Не-конечное значение в строке {line_number}")
            if not 0 <= row.noise_ratio <= 1 or not 0 <= row.recall_at_20 <= 1:
                raise ValueError(f"Значение вне [0, 1] в строке {line_number}")
            if not 0 <= row.ndcg_at_20 <= 1 or row.train_seconds < 0:
                raise ValueError(f"Значение метрики/времени вне диапазона в строке {line_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV пуст: {source}")
    return rows


def _ordered_models(rows: Sequence[ResultRow]) -> list[str]:
    present = {row.model for row in rows}
    return [model for model in MODEL_ORDER if model in present] + sorted(
        present - set(MODEL_ORDER)
    )


def plot_robustness(
    csv_path: str | Path,
    *,
    dataset: str | None = None,
    include_time: bool = True,
) -> Figure:
    """Build a figure; notebook callers may simply display the returned Figure.

    If repeated runs exist for a model/noise level, the line shows their mean
    and the translucent band shows ±1 sample standard deviation.
    """
    rows = load_results(csv_path)
    datasets = sorted({row.dataset for row in rows})
    if dataset is None:
        if len(datasets) != 1:
            raise ValueError(
                f"CSV содержит несколько датасетов {datasets}; выберите один через dataset="
            )
        dataset = datasets[0]
    elif dataset not in datasets:
        raise ValueError(f"Датасет {dataset!r} не найден; доступны: {datasets}")
    rows = [row for row in rows if row.dataset == dataset]

    noise_types = [
        noise for noise in ("noisy_1", "noisy_0", "mixed") if any(row.noise_type == noise for row in rows)
    ]
    noise_types.extend(
        sorted({row.noise_type for row in rows} - set(noise_types))
    )
    metrics = [
        ("recall_at_20", "Recall@20", True),
        ("ndcg_at_20", "NDCG@20", True),
    ]
    if include_time:
        metrics.append(("train_seconds", "Время обучения, с", False))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(
        len(metrics),
        len(noise_types),
        figsize=(5.2 * len(noise_types), 3.0 * len(metrics)),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    figure.suptitle(f"Устойчивость рекомендательных моделей · {dataset}", fontsize=14)
    models = _ordered_models(rows)

    for column, noise_type in enumerate(noise_types):
        noise_rows = [row for row in rows if row.noise_type == noise_type]
        for metric_index, (field, ylabel, percentage) in enumerate(metrics):
            ax = axes[metric_index, column]
            for model_index, model in enumerate(models):
                series = [row for row in noise_rows if row.model == model]
                if not series:
                    continue
                by_level: dict[float, list[float]] = defaultdict(list)
                for row in series:
                    by_level[row.noise_ratio].append(float(getattr(row, field)))
                levels = np.asarray(sorted(by_level), dtype=float)
                means = np.asarray([np.mean(by_level[level]) for level in levels])
                deviations = np.asarray(
                    [np.std(by_level[level], ddof=1) if len(by_level[level]) > 1 else 0.0 for level in levels]
                )
                color, marker = MODEL_STYLES.get(
                    model, (f"C{model_index % 10}", "o")
                )
                ax.plot(
                    levels,
                    means,
                    label=model,
                    color=color,
                    marker=marker,
                    markersize=5,
                    linewidth=2,
                )
                if np.any(deviations > 0):
                    ax.fill_between(
                        levels, means - deviations, means + deviations, color=color, alpha=0.14
                    )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.set_xlim(left=0)
            if percentage:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
                ax.set_ylim(bottom=0)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            if metric_index == 0:
                ax.set_title(NOISE_LABELS.get(noise_type, noise_type), fontsize=11)
            if metric_index == len(metrics) - 1:
                ax.set_xlabel("Уровень шума")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4))
    return figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("RecSys/results/robustness.csv"))
    parser.add_argument("--output", type=Path, default=Path("RecSys/results/robustness.png"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-time", action="store_true", help="Показывать только Recall/NDCG")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    figure = plot_robustness(
        args.input, dataset=args.dataset, include_time=not args.no_time
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"График сохранён: {args.output}")


if __name__ == "__main__":
    main()
