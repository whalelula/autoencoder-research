from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class HistoryWriter:
    FIELDS = [
        "split",
        "epoch",
        "step",
        "total",
        "mrstft",
        "kl",
        "mrstft_sc",
        "mrstft_lm",
        "mrstft_if",
        "mrstft_gd",
        "mrstft_complex",
        "si_sdr",
        "learning_rate",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDS).writeheader()

    def append(
        self,
        split: str,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        row = {
            "split": split,
            "epoch": epoch,
            "step": step,
            **{
                key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                for key, value in metrics.items()
                if key in self.FIELDS
            },
        }
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDS).writerow(row)


def plot_history(history_path: str | Path, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with Path(history_path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, ("total", "mrstft", "kl")):
        for split, style in (("train", "-"), ("val", "--")):
            selected = [
                (int(row["step"]), float(row[metric]))
                for row in rows
                if row["split"] == split and row.get(metric)
            ]
            if selected:
                x, y = zip(*selected)
                axis.plot(x, y, style, label=split, alpha=0.85)
        axis.set_title(metric)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
