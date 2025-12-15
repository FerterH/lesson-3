from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def plot_learning_curves(
    history: Dict[str, List[float]],
    *,
    output_path: Path,
    title: str = "Learning Curves",
) -> None:
    """Построить график сравнения train/test метрик по эпохам."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.plot(epochs, history.get("train_accuracy", []), label="train_acc")
    plt.plot(epochs, history.get("val_accuracy", []), label="val_acc")
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_curves(
    curves: Dict[str, List[float]],
    *,
    output_path: Path,
    title: str = "Accuracy Curves",
) -> None:
    """Построить график train/test accuracy по эпохам."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, series in curves.items():
        epochs = range(1, len(series) + 1)
        plt.plot(epochs, series, label=label)
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_curves(
    curves: Dict[str, List[float]],
    *,
    output_path: Path,
    title: str = "Loss Curves",
) -> None:
    """Построить график train/test loss по эпохам."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, series in curves.items():
        epochs = range(1, len(series) + 1)
        plt.plot(epochs, series, label=label)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    class_names: Sequence[str],
    *,
    output_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Отрисовать и сохранить изображение матрицы ошибок."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(matrix, dtype=float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True) + 1e-8
        data = data / row_sums
    plt.figure(figsize=(6, 5))
    plt.imshow(
        data,
        interpolation="nearest",
        cmap=plt.get_cmap("Blues"),
    )
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Истина")
    plt.xlabel("Предсказание")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    scores: Dict[Tuple[str, str], float],
    *,
    output_path: Path,
    title: str = "Hyperparameter Heatmap",
) -> None:
    """
    Визуализировать результаты перебора гиперпараметров
    в виде тепловой карты.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(scores.keys())
    if not keys:
        return
    x_labels = sorted({k[0] for k in keys})
    y_labels = sorted({k[1] for k in keys})
    grid = np.zeros((len(y_labels), len(x_labels)))
    for (x_key, y_key), val in scores.items():
        grid[y_labels.index(y_key), x_labels.index(x_key)] = val
    plt.figure(figsize=(7, 5))
    plt.imshow(grid, cmap="viridis")
    plt.title(title)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.colorbar(label="score")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_metric_table(metrics: Dict[str, float], *, output_path: Path) -> None:
    """Сохранить агрегированные метрики в таблицу (CSV или Markdown)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric,value"]
    lines.extend(f"{k},{v}" for k, v in metrics.items())
    output_path.write_text("\n".join(lines), encoding="utf-8")
