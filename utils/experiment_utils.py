from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np  # type: ignore
import torch  # type: ignore
from sklearn.datasets import make_classification  # type: ignore
from torch import nn  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore

from . import model_utils


@dataclass
class DatasetConfig:
    """
    Метаданные, необходимые для подготовки датасетов
    в рамках экспериментов.
    """

    name: str
    input_dim: int
    num_classes: int
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Гиперпараметры, общие для всех экспериментов."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: Optional[int] = 42


@dataclass
class ExperimentConfig:
    """Полный набор параметров для отдельного запуска эксперимента."""

    dataset: DatasetConfig
    optimization: OptimizationConfig
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Структурированный контейнер для результатов эксперимента."""

    name: str
    metrics: Dict[str, float]
    history: Dict[str, List[float]]
    artifacts: Dict[str, Path] = field(default_factory=dict)

    def save(self, output_dir: Path) -> None:
        """Сохранить метрики, историю и артефакты на диск."""

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        history_path = output_dir / "history.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


class DatasetBuilder(Protocol):
    """
    Интерфейс вызываемого объекта, подготавливающего
    обучающие, валидационные и тестовые наборы.
    """

    def __call__(self, config: DatasetConfig) -> Dict[str, Any]:
        raise NotImplementedError


class ModelBuilder(Protocol):
    """Интерфейс для фабрики моделей, используемых в экспериментах."""

    def __call__(
        self,
        input_dim: int,
        num_classes: int,
        layer_sizes: Iterable[int],
        *,
        activation: Optional[str] = None,
        dropout: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> Any:
        raise NotImplementedError


class TrainingRoutine(Protocol):
    """Интерфейс, инкапсулирующий цикл обучения."""

    def __call__(
        self,
        model: Any,
        datasets: Dict[str, Any],
        config: ExperimentConfig,
        *,
        callbacks: Optional[List[Callable[..., None]]] = None,
        output_dir: Optional[Path] = None,
    ) -> ExperimentResult:
        raise NotImplementedError


def configure_logging(log_level: str = "INFO") -> None:
    """Настроить логирование для скриптов экспериментов."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ==== Реализации по умолчанию ====


def default_dataset_builder(config: DatasetConfig) -> Dict[str, DataLoader]:
    """
    Построить простые даталоадеры на базе synthetic make_classification
    для быстрой проверки экспериментов.
    """

    train_size = config.train_size or 800
    val_size = config.val_size or 200
    test_size = config.test_size or 200
    total = train_size + val_size + test_size

    X, y = make_classification(
        n_samples=total,
        n_features=config.input_dim,
        n_informative=max(2, config.input_dim // 2),
        n_redundant=0,
        n_classes=config.num_classes,
        random_state=config.train_size or 0,
    )
    X = X.astype(np.float32)

    def to_loader(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        tensors = (
            torch.from_numpy(x_arr),
            torch.from_numpy(y_arr.astype(np.int64)),
        )
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=64, shuffle=shuffle)

    x_train, y_train = X[:train_size], y[:train_size]
    x_val, y_val = (
        X[train_size:train_size + val_size],
        y[train_size:train_size + val_size],
    )
    x_test, y_test = (
        X[train_size + val_size:],
        y[train_size + val_size:],
    )

    return {
        "train": to_loader(x_train, y_train, shuffle=True),
        "val": to_loader(x_val, y_val, shuffle=False),
        "test": to_loader(x_test, y_test, shuffle=False),
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Подсчитать loss и accuracy на указанном даталоадере."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def default_training_routine(
    model: nn.Module,
    datasets: Dict[str, DataLoader],
    config: ExperimentConfig,
    *,
    callbacks: Optional[List[Callable[..., None]]] = None,
    output_dir: Optional[Path] = None,
) -> ExperimentResult:
    """
    Базовый цикл обучения для экспериментов: обучает, считает метрики,
    сохраняет графики и возвращает ExperimentResult.
    """

    device = torch.device(config.optimization.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    start_time = time.perf_counter()
    for _epoch in range(config.optimization.epochs):
        model.train()
        for xb, yb in datasets["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = _evaluate_model(
            model,
            datasets["train"],
            device,
        )
        val_loss, val_acc = _evaluate_model(
            model,
            datasets.get("val", datasets["test"]),
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        if callbacks:
            for cb in callbacks:
                cb()

    total_time = time.perf_counter() - start_time
    test_loss, test_acc = _evaluate_model(model, datasets["test"], device)

    metrics = {
        "train_accuracy": history["train_accuracy"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "test_accuracy": test_acc,
        "train_time": total_time,
        "params": model_utils.count_parameters(model),
    }

    result = ExperimentResult(
        name=config.dataset.name,
        metrics=metrics,
        history=history,
        artifacts={},
    )

    return result
