from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Sequence

from utils import model_utils, visualization_utils  # type: ignore
from utils.experiment_utils import (  # type: ignore
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    ModelBuilder,
    OptimizationConfig,
    TrainingRoutine,
    configure_logging,
    default_dataset_builder,
    default_training_routine,
)

LOGGER = logging.getLogger(__name__)


def build_depth_variants(
    model_builder: ModelBuilder,
    *,
    input_dim: int,
    num_classes: int,
    hidden_width: int,
    depths: Sequence[int],
    activation: str = "relu",
) -> Dict[int, object]:
    """Построить модели для каждого заданного количества слоёв."""

    models: Dict[int, object] = {}
    for depth in depths:
        hidden_layers = [hidden_width] * max(depth - 1, 0)
        model = model_builder(
            input_dim=input_dim,
            num_classes=num_classes,
            layer_sizes=hidden_layers,
            activation=activation,
        )
        model_utils.initialize_weights(model)
        models[depth] = model
    return models


def run_depth_experiments(
    models: Dict[int, object],
    datasets: Dict[str, object],
    training_routine: TrainingRoutine,
    config: ExperimentConfig,
    *,
    output_dir: Path,
) -> Dict[int, ExperimentResult]:
    """Обучить и оценить набор моделей разной глубины."""

    results: Dict[int, ExperimentResult] = {}
    for depth, model in models.items():
        LOGGER.info("Старт обучения модели глубины %s", depth)
        depth_config = replace(
            config,
            description=f"Глубина {depth}",
        )
        results[depth] = training_routine(
            model,
            datasets,
            depth_config,
            callbacks=None,
        )
    return results


def analyze_overfitting_patterns(
    results: Dict[int, ExperimentResult],
    *,
    output_dir: Path,
    dropout_rates: Iterable[float],
    use_batchnorm: bool,
) -> None:
    """
    Подготовить диагностику, которая показывает влияние глубины
    и регуляризации на переобучение.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    acc_curves: Dict[str, list[float]] = {}
    loss_curves: Dict[str, list[float]] = {}
    for depth, result in sorted(results.items()):
        train_acc = result.history.get("train_accuracy", [])
        val_acc = result.history.get("val_accuracy", [])
        gap = [tr - va for tr, va in zip(train_acc, val_acc)]
        overfit_epoch = next(
            (idx + 1 for idx, g in enumerate(gap) if g > 0.05),
            len(train_acc),
        )
        summary_lines.append(
            f"Глубина {depth}: старт переобучения на эпохе {overfit_epoch}"
        )
        acc_curves[f"depth_{depth}"] = val_acc
        loss_curves[f"depth_{depth}"] = result.history.get("val_loss", [])

    visualization_utils.plot_accuracy_curves(
        acc_curves,
        output_path=output_dir / "acc_curve.png",
        title="Кривые точности по глубине (val)",
    )
    visualization_utils.plot_loss_curves(
        loss_curves,
        output_path=output_dir / "loss_curve.png",
        title="Кривые потерь по глубине (val)",
    )
    (output_dir / "overfitting_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def summarize_training_time(
    results: Dict[int, ExperimentResult],
    *,
    report_path: Path,
) -> None:
    """Сформировать отчёт о времени обучения моделей разной глубины."""

    lines = ["depth,train_time,params,test_accuracy"]
    for depth, result in sorted(results.items()):
        lines.append(
            f"{depth},{result.metrics.get('train_time', 0):.4f},"
            f"{result.metrics.get('params', 0)},"
            f"{result.metrics.get('test_accuracy', 0):.4f}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Точка входа для запуска экспериментов с глубиной сети."""

    configure_logging()
    base_results = Path("homework") / "results" / "depth_experiments"
    base_results.mkdir(parents=True, exist_ok=True)

    dataset_cfg = DatasetConfig(
        name="synthetic_depth",
        input_dim=20,
        num_classes=3,
        train_size=1200,
        val_size=300,
        test_size=300,
    )
    optim_cfg = OptimizationConfig(
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        seed=42,
    )
    exp_config = ExperimentConfig(
        dataset=dataset_cfg,
        optimization=optim_cfg,
        description="Эксперименты по глубине",
        tags=["depth"],
    )

    datasets = default_dataset_builder(dataset_cfg)
    models = build_depth_variants(
        model_utils.build_mlp,  # type: ignore[arg-type]
        input_dim=dataset_cfg.input_dim,
        num_classes=dataset_cfg.num_classes,
        hidden_width=128,
        depths=[1, 2, 3, 5, 7],
        activation="relu",
    )
    results = run_depth_experiments(
        models,
        datasets,
        default_training_routine,
        exp_config,
        output_dir=base_results,
    )
    analyze_overfitting_patterns(
        results,
        output_dir=base_results,
        dropout_rates=[0.0, 0.1, 0.3, 0.5],
        use_batchnorm=True,
    )
    summarize_training_time(
        results,
        report_path=base_results / "timing_summary.csv",
    )
    # Сохраняем агрегированные метрики/истории
    metrics_agg = {
        f"depth_{depth}": res.metrics for depth, res in results.items()
    }
    history_agg = {
        f"depth_{depth}": res.history for depth, res in results.items()
    }
    (base_results / "metrics.json").write_text(
        json.dumps(metrics_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base_results / "history.json").write_text(
        json.dumps(history_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
