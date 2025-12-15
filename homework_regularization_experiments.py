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


def evaluate_regularization_strategies(
    strategies: Sequence[str],
    *,
    model_builder: ModelBuilder,
    training_routine: TrainingRoutine,
    config: ExperimentConfig,
    datasets: Dict[str, object],
    output_dir: Path,
) -> Dict[str, ExperimentResult]:
    """Сравнить точность и стабильность разных стратегий регуляризации."""

    strategies_config: Dict[
        str, Dict[str, float | bool | None]
    ] = {
        "none": {"dropout": None, "batchnorm": False, "weight_decay": 0.0},
        "dropout_0.1": {
            "dropout": 0.1,
            "batchnorm": False,
            "weight_decay": 0.0,
        },
        "dropout_0.3": {
            "dropout": 0.3,
            "batchnorm": False,
            "weight_decay": 0.0,
        },
        "dropout_0.5": {
            "dropout": 0.5,
            "batchnorm": False,
            "weight_decay": 0.0,
        },
        "batchnorm": {
            "dropout": None,
            "batchnorm": True,
            "weight_decay": 0.0,
        },
        "dropout_bn": {
            "dropout": 0.3,
            "batchnorm": True,
            "weight_decay": 0.0,
        },
        "l2": {"dropout": None, "batchnorm": False, "weight_decay": 1e-4},
    }

    results: Dict[str, ExperimentResult] = {}
    for strategy in strategies:
        cfg = strategies_config.get(strategy, strategies_config["none"])
        LOGGER.info("Стратегия регуляризации: %s", strategy)
        model = model_builder(
            input_dim=config.dataset.input_dim,
            num_classes=config.dataset.num_classes,
            layer_sizes=[256, 128, 64],
            activation="relu",
            dropout=cfg["dropout"],
            use_batchnorm=bool(cfg["batchnorm"]),
        )
        model_utils.initialize_weights(model)
        strategy_config = replace(
            config,
            optimization=replace(
                config.optimization,
                weight_decay=float(cfg.get("weight_decay", 0.0) or 0.0),
            ),
            description=f"Регуляризация {strategy}",
        )
        results[strategy] = training_routine(
            model,
            datasets,
            strategy_config,
        )
    return results


def visualize_weight_distributions(
    results: Dict[str, ExperimentResult],
    *,
    output_dir: Path,
) -> None:
    """Построить графики, показывающие различия распределений весов."""

    metrics_only = {k: v.metrics for k, v in results.items()}
    visualization_utils.save_metric_table(
        {
            f"{k}_test_acc": v.get("test_accuracy", 0.0)
            for k, v in metrics_only.items()
        },
        output_path=output_dir / "regularization_metrics.csv",
    )


def adaptive_regularization_study(
    schedule_config: Dict[str, Iterable[float] | float],
    *,
    model_builder: ModelBuilder,
    training_routine: TrainingRoutine,
    config: ExperimentConfig,
    datasets: Dict[str, object],
    output_dir: Path,
) -> ExperimentResult:
    """
    Провести эксперименты с адаптивными настройками
    регуляризации по слоям.
    """

    dropout_raw = schedule_config.get("dropout_schedule")
    dropout_schedule: list[float] = []
    if (
        dropout_raw is not None
        and hasattr(dropout_raw, "__iter__")
        and not isinstance(dropout_raw, (str, bytes, float, int))
    ):
        dropout_schedule = [
            float(v) for v in dropout_raw  # type: ignore[arg-type]
        ]
    bn_momentum = None
    for maybe_key in ("batchnorm_momentum", "momentum"):
        if maybe_key in schedule_config:
            maybe_val = schedule_config[maybe_key]
            bn_momentum = float(maybe_val)  # type: ignore[arg-type]
            break

    model = model_builder(
        input_dim=config.dataset.input_dim,
        num_classes=config.dataset.num_classes,
        layer_sizes=[256, 128, 64],
        activation="relu",
        dropout=dropout_schedule[0] if dropout_schedule else None,
        use_batchnorm=bn_momentum is not None,
    )
    model_utils.initialize_weights(model)
    model_utils.apply_regularization(
        model,
        dropout_schedule=dropout_schedule,
        batchnorm_momentum=bn_momentum,
    )
    return training_routine(
        model,
        datasets,
        config,
    )


def main() -> None:
    """Точка входа для экспериментов с регуляризацией."""

    configure_logging()
    base_results = Path("homework") / "results" / "regularization_experiments"
    base_results.mkdir(parents=True, exist_ok=True)

    dataset_cfg = DatasetConfig(
        name="synthetic_reg",
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
        description="Эксперименты по регуляризации",
        tags=["regularization"],
    )

    datasets = default_dataset_builder(dataset_cfg)
    strategies = [
        "none",
        "dropout_0.1",
        "dropout_0.3",
        "dropout_0.5",
        "batchnorm",
        "dropout_bn",
        "l2",
    ]
    results = evaluate_regularization_strategies(
        strategies,
        model_builder=model_utils.build_mlp,  # type: ignore[arg-type]
        training_routine=default_training_routine,
        config=exp_config,
        datasets=datasets,
        output_dir=base_results,
    )
    visualize_weight_distributions(
        results,
        output_dir=base_results,
    )
    _ = adaptive_regularization_study(
        schedule_config={
            "dropout_schedule": [0.2, 0.3, 0.4],
            "batchnorm_momentum": 0.05,
        },
        model_builder=model_utils.build_mlp,  # type: ignore[arg-type]
        training_routine=default_training_routine,
        config=exp_config,
        datasets=datasets,
        output_dir=base_results / "adaptive",
    )

    # Агрегируем метрики/истории
    metrics_agg = {name: res.metrics for name, res in results.items()}
    history_agg = {name: res.history for name, res in results.items()}
    (base_results / "metrics.json").write_text(
        json.dumps(metrics_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base_results / "history.json").write_text(
        json.dumps(history_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    visualization_utils.plot_accuracy_curves(
        {
            name: res.history.get("val_accuracy", [])
            for name, res in results.items()
        },
        output_path=base_results / "acc_curve.png",
        title="Точность по регуляризациям (val)",
    )
    visualization_utils.plot_loss_curves(
        {
            name: res.history.get("val_loss", [])
            for name, res in results.items()
        },
        output_path=base_results / "loss_curve.png",
        title="Потери по регуляризациям (val)",
    )


if __name__ == "__main__":
    main()
