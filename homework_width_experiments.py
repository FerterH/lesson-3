from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from utils import model_utils, visualization_utils
from utils.experiment_utils import (
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


def compare_width_profiles(
    model_builder: ModelBuilder,
    width_profiles: Dict[str, Sequence[int]],
    *,
    input_dim: int,
    num_classes: int,
    training_routine: TrainingRoutine,
    config: ExperimentConfig,
    datasets: Dict[str, object],
    output_dir: Path,
) -> Dict[str, ExperimentResult]:
    """Запустить эксперименты для каждого заданного профиля ширины."""

    results: Dict[str, ExperimentResult] = {}
    for name, widths in width_profiles.items():
        LOGGER.info("Запуск профиля %s: %s", name, widths)
        model = model_builder(
            input_dim=input_dim,
            num_classes=num_classes,
            layer_sizes=widths,
            activation="relu",
        )
        model_utils.initialize_weights(model)
        profile_config = replace(
            config,
            description=f"Профиль {name}",
        )
        results[name] = training_routine(
            model,
            datasets,
            profile_config,
        )
    return results


def compute_param_statistics(
    results: Dict[str, ExperimentResult],
) -> Dict[str, int]:
    """Вернуть количество параметров, соответствующее каждому профилю."""

    return {
        name: int(res.metrics.get("params", 0))
        for name, res in results.items()
    }


def grid_search_width_architectures(
    search_space: Iterable[Tuple[int, int, int]],
    *,
    model_builder: ModelBuilder,
    training_routine: TrainingRoutine,
    config: ExperimentConfig,
    datasets: Dict[str, object],
    metric_name: str,
    output_dir: Path,
) -> ExperimentResult:
    """Выполнить поиск по сетке ширин и определить лучший вариант."""

    best_result: ExperimentResult | None = None
    best_score = float("-inf")
    all_results: Dict[str, ExperimentResult] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for widths in search_space:
        name = f"{widths}"
        LOGGER.info("Пробуем ширины %s", widths)
        model = model_builder(
            input_dim=config.dataset.input_dim,
            num_classes=config.dataset.num_classes,
            layer_sizes=list(widths),
            activation="relu",
        )
        model_utils.initialize_weights(model)
        result = training_routine(
            model,
            datasets,
            config,
        )
        all_results[name] = result
        score = result.metrics.get(metric_name, 0.0)
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise RuntimeError("Не удалось найти лучшую архитектуру.")

    heatmap_scores = {
        (str(w[0]), str(w[1])): res.metrics.get(metric_name, 0.0)
        for w, res in (
            (tuple(map(int, k.strip("()").split(", "))), v)
            for k, v in all_results.items()
        )
    }
    visualization_utils.plot_heatmap(
        heatmap_scores,
        output_path=output_dir / "grid_heatmap.png",
        title="Ширина слоёв: grid search",
    )
    # Сохраняем агрегированные результаты грид-поиска
    metrics_agg = {name: res.metrics for name, res in all_results.items()}
    history_agg = {name: res.history for name, res in all_results.items()}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history_agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_result


def main() -> None:
    """Точка входа для экспериментов, посвящённых ширине слоёв."""

    configure_logging()
    base_results = Path("homework") / "results" / "width_experiments"
    base_results.mkdir(parents=True, exist_ok=True)

    dataset_cfg = DatasetConfig(
        name="synthetic_width",
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
        description="Эксперименты по ширине",
        tags=["width"],
    )

    datasets = default_dataset_builder(dataset_cfg)
    width_profiles: Dict[str, Sequence[int]] = {
        "narrow": [64, 32, 16],
        "medium": [256, 128, 64],
        "wide": [1024, 512, 256],
        "very_wide": [2048, 1024, 512],
    }

    results = compare_width_profiles(
        model_utils.build_mlp,
        width_profiles,
        input_dim=dataset_cfg.input_dim,
        num_classes=dataset_cfg.num_classes,
        training_routine=default_training_routine,
        config=exp_config,
        datasets=datasets,
        output_dir=base_results,
    )

    param_stats = compute_param_statistics(results)
    metric_table_path = base_results / "width_param_stats.csv"
    visualization_utils.save_metric_table(
        {k: float(v) for k, v in param_stats.items()},
        output_path=metric_table_path,
    )

    grid_results_dir = base_results / "grid_search"
    grid_results_dir.mkdir(parents=True, exist_ok=True)
    _ = grid_search_width_architectures(
        search_space=[
            (128, 128, 128),
            (256, 128, 64),
            (512, 256, 128),
        ],
        model_builder=model_utils.build_mlp,
        training_routine=default_training_routine,
        config=exp_config,
        datasets=datasets,
        metric_name="test_accuracy",
        output_dir=grid_results_dir,
    )

    # Агрегируем метрики/истории по профилям
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
    # Общие графики (используем валидационные кривые)
    visualization_utils.plot_accuracy_curves(
        {
            name: res.history.get("val_accuracy", [])
            for name, res in results.items()
        },
        output_path=base_results / "acc_curve.png",
        title="Точность по профилям ширины (val)",
    )
    visualization_utils.plot_loss_curves(
        {
            name: res.history.get("val_loss", [])
            for name, res in results.items()
        },
        output_path=base_results / "loss_curve.png",
        title="Потери по профилям ширины (val)",
    )


if __name__ == "__main__":
    main()
