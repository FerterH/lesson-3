from __future__ import annotations

from typing import Iterable, Optional

from torch import nn  # type: ignore

ForwardPassCallable = nn.Module


def build_mlp(
    input_dim: int,
    num_classes: int,
    layer_sizes: Iterable[int] | None = None,
    hidden_layers: Iterable[int] | None = None,
    *,
    activation: str = "relu",
    dropout: Optional[float] = None,
    use_batchnorm: bool = False,
) -> nn.Module:
    """Создать MLP, соответствующий заданным архитектурным параметрам."""

    layers = []
    prev_dim = input_dim
    hidden_seq = list(layer_sizes or hidden_layers or [])
    activation_cls = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }.get(activation, nn.ReLU)

    for idx, hidden_dim in enumerate(hidden_seq):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_cls())
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)


def initialize_weights(model: nn.Module, *, method: str = "xavier") -> None:
    """Применить выбранную стратегию инициализации к параметрам модели."""

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "kaiming":
                nn.init.kaiming_normal_(module.weight)
            else:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def count_parameters(
    model: nn.Module,
    *,
    trainable_only: bool = True,
) -> int:
    """Вернуть количество (обучаемых) параметров в модели."""

    params = (
        p
        for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )
    return sum(p.numel() for p in params)


def apply_regularization(
    model: nn.Module,
    *,
    dropout_schedule: Optional[Iterable[float]] = None,
    batchnorm_momentum: Optional[float] = None,
) -> None:
    """Настроить адаптивные стратегии регуляризации для модели."""

    if dropout_schedule is not None:
        for module, rate in zip(
            (m for m in model.modules() if isinstance(m, nn.Dropout)),
            dropout_schedule,
        ):
            module.p = rate

    if batchnorm_momentum is not None:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = batchnorm_momentum
