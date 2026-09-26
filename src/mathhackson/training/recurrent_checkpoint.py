"""Validate and migrate complete single-ant recurrent checkpoints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .recurrent import (HIDDEN_WIDTH, INPUT_WIDTH, LEGACY_MEMORY_LAGS,
                        LEGACY_MODEL_VERSION, MEMORY_LAGS, MODEL_VERSION,
                        RecurrentPolicy)

PARAMETER_NAMES = ("motor", "input_weights", "hidden_weights", "hidden_bias",
                   "action_weights", "gate_weights", "write_weights")


def migrate_hidden_weights(old: np.ndarray) -> np.ndarray:
    expected = (HIDDEN_WIDTH, HIDDEN_WIDTH * len(LEGACY_MEMORY_LAGS))
    if old.shape != expected or not np.isfinite(old).all():
        raise ValueError("旧版隐藏状态参数形状或数值无效")
    result = np.zeros((HIDDEN_WIDTH, HIDDEN_WIDTH * len(MEMORY_LAGS)), np.float32)
    for source, lag in enumerate(LEGACY_MEMORY_LAGS):
        destination = MEMORY_LAGS.index(lag)
        result[:, destination * HIDDEN_WIDTH:(destination + 1) * HIDDEN_WIDTH] = \
            old[:, source * HIDDEN_WIDTH:(source + 1) * HIDDEN_WIDTH]
    return result


def read_recurrent_parameters(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(PARAMETER_NAMES) - set(archive.files)
        if missing:
            raise ValueError(f"检查点缺少循环模型参数：{', '.join(sorted(missing))}")
        if "model_version" not in archive:
            raise ValueError("检查点属于旧版循环模型，不能用于稀疏记忆模型对照")
        version = str(archive["model_version"])
        if version not in {LEGACY_MODEL_VERSION, MODEL_VERSION}:
            raise ValueError("检查点属于旧版循环模型，不能用于稀疏记忆模型对照")
        values = tuple(np.asarray(archive[name], np.float32).copy() for name in PARAMETER_NAMES)
    expected = ((2, INPUT_WIDTH), (HIDDEN_WIDTH, INPUT_WIDTH),
                (HIDDEN_WIDTH, HIDDEN_WIDTH * (len(LEGACY_MEMORY_LAGS) if version == LEGACY_MODEL_VERSION
                                                 else len(MEMORY_LAGS))),
                (HIDDEN_WIDTH,), (2, HIDDEN_WIDTH), (HIDDEN_WIDTH + 1,),
                (3, HIDDEN_WIDTH + 1))
    if any(value.shape != shape or not np.isfinite(value).all()
           for value, shape in zip(values, expected, strict=True)):
        raise ValueError("循环模型参数形状或数值无效")
    if version == LEGACY_MODEL_VERSION:
        values = (*values[:2], migrate_hidden_weights(values[2]), *values[3:])
    return values


def load_recurrent_checkpoint(policy: RecurrentPolicy, path: Path) -> None:
    if policy.input_width != INPUT_WIDTH or policy.memory_lags != MEMORY_LAGS:
        raise ValueError("检查点只能载入对应的基础循环模型")
    if policy.outer_updates or policy.self_updates:
        raise ValueError("完整检查点只能载入尚未训练的模型")
    values = read_recurrent_parameters(path)
    with torch.no_grad():
        for parameter, value in zip(policy.parameters, values, strict=True):
            parameter.copy_(torch.from_numpy(value))
    policy.phase = "memory"
    policy.write_mode = "off"
    policy.reset_state()
