"""只保存可塑方向结构参数；不含在线状态、优化器或环境，不支持任意时点续训。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import zipfile

import numpy as np
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from mathhackson.training.direction.policy import DirectionMotor
from ..plastic_direction import PlasticDirection


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    version: Literal["plastic-direction-structure-v1"] = "plastic-direction-structure-v1"
    update: int = Field(ge=0, le=2**63 - 1)
    seed: int = Field(ge=0, le=2**63 - 1)
    feature_width: int = Field(ge=1, le=2**63 - 1)
    max_step: float = Field(gt=0)
    max_fast: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_limits(self) -> CheckpointMetadata:
        if self.max_step > self.max_fast:
            raise ValueError("单次更新上限不能超过累计快权重上限")
        return self


def _model(metadata: CheckpointMetadata) -> PlasticDirection:
    return PlasticDirection(DirectionMotor(metadata.seed), seed=metadata.seed,
        feature_width=metadata.feature_width, max_step=metadata.max_step, max_fast=metadata.max_fast)


def save_checkpoint(path: Path, model: PlasticDirection, *, update: int, seed: int) -> None:
    metadata = CheckpointMetadata(update=update, seed=seed, feature_width=model.feature_width,
                                  max_step=model.max_step, max_fast=model.max_fast)
    expected, actual = _model(metadata).state_dict(), model.state_dict()
    if actual.keys() != expected.keys():
        raise ValueError("结构参数键不匹配")
    weights: dict[str, np.ndarray] = {}
    for key, value in actual.items():
        if (value.shape != expected[key].shape or value.dtype != torch.float32
                or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
            raise ValueError(f"结构参数无效：{key}")
        weights[key] = value.detach().numpy().copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, version=metadata.version, update=np.int64(metadata.update),
            seed=np.int64(metadata.seed), feature_width=np.int64(metadata.feature_width),
            max_step=np.float64(metadata.max_step), max_fast=np.float64(metadata.max_fast), **weights)


def _integer(archive: NpzFile, key: str) -> int:
    value = archive[key]
    if value.shape != () or value.dtype != np.dtype("int64"):
        raise ValueError(f"元数据必须为int64标量：{key}")
    return int(value)


def _real(archive: NpzFile, key: str) -> float:
    value = archive[key]
    if value.shape != () or value.dtype != np.dtype("float64") or not bool(np.isfinite(value)):
        raise ValueError(f"元数据必须为有限float64标量：{key}")
    return float(value)


def load_checkpoint(path: Path, *, trainable: bool = False) -> PlasticDirection:
    if type(trainable) is not bool:
        raise ValueError("trainable必须显式为布尔值")
    with zipfile.ZipFile(path) as container:
        names = container.namelist()
        if len(names) != len(set(names)):
            raise ValueError("检查点包含重复归档键")
    with np.load(path, allow_pickle=False) as archive:
        metadata_keys = set(CheckpointMetadata.model_fields)
        if not metadata_keys.issubset(archive.files):
            raise ValueError("检查点缺少结构元数据")
        version = archive["version"]
        if version.shape != () or version.dtype.kind != "U":
            raise ValueError("版本必须为Unicode字符串标量")
        metadata = CheckpointMetadata(version=str(version), update=_integer(archive, "update"),
            seed=_integer(archive, "seed"), feature_width=_integer(archive, "feature_width"),
            max_step=_real(archive, "max_step"), max_fast=_real(archive, "max_fast"))
        model = _model(metadata)
        expected = model.state_dict()
        allowed = set(expected) | metadata_keys
        if set(names) != {f"{key}.npy" for key in allowed}:
            raise ValueError("检查点参数键缺失或包含未知内容")
        weights: dict[str, torch.Tensor] = {}
        for key, target in expected.items():
            value = archive[key]
            if value.dtype != np.dtype("float32") or value.shape != tuple(target.shape) or not np.isfinite(value).all():
                raise ValueError(f"结构参数类型、形状或数值无效：{key}")
            weights[key] = torch.from_numpy(value.copy())
        model.load_state_dict(weights, strict=True)
    model.requires_grad_(trainable)
    model.train(trainable)
    model.motor.freeze()
    return model
