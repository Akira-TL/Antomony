"""可移植的四连接基础动作快照，只能加载到全新单蚁模型。"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .recurrent import INPUT_WIDTH, LEGACY_MODEL_VERSION, MODEL_VERSION, RecurrentPolicy


class MotorCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    format: Literal["motor-v1"] = "motor-v1"
    model_version: Literal["sparse-memory-v2", "sparse-memory-v3"] = MODEL_VERSION
    source_session: str = Field(pattern=r"^[0-9a-f]{12}$")
    source_episode: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    move_alignment: float = Field(ge=-2., le=2.)
    move_distance: float = Field(ge=-2., le=2.)
    move_bias: float = Field(ge=-2., le=2.)
    turn_direction: float = Field(ge=-2., le=2.)

    def motor_matrix(self) -> torch.Tensor:
        result = torch.zeros(2, INPUT_WIDTH)
        result[0, 0] = self.move_alignment
        result[0, 2] = self.move_distance
        result[0, 15] = self.move_bias
        result[1, 1] = self.turn_direction
        return result


def extract_motor_checkpoint(path: Path) -> MotorCheckpoint:
    with np.load(path, allow_pickle=False) as archive:
        if "model_version" not in archive or str(archive["model_version"]) not in {
                LEGACY_MODEL_VERSION, MODEL_VERSION}:
            raise ValueError("来源不是当前版本的循环模型检查点")
        source_version = str(archive["model_version"])
        motor = np.asarray(archive["motor"], dtype=np.float32)
    if motor.shape != (2, INPUT_WIDTH) or not np.isfinite(motor).all():
        raise ValueError("来源动作矩阵形状或数值无效")
    active = np.zeros_like(motor, dtype=bool)
    active[0, [0, 2, 15]] = True
    active[1, 1] = True
    if np.any(motor[~active] != 0):
        raise ValueError("来源包含未约定的动作连接")
    if not path.stem.startswith("episode-"):
        raise ValueError("来源检查点名称无效")
    return MotorCheckpoint(
        model_version=source_version,
        source_session=path.parent.name,
        source_episode=int(path.stem.removeprefix("episode-")),
        source_sha256=sha256(path.read_bytes()).hexdigest(),
        move_alignment=float(motor[0, 0]),
        move_distance=float(motor[0, 2]),
        move_bias=float(motor[0, 15]),
        turn_direction=float(motor[1, 1]),
    )


def load_motor_checkpoint(policy: RecurrentPolicy, path: Path, *, freeze_motor: bool = True) -> MotorCheckpoint:
    if policy.phase != "motor" or policy.outer_updates or policy.self_updates:
        raise ValueError("动作快照只能加载到尚未训练的新模型")
    checkpoint = MotorCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    with torch.no_grad():
        policy.motor.copy_(checkpoint.motor_matrix())
    policy.phase = "memory" if freeze_motor else "motor"
    policy.write_mode = "off"
    policy.reset_state()
    return checkpoint
