"""保存可冻结加载的动作权重；不包含其他个体的运行状态。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .policy import DirectionMotor, INPUT_NAMES, MODEL_VERSION, WIDTH


class MotorSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["direction-motor-v1"] = MODEL_VERSION
    inputs: tuple[str, str] = INPUT_NAMES
    width: Literal[16] = WIDTH
    seed: int
    updates: int = Field(ge=0)
    move_loss: float | None = None
    turn_loss: float | None = None


def save_motor(path: Path, model: DirectionMotor, metadata: MotorSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, metadata=metadata.model_dump_json(),
            encoder_weight=model.encoder.weight.detach().numpy(),
            encoder_bias=model.encoder.bias.detach().numpy(),
            readout_weight=model.readout.weight.detach().numpy(),
            readout_bias=model.readout.bias.detach().numpy(),
        )


def load_motor(path: Path) -> tuple[DirectionMotor, MotorSnapshot]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = MotorSnapshot.model_validate_json(str(archive["metadata"].item()))
        if metadata.inputs != INPUT_NAMES:
            raise ValueError("检查点不是纯方向输入")
        model = DirectionMotor(metadata.seed)
        pairs = ((model.encoder.weight, "encoder_weight"),
                 (model.encoder.bias, "encoder_bias"),
                 (model.readout.weight, "readout_weight"),
                 (model.readout.bias, "readout_bias"))
        with torch.no_grad():
            for parameter, name in pairs:
                value = archive[name]
                if value.shape != tuple(parameter.shape) or not np.isfinite(value).all():
                    raise ValueError(f"检查点参数 {name} 形状或数值无效")
                parameter.copy_(torch.from_numpy(value))
    return model.freeze(), metadata
