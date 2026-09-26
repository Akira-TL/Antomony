"""无危险局部片段的方向反馈课程；隐藏关联只用于事后计算奖励。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from .curriculum import signal_batch


Condition = Literal["reference", "association", "transient"]
CONDITIONS: tuple[Condition, ...] = ("reference", "association", "transient")


class CueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int = Field(default=96, ge=8, le=256)
    batch: int = Field(default=16, ge=1, le=256)
    cues: int = Field(default=3, ge=2, le=4)

    @model_validator(mode="after")
    def whole_windows(self) -> CueConfig:
        if self.steps % 4:
            raise ValueError("基础课程长度须包含完整的四反馈帧窗口")
        return self


def plastic_features(observation: torch.Tensor, *, channels: Literal["basic", "novel"]) -> torch.Tensor:
    """同一共享更新器处理不同通道；通道编号不作为其输入。"""
    if observation.shape[-1] != 78 or not bool(torch.isfinite(observation).all()):
        raise ValueError("需要有限的78维局部观察")
    receptors = observation[..., :72].reshape(*observation.shape[:-1], 9, 8)
    if channels == "basic":
        selected = torch.cat((receptors[..., :3], torch.zeros_like(receptors[..., 3:5])), dim=-1)
    elif channels == "novel":
        selected = receptors[..., 3:]
    else:
        raise ValueError("未知通道选择")
    contrast = (selected - selected.mean(dim=-2, keepdim=True)) / selected.abs().amax(dim=-2, keepdim=True).clamp_min(.001)
    return contrast.flatten(-2).detach()


@dataclass(frozen=True)
class CueInput:
    observation: torch.Tensor
    features: torch.Tensor


@dataclass(frozen=True)
class CueEpisode:
    config: CueConfig
    conditions: tuple[Condition, ...]
    observations: torch.Tensor
    targets: torch.Tensor
    clean_targets: torch.Tensor
    cue_ids: torch.Tensor
    reward_flips: torch.Tensor

    def input_at(self, tick: int) -> CueInput:
        if not 0 <= tick < self.config.steps:
            raise ValueError("课程帧越界")
        observation = self.observations[tick].clone()
        return CueInput(observation, plastic_features(observation, channels="basic"))

    def feedback(self, tick: int, chosen_directions: torch.Tensor) -> torch.Tensor:
        if not 0 <= tick < self.config.steps:
            raise ValueError("课程反馈帧越界")
        if (chosen_directions.shape != (self.config.batch, 2)
                or not bool(torch.isfinite(chosen_directions).all())
                or not torch.allclose(chosen_directions.norm(dim=-1), torch.ones(self.config.batch), atol=1e-5, rtol=0.)):
            raise ValueError("奖励只能对应实际选择的单位方向")
        return (chosen_directions.detach() * self.targets[tick]).sum(dim=-1)


def cue_episode(seed: int, config: CueConfig, *, condition: Condition | None = None) -> CueEpisode:
    if condition is not None and condition not in CONDITIONS:
        raise ValueError("未知课程条件")
    # 观察生成与隐藏关联使用分开的随机流，避免通过生成顺序泄露条件。
    observed_seed, hidden_seed = np.random.SeedSequence(seed).spawn(2)
    visible, hidden = np.random.default_rng(observed_seed), np.random.default_rng(hidden_seed)
    templates, axes = signal_batch(visible, config.batch * config.cues, budgets=True)
    templates = templates.reshape(config.batch, config.cues, 78)
    axes = axes.reshape(config.batch, config.cues, 2)
    ids = visible.integers(0, config.cues, (config.steps // 4, config.batch)).repeat(4, axis=0)
    conditions = tuple(condition or CONDITIONS[int(hidden.integers(len(CONDITIONS)))] for _ in range(config.batch))
    polarities = hidden.choice(np.asarray([-1., 1.], np.float32), (config.batch, config.cues))
    flips = np.zeros((config.steps, config.batch), dtype=np.bool_)
    for i, selected in enumerate(conditions):
        if selected == "reference":
            polarities[i] = 1.
        if selected == "transient":
            # 反馈变化无输入标记，时点随机；每次仅一帧，不定义为新的来源类别。
            positions = hidden.choice(np.arange(config.steps // 2, config.steps), 2, replace=False)
            flips[positions, i] = True
    batches = np.arange(config.batch)
    observations = torch.stack([templates[batches, row] for row in ids])
    clean = torch.stack([axes[batches, row] * torch.from_numpy(polarities[batches, row, None]) for row in ids])
    targets = clean * torch.from_numpy(np.where(flips, -1., 1.).astype(np.float32)).unsqueeze(-1)
    return CueEpisode(config, conditions, observations, targets, clean, torch.from_numpy(ids), torch.from_numpy(flips))
