"""真实后到反馈、独立随机流及外层策略梯度的基础课程展开。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..feedback_cues import CueEpisode
from ..plastic_direction import PlasticDirection, PlasticState

Mode = Literal["learned", "off", "always", "matched"]


def streams(seed: int, batch: int) -> tuple[torch.Generator, ...]:
    return tuple(torch.Generator().manual_seed(seed + 1009 * i) for i in range(batch))


def matched_schedule(accepted: torch.Tensor, seed: int) -> torch.Tensor:
    if accepted.ndim != 2 or accepted.dtype != torch.bool or accepted.shape[0] % 4:
        raise ValueError("接受记录须为完整四帧窗口的布尔矩阵")
    if accepted[torch.arange(len(accepted)) % 4 != 3].any():
        raise ValueError("非判断帧不得含接受记录")
    result = torch.zeros_like(accepted)
    for i, generator in enumerate(streams(seed, accepted.shape[1])):
        count = int(accepted[:, i].sum())
        windows = torch.randperm(len(accepted) // 4, generator=generator)[:count]
        result[windows * 4 + 3, i] = True
    return result


@dataclass(frozen=True)
class CourseTrace:
    rewards: torch.Tensor
    logp: torch.Tensor
    gate_logp: tuple[tuple[int, torch.Tensor], ...]
    probabilities: torch.Tensor
    directions: torch.Tensor
    accepted: torch.Tensor
    modulation: torch.Tensor
    gate_probability: torch.Tensor
    write_norm: torch.Tensor
    fast: torch.Tensor
    eligibility: torch.Tensor
    hidden: torch.Tensor
    final_state: PlasticState


def rollout(model: PlasticDirection, episode: CueEpisode, base: torch.Tensor, *,
            seed: int, mode: Mode = "learned", training: bool = False,
            schedule: torch.Tensor | None = None) -> CourseTrace:
    steps, batch = episode.config.steps, episode.config.batch
    if base.shape != (steps, batch, 2) or mode not in ("learned", "off", "always", "matched"):
        raise ValueError("基础方向或消融方式不匹配")
    if training and mode != "learned":
        raise ValueError("外层训练只接受模型自己的采样决策")
    if mode == "matched":
        if schedule is None or schedule.shape != (steps, batch) or schedule.dtype != torch.bool:
            raise ValueError("等次数消融需要完整布尔日程")
        if schedule[torch.arange(steps) % 4 != 3].any():
            raise ValueError("消融日程不得在非判断帧接受")
    elif schedule is not None:
        raise ValueError("只有等次数消融使用预定接受日程")
    actions, gates = streams(seed, batch), streams(seed + 100_000_000, batch)
    state = model.initial_state(batch)
    rewards, logp, gate_logp, probabilities, directions = [], [], [], [], []
    accepted, modulation, gate_probability, write_norm = [], [], [], []
    fast, eligibility, hidden = [], [], []
    for tick in range(steps):
        observed = episode.input_at(tick)
        step = model.infer(base[tick], observed.features, state, sampled=True, generators=actions)
        reward = episode.feedback(tick, step.directions)
        override = None
        if tick % 4 == 3:
            if mode == "always":
                override = torch.ones(batch, dtype=torch.bool)
            elif mode == "matched":
                assert schedule is not None
                override = schedule[tick]
        feedback = model.feedback(reward, step.state, terminal=tick == steps - 1,
            frozen=mode == "off", sampled_gate=training, generators=gates,
            accepted_override=override)
        state = feedback.state
        rewards.append(reward)
        logp.append(step.log_probability)
        if training and feedback.gate_logp is not None:
            gate_logp.append((tick, feedback.gate_logp))
        probabilities.append(step.probabilities.detach())
        directions.append(step.directions.detach())
        accepted.append(feedback.accepted.detach())
        modulation.append(feedback.modulation.detach())
        gate_probability.append(feedback.gate_probability.detach())
        write_norm.append(feedback.write_norm.detach())
        fast.append(state.fast.detach().clone())
        eligibility.append(state.eligibility.detach().clone())
        hidden.append(state.history[-1].detach().clone())
    return CourseTrace(torch.stack(rewards), torch.stack(logp), tuple(gate_logp),
        torch.stack(probabilities), torch.stack(directions), torch.stack(accepted),
        torch.stack(modulation), torch.stack(gate_probability), torch.stack(write_norm),
        torch.stack(fast), torch.stack(eligibility), torch.stack(hidden), state)


def returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.ndim != 2 or not 0 <= gamma <= 1 or not bool(torch.isfinite(rewards).all()):
        raise ValueError("回报输入无效")
    future = torch.zeros_like(rewards[0])
    result = []
    for reward in reversed(rewards):
        future = reward + gamma * future
        result.append(future)
    return torch.stack(result[::-1]).detach()


def outer_loss(trace: CourseTrace, baseline: torch.Tensor, *, gamma: float = .97,
               write_cost: float = .002) -> tuple[torch.Tensor, torch.Tensor]:
    if baseline.shape != (len(trace.rewards),) or not bool(torch.isfinite(baseline).all()) or write_cost < 0:
        raise ValueError("外层历史回报基线或写入成本无效")
    utility = trace.rewards - write_cost * trace.accepted.to(torch.float32)
    credit = returns(utility, gamma)
    loss = -(trace.logp * (credit - baseline[:, None])).sum()
    for tick, log_probability in trace.gate_logp:
        # 当前方向的奖励先于写入产生；只把后续回报及自身写入成本归给接受决策。
        following = gamma * credit[tick + 1] if tick + 1 < len(credit) else torch.zeros_like(credit[tick])
        value = following - write_cost * trace.accepted[tick].to(torch.float32)
        control = gamma * baseline[tick + 1] if tick + 1 < len(credit) else baseline.new_zeros(())
        loss = loss - (log_probability * (value - control)).sum()
    return loss / trace.rewards.numel(), credit.mean(dim=1)
