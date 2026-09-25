"""受自指权重矩阵启发的工程原型；任务回报训练行动与写入控制。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Phase = Literal["motor", "meta", "autonomous"]
Group = Literal["action", "gate", "query", "key", "rate"]
GROUPS: tuple[tuple[Group, str, int, int], ...] = (
    ("action", "行动输出", 0, 2), ("gate", "写入判断", 2, 3),
    ("query", "读取位置", 3, 19), ("key", "写入位置", 19, 35),
    ("rate", "写入幅度", 35, 39),
)
INPUTS = ("食物方向余弦", "食物方向正弦", "食物距离", "上次前进", "上次转向",
          "上次奖励", "上次碰撞", "上次进展", "离巢标记左", "离巢标记中",
          "离巢标记右", "食物标记左", "食物标记中", "食物标记右", "回合进度", "常数")
MOTOR_ACTION_INPUTS: tuple[tuple[int, ...], tuple[int, ...]] = ((0, 2, 15), (1,))


@dataclass
class Decision:
    move: bool
    turn: float
    move_probability: float
    write_probability: float
    requested_write: bool
    wrote: bool
    action_logp: Tensor
    gate_logp: Tensor
    entropy: Tensor
    turn_loss: Tensor


class SelfModifyingPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.initial = torch.from_numpy(self.rng.normal(0, .12, (39, 16)).astype(np.float32))
        for row, columns in enumerate(MOTOR_ACTION_INPUTS):
            inactive = [column for column in range(16) if column not in columns]
            self.initial[row, inactive] = 0.
        self.base = torch.nn.Parameter(self.initial.clone())
        self.weights: Tensor = self.base
        self.manual_frozen: set[Group] = set()
        self.phase: Phase = "motor"
        self.self_delta = torch.zeros_like(self.base)
        self.outer_delta = torch.zeros_like(self.base)
        self.self_updates = 0
        self.outer_updates = 0

    def mask(self) -> Tensor:
        mask = torch.ones_like(self.base)
        if self.phase == "motor":
            mask[:2] = 0
            for row, columns in enumerate(MOTOR_ACTION_INPUTS):
                mask[row, list(columns)] = 1
        for group, _, start, end in GROUPS:
            if group in self.manual_frozen or (self.phase == "motor" and group != "action"):
                mask[start:end] = 0
        return mask

    def decide(self, observation: np.ndarray) -> Decision:
        if observation.shape != (16,) or not np.isfinite(observation).all():
            raise ValueError("观察必须是 16 个有限值")
        training = self.phase != "autonomous"
        with torch.set_grad_enabled(training):
            output = self.weights @ torch.from_numpy(observation.astype(np.float32))
            move_dist = torch.distributions.Bernoulli(logits=output[0])
            turn_dist = torch.distributions.Normal(output[1], .35)
            gate_dist = torch.distributions.Bernoulli(logits=output[2])
            if training:
                move = bool(self.rng.random() < float(move_dist.probs.detach()))
                raw_turn = float(self.rng.normal(float(output[1].detach()), .35))
                requested = bool(self.rng.random() < float(gate_dist.probs.detach()))
            else:
                move = bool(move_dist.probs >= .5)
                raw_turn = float(output[1])
                requested = bool(gate_dist.probs >= .5)
            # 扩大输出对可用角度的覆盖；最终角度仍由模型给出，环境上限不变。
            turn = float(np.tanh(3. * raw_turn))
            action_logp = move_dist.log_prob(torch.tensor(float(move))) + turn_dist.log_prob(torch.tensor(raw_turn))
            gate_logp = gate_dist.log_prob(torch.tensor(float(requested)))
            turn_target = torch.tensor((4. / 3.) * float(observation[1]), dtype=output.dtype)
            turn_loss = torch.square(output[1] - turn_target)
            self.self_delta.zero_()
            wrote = False
            # 本步行动先由旧权重产生；写入只能影响下一次推理，包括写入控制本身。
            if self.phase != "motor" and requested:
                query = torch.softmax(output[3:19], dim=0)
                key = torch.softmax(output[19:35], dim=0)
                rates = .08 * torch.sigmoid(output[35:39])
                row_rates = torch.cat((rates[0].expand(3), rates[1].expand(16),
                                       rates[2].expand(16), rates[3].expand(4)))
                delta = torch.outer(self.weights @ query - self.weights @ key, key)
                delta = delta * row_rates[:, None] * self.mask()
                updated = torch.clamp(self.weights + delta, -6., 6.)
                # 冻结参数不能因任何数值截断而改变。
                updated = torch.where(self.mask().bool(), updated, self.weights)
                self.self_delta = (updated - self.weights).detach()
                wrote = bool(torch.count_nonzero(self.self_delta))
                self.weights = updated
                self.self_updates += int(wrote)
            return Decision(move, turn, float(move_dist.probs.detach()), float(gate_dist.probs.detach()),
                            requested, wrote, action_logp, gate_logp, move_dist.entropy() + turn_dist.entropy(), turn_loss)

    def finish(self, decisions: list[Decision], rewards: list[float], baseline: float, optimize: bool) -> None:
        final = self.weights.detach().clone()
        self.outer_delta.zero_()
        if optimize and self.phase != "autonomous" and decisions:
            returns: list[float] = []
            value = 0.
            for reward in reversed(rewards):
                value = reward + .97 * value
                returns.append(value)
            returns.reverse()
            loss = torch.zeros(())
            for i, decision in enumerate(decisions):
                loss = loss - decision.action_logp * (returns[i] - baseline) - .003 * decision.entropy
                if self.phase == "motor":
                    loss = loss + 8. * decision.turn_loss
                if self.phase == "meta":
                    # 判断写入不影响本步动作，故从下一步的实际回报分配信用。
                    future = returns[i + 1] if i + 1 < len(returns) else 0.
                    loss = loss - decision.gate_logp * (future - baseline)
            loss = loss / len(decisions)
            loss.backward()
            if self.base.grad is not None:
                grad = self.base.grad * self.mask()
                norm = torch.linalg.vector_norm(grad)
                grad = grad / torch.clamp(norm, min=1.)
                proposal = torch.clamp(final - .05 * grad, -6., 6.)
                updated = torch.where(self.mask().bool(), proposal, final)
                if not torch.isfinite(updated).all():
                    raise ValueError("非有限外部更新已拒绝")
                self.outer_delta = updated - final
                self.outer_updates += int(bool(torch.count_nonzero(self.outer_delta)))
                final = updated
        with torch.no_grad():
            self.base.copy_(final)
        self.base.grad = None
        self.weights = self.base

    def values(self) -> list[list[float]]:
        return self.weights.detach().tolist()
