"""基于已到达反馈与已见行动分布的候选；不使用模拟未来。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydantic import Field
import torch

from mathhackson.training.direction.policy import DirectionMotor
from .adaptation import AdaptationConfig, AdaptationProposal, NovelDirectionLearner
from .environment import LocalObservation
from .policy import DirectionDecision, ForagingPolicy
from .reward import DIRECTIONS, direction_distribution, discounted_returns


class TrustConfig(AdaptationConfig):
    learning_rate: float = Field(default=1., gt=0.)
    maximum_step_norm: float = Field(default=4., gt=0.)
    maximum_residual_norm: float = Field(default=4., gt=0.)
    maximum_mean_kl: float = Field(default=.02, gt=0.)
    maximum_state_kl: float = Field(default=.08, gt=0.)
    damping: float = Field(default=.01, gt=0.)
    backtracks: int = Field(default=12, ge=1)


@dataclass(frozen=True)
class CandidateDiagnostics:
    mean_kl: float = 0.
    maximum_kl: float = 0.
    surrogate_gain: float = 0.
    backtracks: int = 0


def rotation_probabilities(weights: torch.Tensor, bases: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    angles = math.pi * torch.tanh(features @ weights)
    c, s = torch.cos(angles), torch.sin(angles)
    adjusted = torch.stack((c * bases[:, 0] - s * bases[:, 1],
                            s * bases[:, 0] + c * bases[:, 1]), dim=-1)
    return direction_distribution(adjusted).probs


class TrustDirectionLearner(NovelDirectionLearner):
    adapter_kind = "direction-trust"

    def __init__(self, policy: ForagingPolicy, motor: DirectionMotor, seed: int,
                 config: TrustConfig | None = None) -> None:
        if config is not None and not isinstance(config, TrustConfig):
            raise ValueError("分布约束候选需要专用配置")
        super().__init__(policy, motor, seed, config or TrustConfig())
        self.base_directions: list[torch.Tensor] = []
        self.novel_features: list[torch.Tensor] = []
        self.action_indices: list[int] = []
        self.diagnostics = CandidateDiagnostics()

    def act(self, observation: LocalObservation) -> DirectionDecision:
        history = tuple(self.history)
        result = super().act(observation)
        tensor = torch.from_numpy(observation.vector())
        with torch.no_grad():
            self.base_directions.append(self.policy(tensor, history)[0].detach().clone())
        self.novel_features.append(tensor[:72].reshape(9, 8)[:, 3:].flatten().clone())
        self.action_indices.append(int((DIRECTIONS @ torch.from_numpy(result.direction)).argmax()))
        return result

    def propose(self, next_observation: LocalObservation) -> AdaptationProposal:
        if not self.ready or self.proposal is not None:
            raise ValueError("尚无完整后到反馈，或已有待定提案")
        with torch.no_grad():
            bootstrap = 0. if self.terminal else float(self.predict(
                torch.from_numpy(next_observation.vector()), tuple(self.history))[2])
        advantage = discounted_returns(self.rewards, bootstrap, self.config.gamma) - torch.stack(self.values)
        loss = -(torch.stack(self.log_probabilities) * advantage).mean()
        gradient = torch.autograd.grad(loss, self.offset)[0].detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("非有限梯度不能形成修改")
        bases, features = torch.stack(self.base_directions), torch.stack(self.novel_features)
        current = self.offset.detach().clone()
        old = rotation_probabilities(current, bases, features).detach()
        delta = torch.zeros_like(current)
        self.diagnostics = CandidateDiagnostics()
        if bool(gradient.any()):
            jacobian = torch.autograd.functional.jacobian(
                lambda value: rotation_probabilities(value, bases, features), current, vectorize=True).double()
            fisher = torch.einsum("tap,taq,ta->pq", jacobian, jacobian, 1. / old.double()) / len(bases)
            regularizer = self.config.damping * max(float(fisher.trace()) / current.numel(), 1e-6)
            metric = fisher + regularizer * torch.eye(current.numel(), dtype=torch.float64)
            step = torch.linalg.solve(metric, -gradient.double())
            curvature = float(step @ metric @ step)
            if curvature > 0. and math.isfinite(curvature):
                step *= self.config.learning_rate * math.sqrt(2. * self.config.maximum_mean_kl / curvature)
                step *= min(1., self.config.maximum_step_norm / max(float(step.norm()), 1e-12))
                delta = self._backtrack(current, step.float(), old, bases, features, advantage)
        self.proposal = AdaptationProposal(self.decisions + 1, tuple(float(x) for x in delta),
                                           float(gradient.norm()), float(np.mean(self.rewards)), min(self.rewards),
                                           max(self.responses), len(self.rewards))
        return self.proposal

    def _backtrack(self, current: torch.Tensor, step: torch.Tensor, old: torch.Tensor,
                   bases: torch.Tensor, features: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        rows, choices = torch.arange(len(bases)), torch.tensor(self.action_indices)
        for attempt in range(self.config.backtracks):
            delta = step * (.5 ** attempt)
            if float((torch.from_numpy(self.parameter.fast) + delta).norm()) >= self.config.maximum_residual_norm * (1. - 1e-6):
                continue
            new = rotation_probabilities(current + delta, bases, features)
            divergence = (old * (old.log() - new.log())).sum(-1).clamp_min(0.)
            gain = float(((new[rows, choices] / old[rows, choices] - 1.) * advantage).mean())
            mean_kl, maximum_kl = float(divergence.mean()), float(divergence.max())
            if (0. < gain and mean_kl <= self.config.maximum_mean_kl
                    and maximum_kl <= self.config.maximum_state_kl):
                self.diagnostics = CandidateDiagnostics(mean_kl, maximum_kl, gain, attempt)
                return delta
        self.diagnostics = CandidateDiagnostics(backtracks=self.config.backtracks)
        return torch.zeros_like(current)

    def resolve(self, proposal: AdaptationProposal, *, accept: tuple[bool, ...]) -> bool:
        changed = super().resolve(proposal, accept=accept)
        self.base_directions.clear()
        self.novel_features.clear()
        self.action_indices.clear()
        return changed
