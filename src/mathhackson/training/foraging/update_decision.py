"""已发生反馈驱动的更新决策；未来后果只能作为离线训练标签。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .adaptation import AdaptationProposal
from .environment import LocalObservation
from .trust_candidate import CandidateDiagnostics, TrustDirectionLearner

INPUT_WIDTH = 140
INPUT_VERSION = "local-update-decision-v1"


def decision_features(observation: np.ndarray, hidden: np.ndarray,
                      proposal: AdaptationProposal, diagnostics: CandidateDiagnostics) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float64)
    hidden = np.asarray(hidden, dtype=np.float64)
    delta = np.asarray(proposal.delta, dtype=np.float64)
    if observation.shape != (78,) or hidden.shape != (8,) or delta.shape != (45,):
        raise ValueError("更新决策需要78维观测、8维已有记忆和45维候选")
    statistics = np.asarray((proposal.gradient_norm, proposal.mean_reward, proposal.minimum_reward,
                             proposal.novel_response, proposal.steps, diagnostics.mean_kl,
                             diagnostics.maximum_kl, diagnostics.surrogate_gain, diagnostics.backtracks))
    values = np.concatenate((observation, hidden, delta, statistics))
    if not np.isfinite(values).all():
        raise ValueError("更新决策输入必须全部有限")
    # 固定变换不从测试数据估计均值或方差，也不抹掉浓度的绝对差异。
    return (values / (1. + np.abs(values))).astype(np.float32)


class UpdateDecision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.value = nn.Linear(INPUT_WIDTH, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        self.updates = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in (1, 2) or features.shape[-1] != INPUT_WIDTH or not bool(torch.isfinite(features).all()):
            raise ValueError("接受决策输入维度错误或包含非有限值")
        return self.value(features).squeeze(-1)

    def save(self, path: Path) -> None:
        with path.open("xb") as stream:
            np.savez(stream, version=INPUT_VERSION, updates=self.updates,
                     **{key: value.detach().cpu().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> UpdateDecision:
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != INPUT_VERSION or data["updates"].shape != () or int(data["updates"]) < 0:
                raise ValueError("更新决策检查点不兼容")
            weights = {}
            for key, target in model.state_dict().items():
                value = data[key]
                if value.shape != tuple(target.shape) or not np.isfinite(value).all():
                    raise ValueError("更新决策权重维度错误或非有限")
                weights[key] = torch.from_numpy(value.copy()).to(dtype=target.dtype)
            model.load_state_dict(weights)
            model.updates = int(data["updates"])
        return model


def train_decision_step(model: UpdateDecision, optimizer: torch.optim.Optimizer,
                        features: torch.Tensor, reward_differences: torch.Tensor) -> float:
    if (features.ndim != 2 or features.shape[0] == 0
            or reward_differences.shape != (features.shape[0],)
            or not bool(torch.isfinite(reward_differences).all())):
        raise ValueError("离线训练需要非空输入及每条配对的有限奖励差")
    owned = {id(parameter) for parameter in model.parameters()}
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimized) != len(owned) or {id(parameter) for parameter in optimized} != owned:
        raise ValueError("优化器只能持有当前个体接受模型的全部参数")
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features.detach()), reward_differences.detach())
    if not bool(torch.isfinite(loss)):
        raise ValueError("接受决策训练损失非有限")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.updates += 1
    return float(loss.detach())


@dataclass(frozen=True)
class UpdateChoice:
    predicted_reward_difference: float
    accepted: bool
    changed: bool
    eligible: bool


class IndividualUpdateController:
    def __init__(self, model: UpdateDecision) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    def resolve(self, agent: TrustDirectionLearner, observation: LocalObservation, *, continuing_after_death: bool = False) -> UpdateChoice:
        proposal = agent.proposal
        if proposal is None or agent.awaiting_feedback:
            raise ValueError("接受决策只处理已取得反馈的当前候选")
        hidden = agent.history[-1].detach().numpy() if agent.history else np.zeros(8, dtype=np.float32)
        features = decision_features(observation.vector(), hidden, proposal, agent.diagnostics)
        with torch.inference_mode():
            prediction = float(self.model(torch.from_numpy(features)))
        if not np.isfinite(prediction):
            raise ValueError("接受决策预测非有限，不得写入参数")
        eligible = (not agent.terminal or continuing_after_death) and bool(np.any(proposal.delta))
        accepted = eligible and prediction > 0.
        changed = agent.resolve(proposal, accept=(accepted,))
        return UpdateChoice(prediction, accepted, changed, eligible)
