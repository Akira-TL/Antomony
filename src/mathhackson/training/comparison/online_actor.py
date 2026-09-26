"""连续部署只读取真实反馈，接受模型不访问配对分支未来。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.direction.policy import DirectionAction, DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationProposal
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.trust_candidate import CandidateDiagnostics, TrustConfig, TrustDirectionLearner
from mathhackson.training.foraging.update_decision import IndividualUpdateController, UpdateChoice, UpdateDecision, decision_features

Mode = Literal["learned", "skip", "always"]


class UpdateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tick: int
    individual: int
    proposal: AdaptationProposal
    diagnostics: CandidateDiagnostics
    features: list[float]
    prediction: float | None
    eligible: bool
    accepted: bool
    changed: bool
    terminal: bool
    before: list[float]
    after: list[float]


class OnlineForager:
    def __init__(self, policy: MemoryPolicy, motor: DirectionMotor, decision: UpdateDecision,
                 seed: int, mode: Mode, config: TrustConfig) -> None:
        if mode not in ("learned", "skip", "always"):
            raise ValueError("未知在线更新方式")
        self.mode = mode
        self.agent = TrustDirectionLearner(policy, motor, seed, config)
        self.controller = IndividualUpdateController(decision)
        self.frozen = [parameter.detach().clone() for parameter in self.frozen_parameters()]

    def frozen_parameters(self) -> list[torch.nn.Parameter]:
        return [*self.agent.policy.parameters(), *self.agent.motor.parameters(), *self.controller.model.parameters()]

    def act(self, observation: LocalObservation) -> DirectionAction:
        return self.agent.act(observation).action

    def feedback(self, reward: float, observation: LocalObservation, *, terminal: bool,
                 tick: int, individual: int) -> UpdateRecord | None:
        self.agent.feedback(reward, terminal=terminal)
        if not self.agent.ready:
            return None
        proposal = self.agent.propose(observation)
        diagnostics = self.agent.diagnostics
        hidden = self.agent.history[-1].detach().numpy() if self.agent.history else np.zeros(8, dtype=np.float32)
        features = decision_features(observation.vector(), hidden, proposal, diagnostics)
        before = self.agent.weights().tolist()
        if self.mode == "learned":
            choice = self.controller.resolve(self.agent, observation)
            prediction = choice.predicted_reward_difference
        else:
            eligible = not terminal and bool(np.any(proposal.delta))
            accepted = self.mode == "always" and eligible
            changed = self.agent.resolve(proposal, accept=(accepted,))
            choice = UpdateChoice(0., accepted, changed, eligible)
            prediction = None
        return UpdateRecord(tick=tick, individual=individual, proposal=proposal, diagnostics=diagnostics,
            features=features.tolist(), prediction=prediction, eligible=choice.eligible,
            accepted=choice.accepted, changed=choice.changed, terminal=terminal,
            before=before, after=self.agent.weights().tolist())

    def check_frozen(self) -> None:
        if any(not before.equal(after) for before, after in zip(self.frozen, self.frozen_parameters(), strict=True)):
            raise AssertionError("基础、记忆、动作或接受参数被在线改写")
        if self.mode == "skip" and (self.agent.writes or np.any(self.agent.weights())):
            raise AssertionError("跳过组出现参数写入")

    def save(self, path: Path) -> None:
        self.check_frozen()
        self.agent.save(path)
