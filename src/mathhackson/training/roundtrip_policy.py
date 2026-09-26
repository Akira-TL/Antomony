"""载入完整基础模型后，训练局部循迹、双通道释放与条件自修改。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .recurrent import Action, HIDDEN_WIDTH, MODEL_VERSION, RecurrentPolicy, Write

FOUNDATION_NAMES = ("motor", "input_weights", "hidden_weights", "hidden_bias",
                    "action_weights", "gate_weights", "write_weights")
RELEASE_FAST_LIMIT = .2
RELEASE_FAST_STEP = .012


class RoundTripPolicy(RecurrentPolicy):
    def __init__(self, seed: int, foundation: Path) -> None:
        super().__init__(seed, input_width=17)
        self.release_weights = torch.nn.Parameter(torch.zeros(2, 4))
        with torch.no_grad():
            self.release_weights[:, 0] = torch.tensor([-4., 4.])
            self.release_weights[:, 3] = torch.tensor([2., -2.])
        self.release_write_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .04, (2, HIDDEN_WIDTH + 1)).astype(np.float32)))
        self.parameters += (self.release_weights, self.release_write_weights)
        self.outer_delta.extend(torch.zeros_like(value) for value in self.parameters[-2:])
        self.release_fast = torch.zeros(2)
        self.release_fast_delta = torch.zeros(2)
        self._load_foundation(foundation)
        self.phase = "memory"
        self.write_mode = "off"

    def _load_foundation(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            if "model_version" not in archive or str(archive["model_version"]) != MODEL_VERSION:
                raise ValueError("基础检查点版本不兼容")
            values = [np.asarray(archive[name], np.float32).copy() for name in FOUNDATION_NAMES]
        expected = [(2, 16), (HIDDEN_WIDTH, 16), (HIDDEN_WIDTH, 32), (HIDDEN_WIDTH,),
                    (2, HIDDEN_WIDTH), (HIDDEN_WIDTH + 1,), (3, HIDDEN_WIDTH + 1)]
        if any(value.shape != shape or not np.isfinite(value).all()
               for value, shape in zip(values, expected, strict=True)):
            raise ValueError("基础检查点参数形状或数值无效")
        with torch.no_grad():
            self.motor[:, :16].copy_(torch.from_numpy(values[0]))
            self.motor[:, 16].zero_()
            self.input_weights[:, :16].copy_(torch.from_numpy(values[1]))
            self.input_weights[:, 16].zero_()
            for parameter, value in zip(self.parameters[2:7], values[2:], strict=True):
                parameter.copy_(torch.from_numpy(value))

    def reset_state(self) -> None:
        super().reset_state()
        self.release_fast = torch.zeros(2)
        self.release_fast_delta = torch.zeros(2)

    def trainable_masks(self) -> list[torch.Tensor]:
        masks = [torch.zeros_like(value) for value in self.parameters]
        if self.phase in {"memory", "adaptive"}:
            masks[1][:, 8:14] = 1
            masks[1][:, 16] = 1
            masks[4].fill_(1)
            masks[7].fill_(1)
        if self.phase == "adaptive":
            masks[5].fill_(1)
            masks[6].fill_(1)
            masks[8].fill_(1)
        return masks

    def decide(self, observation: np.ndarray) -> Action:
        action = super().decide(observation)
        with torch.set_grad_enabled(self.phase != "autonomous"):
            features = torch.tensor([observation[16], observation[8], observation[11], 1.],
                                    dtype=torch.float32)
            logits = self.release_weights @ features + self.release_fast
            logits = 3. * torch.tanh(logits / 3.)
            releases = torch.distributions.Bernoulli(logits=logits)
            probabilities = releases.probs.detach().tolist()
            selected = ([bool(self.rng.random() < probability) for probability in probabilities]
                        if self.phase != "autonomous" else
                        [probability >= .5 for probability in probabilities])
            release_values = torch.tensor(selected, dtype=torch.float32)
            action.action_logp = action.action_logp + releases.log_prob(release_values).sum()
            action.entropy = action.entropy + releases.entropy().sum()
            action.release_home, action.release_food = selected
            action.release_home_probability, action.release_food_probability = probabilities
        return action

    def observe_result(self, observation: np.ndarray, *, terminal: bool = False) -> Write:
        write = super().observe_result(observation, terminal=terminal)
        self.release_fast_delta = torch.zeros_like(self.release_fast)
        if write.requested and not terminal and self.phase in {"adaptive", "autonomous"}:
            with torch.set_grad_enabled(self.phase == "adaptive"):
                features = torch.cat((self.hidden, torch.ones(1)))
                proposed = torch.clamp(
                    self.release_fast + torch.tanh(self.release_write_weights @ features) * RELEASE_FAST_STEP,
                    -RELEASE_FAST_LIMIT, RELEASE_FAST_LIMIT)
                self.release_fast_delta = (proposed - self.release_fast).detach()
                self.release_fast = proposed
                if not write.wrote and bool(torch.count_nonzero(self.release_fast_delta)):
                    self.self_updates += 1
                    write.wrote = True
        return write

    def finish(self, actions: list[Action], writes: list[Write], rewards: list[float],
               baseline: float, optimize: bool) -> None:
        super().finish(actions, writes, rewards, baseline, optimize)
        self.release_fast = self.release_fast.detach()
