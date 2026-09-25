"""无神经网络的局部规则对照：相同传感、意向与接触约束，无训练。"""
from __future__ import annotations
import numpy as np
from ..geometry import Array


class RuleController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed+700001)
        self.updates = 0
        self.last_loss = self.last_delta = self.birth_loss = self.warm_loss = 0.
        self.frozen = True
        self.hidden = np.zeros(16, np.float32)
        self.output = np.zeros(3, np.float32)

    def predict(self, x: Array) -> Array:
        values = np.atleast_2d(x)
        mobility = np.clip(values[:, 2]*2.4/.18, 0., 1.)
        # 与神经版相同的候选动作；规则按局部净空和名义位移选择，不获得外力真值。
        return np.column_stack((values[:, :2]*mobility[:, None], mobility<.999)).astype(np.float32)

    def inspect(self, x: Array) -> Array:
        self.output = self.predict(x)[0].copy()
        return self.output.copy()

    def fingerprint(self) -> str:
        return ''

    @property
    def drift(self) -> float:
        return 0.
