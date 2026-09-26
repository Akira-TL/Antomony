"""固定直线路径返巢诊断；人工全域信号仅作正对照，不进入训练。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.rules import LocalRuleController, local_gradient
from mathhackson.training.foraging.signals import SignalSource, receptor_points


@dataclass(frozen=True)
class Probe:
    treatment: str
    carrying: bool
    returned: bool
    distance: float
    forward_steps: int
    stopped_steps: int
    backward_gradients: int
    sampled_gradients: int
    initial_wrong_gradient_fraction: float


def prepare(treatment: str, carrying: bool) -> ColonyEnvironment:
    bounded = treatment.startswith("bounded-")
    nest = treatment in ("nest-source", "bounded-initial-nest")
    config = ColonyConfig(ants=1, exploration_steps=30, reserve_steps=160,
                          trail_profile="bounded-local-v2" if bounded else "additive-cell-v1",
                          nest_signal_strength=float(nest))
    env = ColonyEnvironment(9701, config)
    env.stock = 0
    ant = env.ants[0]
    ant.heading = 0.
    if treatment == "bounded-no-initial":
        env.signals.trails.values.fill(0.)
    while ant.exploration_left:
        env.step([DirectionAction(True, 0., 1.)])
    ant.carrying = carrying
    if treatment == "no-return-deposit":
        deposit = env.signals.trails.deposit

        def only_food(position: np.ndarray, channel: int, amount: float) -> None:
            if channel != 0:
                deposit(position, channel, amount)

        env.signals.trails.deposit = only_food
    elif treatment == "reverse-at-start":
        ant.heading = math.pi
    elif treatment == "ideal-source":
        env.signals.trails.enabled = False
        env.extra_sources = (SignalSource(0., 0., 14., 1., (0., 1., 0., 0., 0., 0., 0., 0.)),)
    return env


def measure(treatment: str, carrying: bool) -> Probe:
    env = prepare(treatment, carrying)
    ant = env.ants[0]
    gradients = [local_gradient(env.signals.sample(
        receptor_points(np.asarray([x, 0.], dtype=np.float32), 0.), env.sources())[:, 1])[0]
        for x in np.linspace(.7, 5.5, 49)]
    controller = LocalRuleController(9701)
    forward = stopped = backward = samples = 0
    while not env.done and not ant.budget_returns:
        obs = env.observation(0)
        gradient = local_gradient(obs.receptors[:, 1])
        # 巢穴坐标仅作事后诊断，不进入控制器输入。
        c, s = math.cos(ant.heading), math.sin(ant.heading)
        home_body = np.asarray([-c * ant.position[0] - s * ant.position[1],
                               s * ant.position[0] - c * ant.position[1]])
        backward += int(gradient @ home_body < 0.)
        samples += 1
        action = controller.act(obs)
        forward += int(action.move)
        stopped += int(not action.move)
        env.step([action])
    return Probe(treatment, carrying, bool(ant.budget_returns), float(np.linalg.norm(ant.position)),
                 forward, stopped, backward, samples, float(np.mean(np.asarray(gradients) >= 0.)))


def run() -> None:
    for treatment in ("original", "no-return-deposit", "reverse-at-start", "ideal-source",
                      "nest-source", "bounded-no-initial", "bounded-initial", "bounded-initial-nest"):
        for carrying in (False, True):
            print(json.dumps(asdict(measure(treatment, carrying))), flush=True)


if __name__ == "__main__":
    run()
