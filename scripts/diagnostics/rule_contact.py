"""固定既有失败世界，检查接触转向是否持续阻止前进。"""
from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.rules import LocalRuleController, local_gradient


@dataclass
class AntRecord:
    contacts: int = 0
    escape_stops: int = 0
    carried_steps: int = 0
    carried_moves: int = 0
    weak_return: int = 0
    pickups: int = 0
    deliveries: int = 0
    final_radius: float = 0.


def run(ignore_contact: bool) -> None:
    env = ColonyEnvironment(9603, ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.))
    controls = [LocalRuleController(9603 * 32 + i) for i in range(env.config.ants)]
    records = [AntRecord() for _ in controls]
    while not env.done:
        actions = []
        for i, (control, ant, record) in enumerate(zip(controls, env.ants, records, strict=True)):
            if ant.exhausted:
                actions.append(DirectionAction(False, 0., 0.))
                continue
            observation = env.observation(i)
            record.contacts += int(observation.contact)
            action = control.act(replace(observation, contact=False) if ignore_contact else observation)
            actions.append(action)
            record.escape_stops += int(not action.move and control.escape_left > 0)
            record.carried_steps += int(ant.carrying)
            record.carried_moves += int(ant.carrying and action.move)
            record.weak_return += int(ant.carrying and np.linalg.norm(local_gradient(observation.receptors[:, 1])) <= .0001)
        env.step(actions)
    for ant, record in zip(env.ants, records, strict=True):
        record.pickups, record.deliveries = ant.pickups, ant.deliveries
        record.final_radius = float(np.linalg.norm(ant.position))
    print(json.dumps({"ignore_contact": ignore_contact, "ants": [asdict(record) for record in records]}), flush=True)


if __name__ == "__main__":
    run(False)
    run(True)
