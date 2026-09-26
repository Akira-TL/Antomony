from pathlib import Path

import numpy as np

from mathhackson.training.comparison.continuous import ContinuousPlan
from mathhackson.training.foraging.disturbance import DisturbedColony


ROOT = Path(__file__).resolve().parents[2]


def test_distant_layout_separates_food_and_hazards_from_nest():
    plan = ContinuousPlan.model_validate_json((ROOT / "docs/engineering/distant-acceptance.json").read_text())
    assert plan.environment.food_distance_min == 8.
    assert plan.environment.food_distance_span == 2.
    assert plan.seeds == (19201,)
    for seed in range(19201, 19301):
        for condition in plan.conditions:
            env = DisturbedColony(seed, plan.environment, condition.disturbance)
            assert 8. - 1e-6 <= np.linalg.norm(env.food) <= 10. + 1e-6
            # 垂直巡游不会比连线中心更靠近巢穴；接收器最远伸出0.9。
            clearance = float(np.linalg.norm(env.source_center)) - condition.disturbance.signal_radius
            assert clearance > plan.environment.nest_signal_radius + .9
            for i in range(plan.environment.ants):
                assert env.observation(i).receptors[:, 0].max() == 0.
            if condition.name == "moving-danger":
                for tick in range(condition.disturbance.motion_period_steps):
                    env.steps = tick
                    env.refresh_source()
                    assert np.linalg.norm(env.source_position) >= np.linalg.norm(env.source_center) - 1e-6


def test_distant_preview_changes_layout_not_models_or_learning_settings():
    near = ContinuousPlan.model_validate_json((ROOT / ".research/protocols/continuous-adaptation.json").read_text())
    far = ContinuousPlan.model_validate_json((ROOT / "docs/engineering/distant-acceptance.json").read_text())
    expected = near.model_dump()
    expected["seeds"] = far.seeds
    expected["environment"]["food_distance_min"] = 8.
    expected["environment"]["food_distance_span"] = 2.
    assert far.model_dump() == expected
