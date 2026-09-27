import numpy as np
import pytest
from pydantic import ValidationError

from mathhackson.interactive.protocol import SessionConfig
from mathhackson.interactive.session import make_plan
from mathhackson.interactive.world import INTERACTIVE_HALF, EditableColony


def test_live_defaults_open_64_independent_body_slots_in_the_larger_nest():
    config = SessionConfig()
    assert config.ants == 64 and config.stock == 768
    plan = make_plan(config)
    world = EditableColony(config.seed, plan.environment, rich_scene=True)

    assert len(world.ants) == 64
    assert not world.pending.any() and not world.waiting
    assert tuple(world.signals.trails.half) == INTERACTIVE_HALF
    for index, ant in enumerate(world.ants):
        assert np.linalg.norm(ant.position) < plan.environment.home_radius
        assert all(np.linalg.norm(ant.position - other.position) >= .36
                   for other in world.ants[:index])


def test_live_ant_limit_matches_the_renderer_capacity():
    assert SessionConfig(ants=64).ants == 64
    with pytest.raises(ValidationError):
        SessionConfig(ants=65)


def test_expanded_pheromone_grid_round_trips_new_scene_edges():
    plan = make_plan(SessionConfig(ants=1))
    world = EditableColony(1, plan.environment, rich_scene=True)
    point = np.asarray([16.7, 11.7], dtype=np.float32)
    world.signals.trails.deposit(point, 0, 3.)
    assert world.signals.trails.sample(point, 0) > 0.
