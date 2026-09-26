import pytest

from mathhackson.training.comparison.continuous import AntFrame, Frame
from mathhackson.training.comparison.survival_timing import describe


def frame(tick, *, injury=0., writes=0, killed=False, respawned=False, x=.5):
    return Frame(tick=tick, source_position=[1., 0.], source_active=True, ants=[AntFrame(
        observation=[0.] * 78, active=True, move=True, turn=0., position=[x, 0.], heading=0.,
        carrying=False, exploration_left=100, reserve_left=100, picked_up=False, delivered=False,
        budget_return=False, exhausted=killed, killed=killed, injury=injury, reward=0., writes=writes,
        respawned=respawned)])


def describe_one(trace, changes):
    return describe(trace, changes, [[.5, 0.]], variant='test', seed=1, arm='always')


def test_death_tick_write_cannot_have_changed_the_action_that_killed():
    result = describe_one([frame(1, injury=.1), frame(2, injury=1., writes=1, killed=True)], {(2, 0)})
    assert result.first_write_delays == [1]
    assert result.injured_deaths == result.deaths_without_prior_injury_write == 1


def test_live_write_and_inherited_death_are_distinguished_without_teleport_distance():
    trace = [frame(1, injury=.1, writes=1, x=4.), frame(2, injury=1., writes=1, killed=True, x=4.),
             frame(3, injury=.1, writes=1, respawned=True), frame(4, injury=1., writes=2, killed=True)]
    result = describe_one(trace, {(1, 0), (4, 0)})
    assert result.injured_deaths == 2 and result.deaths_without_prior_injury_write == 1
    assert result.deaths_with_inherited_writes == 1
    assert result.periods[0].distance == 3.5
    assert result.periods[0].omitted_respawn_moves == 1
    assert result.injured_lives[1].start == 3 and result.injured_lives[1].inherited_writes == 1


def test_horizon_censoring_is_not_death_or_missing_write_failure():
    result = describe_one([frame(1, injury=.1)], set())
    assert result.injured_lives[0].end is None
    assert result.injured_deaths == result.deaths_without_prior_injury_write == 0
    assert result.first_write_delays == []
    with pytest.raises(ValueError, match='写入'):
        describe_one([frame(1, writes=1)], set())
