import numpy as np
import pytest
from pydantic import ValidationError

from mathhackson.interactive.protocol import Control, Edit
from mathhackson.training.foraging.rules import LocalRuleController


def test_paired_clock_independent_worlds_and_real_parameter_views(live):
    assert live.paused and live.tick == 0
    assert all(isinstance(a, LocalRuleController) for a in live.groups[2].actors)
    first, other = live.groups[0].actors
    assert first.agent.offset.data_ptr() != other.agent.offset.data_ptr()
    assert not np.shares_memory(live.groups[0].world.signals.trails.values, live.groups[1].world.signals.trails.values)
    live.control(Control(kind="step"))
    assert [g.world.steps for g in live.groups] == [1, 1, 1]
    assert live.paused
    assert live.parameters("adaptive", 0).modules[0].points[-1].values == first.agent.weights().tolist()
    assert all(m.frozen for m in live.parameters("mlp", 1).modules)
    assert live.parameters("rules", 1).modules == []
    with pytest.raises(ValueError):
        live.parameters("rules", 2)


def test_advance_returns_the_exact_recorded_frame(live):
    frame = live.advance()
    assert frame == live.records.replay(1)
    assert frame.tick == 1


def test_edit_validates_all_three_before_changing_any_world(live):
    edit = Edit(kind="wall", x=-4., y=3.)
    live.groups[1].world.ants[0].position[:] = [-4., 3.]
    checked = live.preview(edit)
    assert checked.valid and "0 / 1 / 0" in checked.message
    assert all(len(g.world.walls) == 3 for g in live.groups)
    assert live.edit(edit).valid
    assert [len(g.world.walls) for g in live.groups] == [4, 4, 4]
    assert not live.groups[1].world.walls[-1].overlaps(live.groups[1].world.ants[0].position, .18)
    assert live.tick == 0
    assert live.edit(Edit(kind="food", x=-8., y=-6., stock=7)).valid
    assert [g.world.stock for g in live.groups] == [55, 55, 55]
    live.groups[0].world.foods[-1].stock = 3
    assert live.groups[1].world.foods[-1].stock == 7
    assert len((live.records.directory / "interventions.jsonl").read_text().splitlines()) == 2


def test_continuous_run_records_real_fields_memories_and_control_changes(live):
    live.control(Control(kind="learning", enabled=False))
    live.control(Control(kind="pause", enabled=False))
    with pytest.raises(ValueError, match="暂停"):
        live.control(Control(kind="step"))
    for _ in range(16):
        live.advance()
    assert not live.done and not live.paused
    assert all(a.agent.writes == 0 for a in live.groups[0].actors)
    assert live.groups[0].decisions == 8
    assert live.records.replay(4).tick == 4
    assert live.records.replay(16).groups == live.frame().groups
    assert live.records.replay(0).groups[0].field != live.records.replay(16).groups[0].field
    assert (live.records.directory / "adaptive/tick-0008-ant-00.memory.npz").is_file()
    assert (live.records.directory / "adaptive/tick-0016.field.npz").is_file()
    with pytest.raises(ValueError):
        live.records.replay(17)
    assert live.edit(Edit(kind="food", x=4., y=4.)).valid
    assert not list((live.records.directory / "rules").glob("*-ant-*.npz"))


def test_running_single_step_and_invalid_period_are_rejected(live):
    with pytest.raises(ValidationError):
        Edit(kind="trap", period=2)
    assert live.edit(Edit(kind="trap", x=5., y=4., period=8)).valid
    assert [g.world.traps for g in live.groups].count(live.groups[0].world.traps) == 3
    live.control(Control(kind="speed", rate=8))
    live.control(Control(kind="checkpoint"))
    assert live.rate == 8 and live.checkpoint_tick == 0
