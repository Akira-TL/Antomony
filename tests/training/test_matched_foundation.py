import gzip

import pytest
from pydantic import ValidationError

from mathhackson.training.comparison.matched_foundation import MatchedPlan, Execution, run
from mathhackson.training.comparison.run import WorldRecord
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.mlp import FeedforwardPolicy


@pytest.mark.parametrize("changes", [{"model_seeds": (81,)}, {"model_seeds": (81,) * 8},
    {"evaluation_seeds": ()}, {"evaluation_seeds": (1, 1)}, {"signal_updates": 0}, {"checkpoint_every": 0}])
def test_plan_rejects_incomplete_units_and_budget(changes):
    with pytest.raises(ValidationError):
        MatchedPlan(**changes)


def test_short_training_evaluation_is_complete_frozen_and_exclusive(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    for seed in (81, 82):
        FeedforwardPolicy(seed).save(source / f"seed-{seed}-signal-002400.npz", update=2400, phase="signal")
    motor_path = source / "motor.npz"
    save_motor(motor_path, DirectionMotor(41), MotorSnapshot(updates=0, seed=41))
    plan = MatchedPlan(motor=str(motor_path), original_directory=str(source))
    output = tmp_path / "result"
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    run(plan, output, smoke=True)
    execution = Execution.model_validate_json((output / "execution.json").read_text())
    assert execution.smoke and execution.completed_at and execution.evaluation_updates == 0
    assert execution.training_updates_per_individual == 2
    assert [model.policy_parameters for model in execution.models] == [1650, 1365]
    rows = [WorldRecord.model_validate_json(line) for line in (output / "worlds.jsonl").read_text().splitlines()]
    assert {(row.model, row.task) for row in rows} == {
        (name, task) for name in ("mlp-matched", "mlp-original", "rules") for task in ("food", "empty")}
    assert len(rows) == 6 and all(not any(row.updates) and row.steps == 8 for row in rows)
    for row in rows:
        with gzip.open(output / f"{row.model}-{row.task}-{row.seed}-{row.sampled}.jsonl.gz", "rt") as stream:
            assert len(stream.readlines()) == row.steps
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    assert len(list(output.glob("seed-*-signal-*.npz"))) == 6
    with pytest.raises(FileExistsError):
        run(plan, output, smoke=True)
