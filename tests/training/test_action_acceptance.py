import numpy as np
import pytest
from dataclasses import replace

from mathhackson.training.comparison.acceptance_scores import score_world, summarize
from mathhackson.training.comparison.action_acceptance import ProfileResult, compare


def result(profile, predictions, delivery=1.):
    labels = np.asarray([2., -1.])
    worlds = [score_world(f"initial-0-seed-{seed}", seed, 0., labels, np.asarray(p),
                         np.asarray([delivery, 0.]), np.zeros(2))
              for seed, p in zip((19811, 19812), predictions, strict=True)]
    return ProfileResult(profile=profile, fits=[], worlds=worlds,
                         decision=summarize(worlds, (19811, 19812), (0.,), minimum_passing=2))


def test_both_seeds_must_improve_over_all_accept_skip_and_old_input():
    old = result("parameters", [[1., 1.], [1., 1.]])
    good = result("actions", [[1., -1.], [1., -1.]])
    comparison = compare(old, good)
    assert comparison.development_continue and comparison.selected_differences == [.5, .5]
    weak = result("actions", [[1., -1.], [-1., -1.]])
    assert not compare(old, weak).development_continue
    tied = result("parameters", [[1., -1.], [1., -1.]])
    assert not compare(tied, good).development_continue
    no_delivery = result("actions", [[1., -1.], [1., -1.]], delivery=0.)
    assert not compare(old, no_delivery).development_continue


def test_mismatched_samples_or_profiles_are_rejected():
    old = result("parameters", [[1., 1.], [1., 1.]])
    new = result("actions", [[1., -1.], [1., -1.]])
    with pytest.raises(ValueError, match="顺序"):
        compare(new, old)
    new.worlds[0].count += 1
    with pytest.raises(ValueError, match="相同"):
        compare(old, new)


def test_both_profiles_finish_training_before_held_prediction(tmp_path, monkeypatch):
    from test_action_curriculum import action_plan
    from test_basic_update_learning import record
    from mathhackson.training.comparison import action_acceptance as module
    from mathhackson.training.comparison.acceptance_audit import Audit
    from mathhackson.training.foraging.action_features import ActionFeatureRecord
    from mathhackson.training.foraging.update_curriculum import CurriculumExecution, world_key
    plan = action_plan()
    execution = CurriculumExecution(commit="test", plan=plan, plan_sha256="test", sources=[], smoke=True)
    monkeypatch.setattr(module, "audit_inputs", lambda *args: (execution, Audit(files=0, snapshots=0, frames=0)))
    config = module.Config(input_directory=tmp_path / "input", manifest=tmp_path / "manifest",
                           protocol=tmp_path / "protocol", output_directory=tmp_path / "outputs")
    for seed in plan.train_seeds + plan.held_seeds:
        folder = config.input_directory / world_key(0, seed)
        folder.mkdir(parents=True)
        base = record(seed)
        row = base.model_copy(update={"proposal": replace(base.proposal, steps=4),
            "action_effects": ActionFeatureRecord(values=[.1] * 38)})
        (folder / "pairs.jsonl").write_text(row.model_dump_json() + "\n")
    read = module.read_partition
    visits = []

    def checked_read(directory, current, partition):
        if partition == "held":
            for profile in ("parameters", "actions"):
                for focal in range(plan.probe.environment.ants):
                    assert (config.output_directory / profile / f"ant-{focal:02d}" / "step-0004.npz").exists()
        visits.append((current.decision_profile, partition))
        return read(directory, current, partition)

    monkeypatch.setattr(module, "read_partition", checked_read)
    result = module.run(config)
    assert visits == [("parameters", "train"), ("actions", "train"), ("parameters", "held"), ("actions", "held")]
    assert result.parameters.fits[0].steps == result.actions.fits[0].steps == 4
    assert all(fit.steps == 0 for fit in result.actions.fits[1:])
    assert not result.comparison.development_continue
