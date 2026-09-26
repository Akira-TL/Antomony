import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationConfig, NovelDirectionLearner, NovelSignalLearner
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner
from mathhackson.training.foraging.update_decision import IndividualUpdateController, UpdateDecision


def observation(novel=True):
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 1] = np.linspace(.1, 1., 9)
    if novel:
        receptors[:, 3] = np.linspace(.05, 2., 9)
    return LocalObservation(receptors, False, False, True, .1, (.8, 1.))


def learner():
    return TrustDirectionLearner(MemoryPolicy(FeedforwardPolicy(71)), DirectionMotor(41), 14,
                                 TrustConfig(window=4, feedback_mode="observed-window"))


def feedback(agent, rewards=(-2., -1., 0., 1.)):
    for reward in rewards:
        agent.act(observation())
        agent.feedback(reward)


def test_uncalibrated_memory_cannot_silently_use_critic_or_signal_adapter():
    policy, motor = MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41)
    with pytest.raises(ValueError, match="尚未校准"):
        TrustDirectionLearner(policy, motor, 0)
    with pytest.raises(ValueError, match="方向修正"):
        NovelSignalLearner(policy, motor, 0, AdaptationConfig(feedback_mode="observed-window"))


def test_observed_returns_match_hand_calculation_and_require_feedback():
    agent = NovelDirectionLearner(MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41), 0,
                                  AdaptationConfig(window=3, gamma=.5, feedback_mode="observed-window"))
    with pytest.raises(ValueError):
        agent.propose(observation())
    feedback(agent, (2., -1., 3.))
    torch.testing.assert_close(agent.feedback_advantage(observation()), torch.tensor([2.25, .5, 3.]), rtol=0., atol=0.)


def test_candidate_is_independent_of_untrained_value_and_next_observation():
    first, second = learner(), learner()
    with torch.no_grad():
        second.policy.base.value.weight.fill_(1000.)
        second.policy.base.value.bias.fill_(-1000.)
    feedback(first)
    feedback(second)
    assert first.propose(observation()) == second.propose(observation(False))
    assert first.diagnostics == second.diagnostics
    assert np.any(first.proposal.delta)


def test_zero_observed_reward_cannot_create_update_from_random_value():
    agent = learner()
    feedback(agent, (0., 0., 0., 0.))
    proposal = agent.propose(observation())
    assert not np.any(proposal.delta) and proposal.gradient_norm == 0.
    assert not agent.resolve(proposal, accept=(True,))
    assert agent.writes == 0


def test_accept_or_skip_changes_only_individual_direction_parameters():
    first, other = learner(), learner()
    before = [p.detach().clone() for p in first.policy.parameters()]
    motor_before = [p.detach().clone() for p in first.motor.parameters()]
    feedback(first)
    proposal = first.propose(observation())
    assert not first.resolve(proposal, accept=(False,))
    np.testing.assert_array_equal(first.weights(), np.zeros(45))
    feedback(first)
    proposal = first.propose(observation())
    assert first.resolve(proposal, accept=(True,))
    assert not other.weights().any()
    assert all(a.equal(b) for a, b in zip(before, first.policy.parameters(), strict=True))
    assert all(a.equal(b) for a, b in zip(motor_before, first.motor.parameters(), strict=True))
    assert all(not p.requires_grad for p in first.policy.parameters())
    assert first.offset.requires_grad


def test_offline_branch_preserves_parent_memory_and_random_state():
    agent = learner()
    feedback(agent)
    proposal = agent.propose(observation())
    before = agent.random.get_state().clone()
    history = [h.clone() for h in agent.history]
    branch = EvaluationActor(agent, accept=True)
    assert branch.changed
    branch.act(observation())
    assert agent.proposal is proposal and not agent.weights().any()
    assert agent.random.get_state().equal(before)
    assert all(a.equal(b) for a, b in zip(history, agent.history, strict=True))
    assert {p.data_ptr() for p in branch.agent.policy.parameters()}.isdisjoint(p.data_ptr() for p in agent.policy.parameters())
    assert {h.data_ptr() for h in branch.agent.history}.isdisjoint(h.data_ptr() for h in agent.history)


@pytest.mark.parametrize("prediction,accept", [(-1., False), (1., True)])
def test_existing_decision_controller_accepts_memory_candidates(prediction, accept):
    agent = learner()
    decision = UpdateDecision()
    with torch.no_grad():
        decision.value.bias.fill_(prediction)
    controller = IndividualUpdateController(decision)
    feedback(agent)
    agent.propose(observation())
    result = controller.resolve(agent, observation())
    assert result.eligible and result.accepted == accept and result.changed == accept
    assert agent.writes == int(accept)


@pytest.mark.parametrize("memory", [True, False])
def test_checkpoint_preserves_policy_and_feedback_configuration(tmp_path, memory):
    policy = MemoryPolicy(FeedforwardPolicy(71)) if memory else ForagingPolicy(71).with_budgets()
    agent = TrustDirectionLearner(policy, DirectionMotor(41), 14,
                                  TrustConfig(window=4, feedback_mode="observed-window"))
    feedback(agent)
    agent.resolve(agent.propose(observation()), accept=(True,))
    path = tmp_path / "agent.npz"
    agent.save(path)
    restored = TrustDirectionLearner.load(path, agent.motor, 14)
    assert restored.config == agent.config and type(restored.policy) is type(policy)
    assert restored.writes == agent.writes and restored.decisions == agent.decisions
    np.testing.assert_array_equal(restored.weights(), agent.weights())
    assert all(a.equal(b) for a, b in zip(restored.policy.parameters(), agent.policy.parameters(), strict=True))
    with pytest.raises(ValueError, match="配置不一致"):
        TrustDirectionLearner.load(path, agent.motor, 14, TrustConfig())


def test_old_snapshot_without_saved_config_still_loads(tmp_path):
    agent = TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14)
    path = tmp_path / "legacy.npz"
    agent.save(path)
    residual = path.with_suffix(".residual.npz")
    with np.load(residual, allow_pickle=False) as data:
        old = {key: data[key].copy() for key in data.files if key != "config_json"}
    np.savez(residual, **old)
    restored = TrustDirectionLearner.load(path, agent.motor, 14)
    assert restored.config == TrustConfig()


def test_probe_records_explicit_policy_identity_without_changing_old_plans():
    from pathlib import Path
    from mathhackson.training.foraging.candidate_probe import ProbePlan

    old = ProbePlan.model_validate_json(Path(".research/protocols/trust-candidate-value.json").read_text())
    assert old.policy_kind == "recurrent" and old.policy_episode == 8
    assert old.policy_path(3).name == "episode-0008-ant-03.npz"
    fields = old.model_dump() | {"policy_kind": "mlp-memory", "policy_episode": 0}
    with pytest.raises(ValueError, match="已发生窗口反馈"):
        ProbePlan.model_validate(fields)
    fields["adaptation"]["feedback_mode"] = "observed-window"
    plan = ProbePlan.model_validate(fields)
    assert ProbePlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.policy_path(3).name == "episode-0000-ant-03.npz"


def test_real_probe_loads_memory_saves_snapshots_and_keeps_parent_frozen(tmp_path):
    import gzip
    from pathlib import Path
    from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
    from mathhackson.training.foraging.candidate_probe import CandidateRecord, ProbePlan, collect

    sources, output = tmp_path / "sources", tmp_path / "output"
    sources.mkdir()
    output.mkdir()
    motor = DirectionMotor(41)
    save_motor(sources / "motor.npz", motor, MotorSnapshot(seed=41, updates=0))
    for i in range(2):
        MemoryPolicy(FeedforwardPolicy(i)).save(sources / f"episode-0000-ant-{i:02d}.npz", update=0, phase="frozen")
    original = {path: path.read_bytes() for path in sources.iterdir()}
    fields = ProbePlan.model_validate_json(Path(".research/protocols/trust-candidate-value.json").read_text()).model_dump()
    fields.update(policy_kind="mlp-memory", policy_episode=0, policy_directory=str(sources),
                  motor_path=str(sources / "motor.npz"), pairs_per_world=1, branch_horizon=2,
                  include_zero_candidates=True)
    fields["environment"].update(ants=2, horizon=8)
    fields["adaptation"].update(window=4, feedback_mode="observed-window")
    result = collect(ProbePlan.model_validate(fields), 9599, "benign", output)
    assert result.steps == 8 and result.pairs == 1
    pairs = [CandidateRecord.model_validate_json(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 1 and pairs[0].tick == 4
    with gzip.open(output / "parent.jsonl.gz", "rt") as stream:
        assert len(stream.readlines()) == 8
    for i in range(2):
        restored = TrustDirectionLearner.load(output / f"tick-0008-ant-{i:02d}.npz", motor, 0)
        assert isinstance(restored.policy, MemoryPolicy) and restored.writes == 0
        assert not restored.weights().any()
    assert all(path.read_bytes() == content for path, content in original.items())

    from mathhackson.training.foraging.update_curriculum import CurriculumExecution, CurriculumPlan, collect_curriculum
    fields.update(seeds=(9599, 9600), conditions=("benign",))
    fields["disturbance"]["injury_per_step"] = 0.
    curriculum = CurriculumPlan(train_seeds=(9599,), held_seeds=(9600,), initial_norms=(0.,),
                                probe=ProbePlan.model_validate(fields))
    curriculum_output = tmp_path / "curriculum"
    collect_curriculum(curriculum, curriculum_output, plan_sha256="engineering-test", smoke=True)
    execution = CurriculumExecution.model_validate_json((curriculum_output / "execution.json").read_text())
    assert execution.smoke and execution.plan.probe.policy_kind == "mlp-memory"
    assert {Path(item.path).name for item in execution.sources} == {
        "episode-0000-ant-00.npz", "episode-0000-ant-01.npz", "motor.npz"}
    assert all(path.read_bytes() == content for path, content in original.items())
