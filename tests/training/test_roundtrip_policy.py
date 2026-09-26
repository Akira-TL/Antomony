from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.roundtrip_policy import ROUNDTRIP_NAMES, RoundTripPolicy

FOUNDATION = Path(__file__).resolve().parents[2] / "checkpoints/recurrent/foundation-episode-002570.npz"


def test_foundation_weights_are_loaded_and_motor_stays_frozen():
    model = RoundTripPolicy(7, FOUNDATION)
    with np.load(FOUNDATION, allow_pickle=False) as archive:
        assert torch.equal(model.motor[:, :16], torch.from_numpy(archive["motor"]))
        assert torch.equal(model.input_weights[:, :16], torch.from_numpy(archive["input_weights"]))
    assert model.motor.shape == (2, 17)
    assert model.input_weights.shape == (8, 17)
    assert not torch.count_nonzero(model.trainable_masks()[0])
    assert torch.count_nonzero(model.trainable_masks()[1]) == 56
    assert torch.count_nonzero(model.trainable_masks()[7]) == 8
    assert not torch.count_nonzero(model.trainable_masks()[8])


def test_release_actions_receive_full_episode_training_signal():
    model = RoundTripPolicy(8, FOUNDATION)
    observation = np.zeros(17, np.float32)
    observation[15] = 1.
    action = model.decide(observation)
    write = model.observe_result(observation, terminal=True)
    motor = model.motor.detach().clone()
    initial = model.release_weights.detach().clone()
    model.finish([action], [write], [2.], 0., True)
    assert torch.equal(model.motor, motor)
    assert not torch.equal(model.release_weights, initial)
    assert model.release_fast.shape == (2,)


def test_release_initialization_prefers_opposite_trail_on_each_leg():
    model = RoundTripPolicy(8, FOUNDATION)
    model.phase = "autonomous"
    observation = np.zeros(17, np.float32)
    observation[15] = 1.
    outward = model.decide(observation)
    observation[16] = 1.
    returning = model.decide(observation)
    assert outward.release_home and not outward.release_food
    assert returning.release_food and not returning.release_home
    assert outward.release_home_probability > outward.release_food_probability
    assert returning.release_food_probability > returning.release_home_probability


def test_feedback_write_can_adjust_release_without_changing_slow_motor():
    model = RoundTripPolicy(9, FOUNDATION)
    model.phase = "adaptive"
    model.write_mode = "always"
    with torch.no_grad():
        model.release_write_weights.zero_()
        model.release_write_weights[:, -1] = 1.
    observation = np.zeros(17, np.float32)
    observation[15] = 1.
    motor = model.motor.detach().clone()
    write = model.observe_result(observation)
    assert write.requested and write.wrote
    assert torch.all(model.release_fast > 0)
    assert torch.equal(model.motor, motor)
    model.reset_state()
    assert torch.count_nonzero(model.release_fast) == 0


def test_autonomous_write_also_adjusts_release_without_building_gradients():
    model = RoundTripPolicy(9, FOUNDATION)
    model.phase = "autonomous"
    model.write_mode = "always"
    with torch.no_grad():
        model.release_write_weights.zero_()
        model.release_write_weights[:, -1] = 1.
    observation = np.zeros(17, np.float32)
    observation[15] = 1.
    before = model.decide(observation).release_home_probability
    write = model.observe_result(observation)
    after = model.decide(observation).release_home_probability
    assert write.requested and write.wrote
    assert torch.all(model.release_fast > 0)
    assert model.release_fast.grad_fn is None
    assert after > before


def test_roundtrip_action_head_can_turn_nearly_ten_degrees_per_step():
    model = RoundTripPolicy(11, FOUNDATION)
    model.phase = "autonomous"
    with torch.no_grad():
        model.action_weights[1].fill_(1.)
    model.hidden = torch.ones(8)
    observation = np.zeros(17, np.float32)
    observation[15] = 1.
    assert model.decide(observation).turn > .9


def test_saved_roundtrip_parameters_can_be_loaded_for_replay(tmp_path):
    original = RoundTripPolicy(12, FOUNDATION)
    with torch.no_grad():
        original.action_weights[1, 0] = .75
    path = tmp_path / "scent.npz"
    np.savez_compressed(
        path, model_version="roundtrip-v1",
        **{name: value.detach().numpy() for name, value in
           zip(ROUNDTRIP_NAMES, original.parameters, strict=True)})
    restored = RoundTripPolicy(13, FOUNDATION)
    restored.load_roundtrip_checkpoint(path)
    assert all(torch.equal(a, b) for a, b in zip(original.parameters, restored.parameters, strict=True))
    assert torch.count_nonzero(restored.fast) == 0


def test_incomplete_roundtrip_checkpoint_is_rejected_without_changing_model(tmp_path):
    model = RoundTripPolicy(14, FOUNDATION)
    before = [value.detach().clone() for value in model.parameters]
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, model_version="roundtrip-v1", motor=before[0].numpy())
    with pytest.raises(ValueError, match="缺少参数"):
        model.load_roundtrip_checkpoint(path)
    assert all(torch.equal(old, current) for old, current in zip(before, model.parameters, strict=True))
