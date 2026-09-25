from pathlib import Path

import numpy as np
import torch

from mathhackson.training.roundtrip_policy import RoundTripPolicy

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
