import math

import numpy as np
import pytest
import torch

from mathhackson.training.direction.adaptation import DirectionCorrection, local_direction_loss
from mathhackson.training.direction.environment import DirectionEnvironment
from mathhackson.training.direction.policy import DirectionMotor


def test_observed_direction_error_gives_correct_signed_local_gradient():
    turn = torch.tensor(.2, requires_grad=True)
    loss = local_direction_loss(turn, .1)
    assert float(loss.detach()) == pytest.approx(1. - math.cos(.1), abs=1e-7)
    loss.backward()
    # 理想小角度执行模型下，增大左转应减小剩余的正向误差。
    assert float(turn.grad) == pytest.approx(-math.sin(.1) * math.pi / 18, abs=1e-7)


def test_execution_change_does_not_change_observation_before_action():
    first = DirectionEnvironment(heading=0., desired=1.)
    second = DirectionEnvironment(heading=0., desired=1.)
    np.testing.assert_array_equal(first.observation(), second.observation())
    first.step(True, 0., execution_bias=.25)
    second.step(True, 0., execution_bias=-.25)
    assert first.heading == pytest.approx(math.pi / 72)
    assert second.heading == pytest.approx(-math.pi / 72)


def test_local_update_changes_only_the_direction_offset():
    motor = DirectionMotor(41).freeze()
    correction = DirectionCorrection(motor)
    before = [p.detach().clone() for p in motor.parameters()]
    optimizer = torch.optim.SGD(correction.plastic_parameters(), lr=.1)
    decision = correction.command(np.array([1., 0.], dtype=np.float32))
    local_direction_loss(decision.turn, .1).backward()
    optimizer.step()
    correction.project()
    assert correction.offset.item() != 0.
    assert all(a.equal(b) for a, b in zip(before, motor.parameters(), strict=True))
    assert all(not p.requires_grad for p in motor.parameters())


def test_zero_offset_preserves_loaded_motor_actions():
    motor = DirectionMotor(8).freeze()
    correction = DirectionCorrection(motor)
    direction = np.array([0., -1.], dtype=np.float32)
    original = motor.decide(direction)
    action = correction.command(direction)
    assert action.move == original.move
    assert float(action.turn.detach()) == original.turn
