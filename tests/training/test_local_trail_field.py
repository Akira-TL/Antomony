import numpy as np
import pytest

from mathhackson.colony.pheromone import Pheromones
from mathhackson.training.foraging.signals import LocalSignals


def test_legacy_profile_is_identical_to_original_field():
    old, actual = Pheromones(), LocalSignals().trails
    for step in range(40):
        point = np.asarray([step * .03, step * -.02], dtype=np.float32)
        for field in (old, actual):
            field.deposit(point, step % 2, .12)
            field.tick(.1)
    np.testing.assert_array_equal(actual.values, old.values)


def test_bounded_profile_refreshes_but_does_not_accumulate_stationary_peak():
    field = LocalSignals(trail_profile="bounded-local-v2").trails
    point = np.asarray([3., 1.], dtype=np.float32)
    field.deposit(point, 0, .12)
    first = field.values.copy()
    for _ in range(100):
        field.deposit(point, 0, .12)
    np.testing.assert_array_equal(field.values, first)
    assert 0. < field.values.max() <= .12
    field.tick(.1)
    faded = field.values.copy()
    field.deposit(point, 0, .12)
    assert (field.values >= faded).all() and (field.values > faded).any()
    assert not field.values[1].any()
    assert not field.values[:, :, :10].any()
    field.blocked.fill(True)
    field.values.fill(0.)
    field.deposit(point, 0, .12)
    assert not field.values.any()


def test_invalid_profile_rejected_and_disabled_field_stays_empty():
    with pytest.raises(ValueError):
        LocalSignals(trail_profile="invalid")
    field = LocalSignals(trail_profile="bounded-local-v2").trails
    field.enabled = False
    field.deposit(np.zeros(2, dtype=np.float32), 0, .12)
    assert not field.values.any()
