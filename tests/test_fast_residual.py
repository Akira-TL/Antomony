from __future__ import annotations

import numpy as np

from mathhackson.fast_residual import FastResidualParameter, RollbackRequest


def test_new_delta_enters_fast_and_recent_exactly_once() -> None:
    parameter = FastResidualParameter(np.asarray([1.0, 2.0], dtype=np.float32), recent_capacity=2)
    delta = np.asarray([0.25, -0.5], dtype=np.float32)

    parameter.add_delta(delta)

    np.testing.assert_array_equal(parameter.fast, delta)
    np.testing.assert_array_equal(parameter.effective, np.asarray([1.25, 1.5], dtype=np.float32))
    assert len(parameter.recent) == 1
    np.testing.assert_array_equal(parameter.recent[0], delta)


def test_recent_overflow_discards_identity_without_changing_effective_parameter() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=2)
    parameter.add_delta(np.asarray([0.1], dtype=np.float32))
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    before = parameter.effective.copy()

    parameter.add_delta(np.asarray([0.3], dtype=np.float32))

    np.testing.assert_allclose(parameter.effective, before + 0.3, rtol=0.0, atol=2e-7)
    assert len(parameter.recent) == 2
    np.testing.assert_allclose(parameter.recent[0], [0.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[1], [0.3], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.fast, [0.6], rtol=0.0, atol=1e-7)


def test_partial_rollback_removes_only_selected_retained_identity() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))
    parameter.add_delta(np.asarray([-0.1], dtype=np.float32))

    parameter.rollback(RollbackRequest(index_from_oldest=1, fraction=0.25))

    np.testing.assert_allclose(parameter.fast, [0.4], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[0], [0.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[1], [0.3], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[2], [-0.1], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.effective, [1.4], rtol=0.0, atol=1e-7)


def test_full_rollback_removes_exhausted_recent_identity() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))

    parameter.rollback(RollbackRequest(index_from_oldest=0, fraction=1.0))

    np.testing.assert_allclose(parameter.fast, [0.4], rtol=0.0, atol=1e-7)
    assert len(parameter.recent) == 1
    np.testing.assert_allclose(parameter.recent[0], [0.4], rtol=0.0, atol=1e-7)


def test_full_consolidation_clears_exhausted_rollback_identities() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))
    before = parameter.effective.copy()

    parameter.consolidate(1.0)

    np.testing.assert_allclose(parameter.effective, before, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.fast, [0.0], rtol=0.0, atol=0.0)
    assert parameter.recent == []


def test_zero_delta_does_not_create_rollback_identity() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)

    parameter.add_delta(np.asarray([0.0], dtype=np.float32))

    np.testing.assert_allclose(parameter.fast, [0.0], rtol=0.0, atol=0.0)
    assert parameter.recent == []


def test_consolidation_preserves_effective_function_and_scales_recent_identities() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=2)
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))
    before = parameter.effective.copy()

    parameter.consolidate(0.25)

    np.testing.assert_allclose(parameter.effective, before, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.stable, [1.15], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.fast, [0.45], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[0], [0.15], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[1], [0.3], rtol=0.0, atol=1e-7)


def test_finalize_coordinates_preserves_unselected_fast_and_rollback_identity() -> None:
    parameter = FastResidualParameter(
        np.asarray([0.6, 0.4], dtype=np.float32),
        recent_capacity=4,
    )
    parameter.add_delta(np.asarray([-0.6, 0.0], dtype=np.float32))
    parameter.add_delta(np.asarray([0.0, 0.25], dtype=np.float32))
    before = parameter.effective.copy()

    parameter.finalize_coordinates(np.asarray([True, False]))

    np.testing.assert_array_equal(parameter.effective, before)
    np.testing.assert_array_equal(parameter.stable, np.asarray([0.0, 0.4], dtype=np.float32))
    np.testing.assert_array_equal(parameter.fast, np.asarray([0.0, 0.25], dtype=np.float32))
    assert len(parameter.recent) == 1
    np.testing.assert_array_equal(parameter.recent[0], np.asarray([0.0, 0.25], dtype=np.float32))

    parameter.rollback(RollbackRequest(index_from_oldest=0, fraction=1.0))
    np.testing.assert_array_equal(parameter.effective, np.asarray([0.0, 0.4], dtype=np.float32))
    assert parameter.recent == []


def test_finalize_coordinates_rejects_wrong_mask_shape() -> None:
    parameter = FastResidualParameter(np.asarray([1.0, 2.0], dtype=np.float32), recent_capacity=1)

    with np.testing.assert_raises(ValueError):
        parameter.finalize_coordinates(np.asarray([True]))


def test_commit_consolidates_old_fast_before_adding_current_new_delta() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))

    parameter.commit(
        np.asarray([0.2], dtype=np.float32),
        consolidation_fraction=0.5,
    )

    # Old 0.4 is split 0.2 stable + 0.2 fast; new 0.2 stays fully fast.
    np.testing.assert_allclose(parameter.stable, [1.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.fast, [0.4], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[0], [0.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[1], [0.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.effective, [1.6], rtol=0.0, atol=1e-7)


def test_commit_order_is_rollback_then_consolidation_then_new_delta() -> None:
    parameter = FastResidualParameter(np.asarray([1.0], dtype=np.float32), recent_capacity=3)
    parameter.add_delta(np.asarray([0.2], dtype=np.float32))
    parameter.add_delta(np.asarray([0.4], dtype=np.float32))

    parameter.commit(
        np.asarray([0.3], dtype=np.float32),
        rollback=RollbackRequest(index_from_oldest=1, fraction=0.5),
        consolidation_fraction=0.5,
    )

    # Before event: F=0.6. Rollback half of second identity => F=0.4.
    # Consolidate half => stable +0.2, old fast 0.2, retained identities 0.1/0.1.
    # New delta 0.3 then enters fast and Recent unscaled.
    np.testing.assert_allclose(parameter.stable, [1.2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.fast, [0.5], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[0], [0.1], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[1], [0.1], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.recent[2], [0.3], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(parameter.effective, [1.7], rtol=0.0, atol=1e-7)
