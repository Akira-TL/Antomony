"""Fast reversible parameter residual used by online V2V learning.

The container follows the frozen semantics:

    theta_effective = theta_stable + fast_residual

Recent entries duplicate only the latest delta identities for precise rollback; they are
already included in ``fast_residual`` and are never added a second time in forward use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RollbackRequest:
    """Rollback one currently retained recent delta identity."""

    index_from_oldest: int
    fraction: float


class FastResidualParameter:
    """Stable parameter + unconsolidated fast residual + bounded recent identities."""

    def __init__(self, stable: npt.ArrayLike, *, recent_capacity: int) -> None:
        if recent_capacity < 0:
            raise ValueError("recent_capacity must be non-negative")
        self.stable = np.asarray(stable, dtype=np.float32).copy()
        self.fast = np.zeros_like(self.stable)
        self.recent_capacity = recent_capacity
        self.recent: list[npt.NDArray[np.float32]] = []

    @property
    def effective(self) -> npt.NDArray[np.float32]:
        return self.stable + self.fast

    def _validated_delta(self, delta: npt.ArrayLike) -> npt.NDArray[np.float32]:
        array = np.asarray(delta, dtype=np.float32)
        if array.shape != self.stable.shape:
            raise ValueError(f"expected delta shape {self.stable.shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("delta must contain only finite values")
        return array

    def add_delta(self, delta: npt.ArrayLike) -> None:
        """Add a new fast modification and retain its identity if capacity allows."""

        array = self._validated_delta(delta)
        self.fast += array
        if self.recent_capacity == 0 or np.count_nonzero(array) == 0:
            return
        self.recent.append(array.copy())
        if len(self.recent) > self.recent_capacity:
            # Losing the oldest identity must not change the effective parameter.
            self.recent.pop(0)

    def rollback(self, request: RollbackRequest) -> None:
        """Partially or fully remove one retained delta from the fast residual."""

        if not 0.0 <= request.fraction <= 1.0:
            raise ValueError("rollback fraction must be in [0, 1]")
        if not 0 <= request.index_from_oldest < len(self.recent):
            raise IndexError("recent rollback index out of range")
        identity = self.recent[request.index_from_oldest]
        amount = np.float32(request.fraction) * identity
        self.fast -= amount
        identity *= np.float32(1.0 - request.fraction)
        if np.count_nonzero(identity) == 0:
            self.recent.pop(request.index_from_oldest)

    def consolidate(self, fraction: float) -> None:
        """Move a fraction of the existing fast residual into stable storage.

        This operation is function-preserving: ``stable + fast`` remains unchanged.
        Every retained recent identity is scaled by the same remaining fraction so future
        rollback still refers to the unconsolidated part that physically remains in fast.
        """

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("consolidation fraction must be in [0, 1]")
        consolidate_fraction = np.float32(fraction)
        remaining_fraction = np.float32(1.0 - fraction)
        old_fast = self.fast.copy()
        self.stable += consolidate_fraction * old_fast
        self.fast[...] = remaining_fraction * old_fast
        for identity in self.recent:
            identity *= remaining_fraction
        self.recent = [identity for identity in self.recent if np.count_nonzero(identity) != 0]

    def finalize_coordinates(self, mask: npt.ArrayLike) -> None:
        """Fully consolidate only selected parameter coordinates.

        This is a physical execution primitive, not a policy for choosing what should become
        stable.  Selected coordinates move their entire current fast residual into ``stable``
        in one operation and become fast-zero.  The same coordinate components are removed
        from retained recent identities, while every unselected coordinate and its rollback
        identity remain untouched.

        The operation is function-preserving at execution time and is suitable for an exact
        mature-Edge withdrawal once a model-controlled long-timescale process has decided to
        make that withdrawal permanent.
        """

        selected = np.asarray(mask, dtype=np.bool_)
        if selected.shape != self.stable.shape:
            raise ValueError(
                f"expected finalization mask shape {self.stable.shape}, got {selected.shape}"
            )
        if not np.any(selected):
            return
        self.stable[selected] += self.fast[selected]
        self.fast[selected] = np.float32(0.0)
        for identity in self.recent:
            identity[selected] = np.float32(0.0)
        self.recent = [identity for identity in self.recent if np.count_nonzero(identity) != 0]

    def commit(
        self,
        new_delta: npt.ArrayLike,
        *,
        rollback: RollbackRequest | None = None,
        consolidation_fraction: float = 0.0,
    ) -> None:
        """Apply one event in the required causal order.

        Order is strictly:

        1. rollback an older retained modification, if requested;
        2. consolidate the remaining *old* fast residual;
        3. add the current new delta.

        The current delta therefore cannot consolidate in the event that creates it.
        """

        delta = self._validated_delta(new_delta)
        if rollback is not None:
            self.rollback(rollback)
        self.consolidate(consolidation_fraction)
        self.add_delta(delta)
