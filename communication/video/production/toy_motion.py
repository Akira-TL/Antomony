"""Small, deterministic teaching model. NOT the MathHackson research engine.

All paths are computed from the same local tracking law and the same external
push. Only the update policy differs. The optional five-coefficient gate is a
separately calibrated toy surrogate, not the proposed research controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "communication/video/rendered/v2"
DT = 1.0 / 60.0


def trail(x: float | np.ndarray) -> float | np.ndarray:
    return 0.15 * np.sin(0.65 * x) + 0.10 * np.cos(1.2 * x)


def probability(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -20, 20))))


def features(weight: float, history: list[float]) -> np.ndarray:
    errors = np.asarray(history, dtype=float) - weight
    error = float(errors[-1])
    agreement = float(np.mean(np.sign(errors[-3:]) * np.sign(error))) if abs(error) > 1e-4 else -1.0
    return np.array([1.0, agreement, abs(error), float(np.std(history)), float(abs(error) > 0.015)])


def calibrate_gate() -> np.ndarray:
    """Offline explanatory calibration. Later targets never enter its input.

This predicts the value of a small update from causal features. Target labels
are built from observations arriving later, using two predictions saved at the
choice time; this is NOT a rollout oracle at display/inference time. Its local
logistic fit is not represented as a main-project experimental result.
"""
    count = 30000
    rng_stream = np.random.default_rng(42)
    stream = np.zeros(count + 4)
    level = 0.0
    for t in range(count + 4):
        if rng_stream.random() < 0.035:
            level = float(rng_stream.uniform(-1, 1))
        stream[t] = level
    flags = rng_stream.random(count + 4) < 0.18
    stream += np.where(flags, rng_stream.normal(0, 1.2, count + 4), rng_stream.normal(0, 0.03, count + 4))
    rng = np.random.default_rng(24)
    phi = np.zeros(5)
    history = [0.0] * 4
    weight = 0.0
    for t in range(count):
        observed = float(stream[t])
        history = (history + [observed])[-4:]
        available = features(weight, history)
        correction = 0.65 * (observed - weight)
        # Calibration targets only: outcome data are not available to features.
        later = stream[t + 1:t + 3]
        benefit = float(np.mean((later - weight) ** 2 - (later - weight - correction) ** 2)) - 1e-5
        phi += 0.015 * (float(benefit > 0) - probability(float(phi @ available))) * available
        if rng.random() < 0.4:
            weight += correction
    return phi


@lru_cache(maxsize=1)
def gate_parameters() -> np.ndarray:
    path = OUT / "toy-gate.npy"
    if not path.exists():
        raise RuntimeError("Run scripts/video/prepare-toy.sh first; calibration is separate from rendering.")
    return np.load(path, allow_pickle=False)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float
    phase: float
    weight: float
    push: float
    error: float
    gate: float


@dataclass
class Motion:
    samples: np.ndarray
    seconds: float
    policy: str
    scenario: str

    def pose(self, t: float) -> Pose:
        index = float(np.clip(t / DT, 0, len(self.samples) - 1))
        low = min(int(index), len(self.samples) - 2)
        mix = index - low
        values = (1.0 - mix) * self.samples[low] + mix * self.samples[low + 1]
        return Pose(*map(float, values[1:9]))

    @property
    def xy(self) -> np.ndarray:
        return self.samples[:, 1:3]


def external_push(t: float, duration: float, scenario: str) -> float:
    f = t / duration
    if scenario == "shift":
        return 0.85 if f > 0.28 else 0.0
    if scenario == "gust":
        return 1.25 if 0.27 < f < 0.292 else 0.0
    if scenario == "mixed":
        return (0.72 if f > 0.5 else 0.0) + (1.2 if 0.18 < f < 0.20 else 0.0) + (-1.3 if 0.33 < f < 0.35 else 0.0)
    if scenario == "quiet":
        return 0.0
    raise ValueError(f"Unknown scenario: {scenario}")


@lru_cache(maxsize=48)
def rollout(seconds: float, policy: str, scenario: str = "shift") -> Motion:
    if policy not in ("fixed", "always", "selective"):
        raise ValueError(policy)
    phi = gate_parameters()
    x, y, w, command, phase = -5.6, float(trail(-5.6)), 0.0, 0.0, 0.0
    history = [0.0] * 4
    n = round(seconds / DT)
    data = np.zeros((n + 1, 10))
    speed = 11.0 / seconds
    p = 0.0
    for i in range(n + 1):
        t = i * DT
        push = external_push(t, seconds, scenario)
        local_cross_track = float(trail(x)) - y
        wanted = float(np.clip(0.75 * local_cross_track - w, -1.3, 1.3))
        command += min(1.0, DT / 0.13) * (wanted - command)
        heading = math.atan2(command, speed)
        next_y = y + DT * (command + push)
        # Actual observed outcome after the action, not a hidden rule label.
        observed = (next_y - y) / DT - command
        error = observed - w
        apply = 0.0
        if i % 18 == 17:
            history = (history + [observed])[-4:]
            p = probability(float(phi @ features(w, history)))
            apply = float(policy == "always" or (policy == "selective" and p > 0.55))
            w += apply * 0.65 * error
        data[i] = (t, x, y, heading, phase, w, push, error, p, apply)
        x += speed * DT
        y = next_y
        phase += math.hypot(speed, command) * DT * 13.0
    return Motion(data, seconds, policy, scenario)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phi = calibrate_gate()
    np.save(OUT / "toy-gate.npy", phi)
    gate_parameters.cache_clear()
    lines: list[str] = []
    for scenario in ("shift", "gust", "mixed"):
        for policy in ("fixed", "always", "selective"):
            motion = rollout(20.0, policy, scenario)
            if not np.isfinite(motion.samples).all():
                raise AssertionError("Non-finite toy path")
            if not np.all(np.diff(motion.samples[:, 1]) > 0):
                raise AssertionError("Ant did not keep advancing")
            if policy == "fixed" and not np.all(motion.samples[:, 5] == 0):
                raise AssertionError("Frozen comparison changed weights")
            lines.append(f"{scenario}/{policy}: {len(motion.samples)} poses, continuous forward motion")
    a = rollout(20., "fixed", "mixed")
    b = rollout(20., "always", "mixed")
    c = rollout(20., "selective", "mixed")
    assert np.array_equal(a.samples[:, 6], b.samples[:, 6])
    assert np.array_equal(a.samples[:, 6], c.samples[:, 6])
    assert np.max(abs(a.xy - b.xy)) > 0.2
    (OUT / "toy-provenance.json").write_text(json.dumps({
        "status": "teaching simulation, not research evidence",
        "calibration_samples": 30000, "stream_seed": 42, "selection_seed": 24,
        "features": "constant; signed agreement; absolute error; history spread; error activity",
        "phi": phi.tolist(), "gate_threshold": 0.55,
        "comparison": "same local control law, push schedule, speed and actuation bounds",
        "gate_scope": "offline-calibrated toy surrogate; local online credit in film is a proposal illustration",
    }, indent=2))
    print("\n".join(lines))
    print("Matched perturbations, frozen weights, finite paths and differing motion checked.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
