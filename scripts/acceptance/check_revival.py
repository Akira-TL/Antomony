"""只核对复活记录的机械一致性，不估计或判定模型效果。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mathhackson.training.comparison.acceptance import TapeStore
from mathhackson.training.comparison.continuous import Frame
from mathhackson.training.comparison.online_actor import UpdateRecord


class Checks(BaseModel):
    worlds: int = 0
    frames: int = 0
    proposals: int = 0
    memory_snapshots: int = 0
    source_hashes: int = 0


def check(directory: Path) -> Checks:
    store = TapeStore(directory)
    plan = store.execution.plan
    if not plan.respawn:
        raise ValueError("此检查只适用于显式复活批次")
    result = Checks()
    for source in store.execution.sources:
        assert hashlib.sha256(Path(source.path).read_bytes()).hexdigest() == source.sha256, "源参数改变"
        result.source_hashes += 1
    for key, world in store.worlds.items():
        path, _ = store.world(*key)
        previous: Frame | None = None
        deliveries = pickups = active_steps = 0
        rewards = 0.
        with gzip.open(path / "trajectory.jsonl.gz", "rt") as stream:
            for tick, line in enumerate(stream, 1):
                frame = Frame.model_validate_json(line)
                assert frame.tick == tick and len(frame.ants) == plan.environment.ants
                for i, ant in enumerate(frame.ants):
                    prior = previous.ants[i] if previous else None
                    assert ant.cumulative_deaths == (prior.cumulative_deaths if prior else 0) + int(ant.active and ant.killed)
                    assert ant.cumulative_terminations == (prior.cumulative_terminations if prior else 0) + int(ant.active and ant.exhausted)
                    assert ant.revivals == (prior.revivals if prior else 0) + int(ant.respawned)
                    if ant.respawned:
                        assert prior is not None and prior.pending and prior.cumulative_terminations > 0
                        assert np.linalg.norm(ant.position) <= .65 + .18 + 1e-5, "复活未从巢内出发"
                    assert ant.writes >= (prior.writes if prior else 0)
                deliveries += sum(a.delivered for a in frame.ants)
                pickups += sum(a.picked_up for a in frame.ants)
                active_steps += sum(a.active for a in frame.ants)
                rewards += sum(a.reward for a in frame.ants)
                assert frame.food_stock is not None
                assert frame.food_stock + deliveries + sum(a.carrying for a in frame.ants) == plan.environment.stock, "食物不守恒"
                previous = frame
                result.frames += 1
        assert previous is not None and previous.tick == world.steps
        assert deliveries == world.deliveries and pickups == world.pickups and active_steps == world.active_individual_steps
        assert abs(rewards - world.reward) < 1e-5
        assert sum(a.cumulative_deaths for a in previous.ants) == world.deaths
        assert sum(a.cumulative_terminations for a in previous.ants) == world.exhausted
        assert sum(a.revivals for a in previous.ants) == world.revivals
        assert [a.writes for a in previous.ants] == world.writes
        records = [UpdateRecord.model_validate_json(line) for line in (path / "updates.jsonl").read_text().splitlines()]
        assert len(records) == world.decisions
        assert sum(r.changed for r in records) == sum(world.writes)
        assert sum(r.accepted for r in records) == world.accepted
        assert all(not r.changed or r.accepted for r in records)
        assert all(not r.accepted or not r.terminal or r.continuing_after_death for r in records)
        assert all(not r.continuing_after_death or r.terminal and r.tick < world.steps for r in records)
        result.proposals += len(records)
        if world.arm == "rules":
            assert not records and not list(path.glob("*.npz"))
        else:
            for individual in range(plan.environment.ants):
                selected = [r for r in records if r.individual == individual]
                offset = [0.] * 45
                for record in selected:
                    assert record.before == offset, "参数更新链断开"
                    offset = record.after
                for tick in world.snapshots:
                    with np.load(path / f"tick-{tick:04d}-ant-{individual:02d}.memory.npz", allow_pickle=False) as saved:
                        assert saved["history"].ndim == 2 and saved["history"].shape[1] == 8
                        assert saved["history"].shape[0] <= 16 and np.isfinite(saved["history"]).all()
                        assert saved["random_state"].dtype == np.uint8
                    result.memory_snapshots += 1
                    if world.arm != "mlp":
                        expected = [r.after for r in selected if r.tick <= tick]
                        with np.load(path / f"tick-{tick:04d}-ant-{individual:02d}.residual.npz", allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved["fast"] + saved["stable"], expected[-1] if expected else np.zeros(45))
        result.worlds += 1
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(check(parser.parse_args().directory).model_dump_json(indent=2))
