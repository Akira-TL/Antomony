"""既有轨迹的受伤、写入与行动时序；描述关联，不模拟反事实。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import Frame
from .online_actor import UpdateRecord
from .return_reward_audit import frames


class InjuredLife(BaseModel):
    individual: int
    start: int
    first_injury: int
    end: int | None = None
    killed: bool = False
    exhausted: bool = False
    first_write: int | None = None
    inherited_writes: int = 0
    writes_before_injury: int = 0


class Period(BaseModel):
    start: int
    end: int
    active_steps: int = 0
    move_steps: int = 0
    near_home_steps: int = 0
    novel_steps: int = 0
    injury_steps: int = 0
    deaths: int = 0
    exhaustion: int = 0
    writes: int = 0
    deliveries: int = 0
    distance: float = 0.
    omitted_respawn_moves: int = 0


class Timing(BaseModel):
    variant: str
    seed: int
    arm: str
    periods: list[Period]
    injured_lives: list[InjuredLife]
    injured_deaths: int
    deaths_without_prior_injury_write: int
    deaths_with_inherited_writes: int
    first_write_delays: list[int]


def describe(trace: Iterable[Frame], changes: set[tuple[int, int]], initial_positions: list[list[float]],
             *, variant: str, seed: int, arm: str) -> Timing:
    count = len(initial_positions)
    lives: list[InjuredLife | None] = [None] * count
    started = [1] * count
    inherited = [0] * count
    collected: list[InjuredLife] = []
    periods: list[Period] = []
    previous: Frame | None = None
    for tick, frame in enumerate(trace, 1):
        if frame.tick != tick or len(frame.ants) != count:
            raise ValueError('时序或个体数不一致')
        if (tick - 1) % 512 == 0:
            periods.append(Period(start=tick, end=tick))
        period = periods[-1]
        period.end = tick
        for i, ant in enumerate(frame.ants):
            prior = previous.ants[i] if previous else None
            before_writes = prior.writes if prior else 0
            changed = (tick, i) in changes
            if ant.writes - before_writes != int(changed):
                raise ValueError('写入事件与累计计数不符')
            if ant.respawned:
                if lives[i] is not None:
                    raise ValueError('复活前生命未结束')
                started[i], inherited[i] = tick, before_writes
            if not ant.active:
                continue
            injury = ant.injury - (prior.injury if prior and not ant.respawned else 0.)
            if injury < -1e-6:
                raise ValueError('非复活的伤害倒退')
            if injury > 1e-6 and lives[i] is None:
                lives[i] = InjuredLife(individual=i, start=started[i], first_injury=tick,
                    inherited_writes=inherited[i], writes_before_injury=before_writes)
            life = lives[i]
            if life is not None and changed and life.first_write is None:
                life.first_write = tick
            if ant.exhausted and life is not None:
                life.end, life.killed, life.exhausted = tick, ant.killed, not ant.killed
                collected.append(life)
                lives[i] = None
            period.active_steps += 1
            period.move_steps += ant.move
            period.near_home_steps += math.hypot(*ant.position) < .65
            period.novel_steps += any(ant.observation[j] > 0. for j in range(72) if j % 8 >= 3)
            period.injury_steps += injury > 1e-6
            period.deaths += ant.killed
            period.exhaustion += ant.exhausted and not ant.killed
            period.writes += changed
            period.deliveries += ant.delivered
            if ant.respawned:
                period.omitted_respawn_moves += ant.move
            else:
                start_position = prior.position if prior else initial_positions[i]
                period.distance += math.dist(start_position, ant.position)
        previous = frame
    collected.extend(life for life in lives if life is not None)
    deaths = [life for life in collected if life.killed]
    return Timing(variant=variant, seed=seed, arm=arm, periods=periods, injured_lives=collected,
        injured_deaths=len(deaths), deaths_without_prior_injury_write=sum(
            life.first_write is None or life.first_write >= life.end for life in deaths),
        deaths_with_inherited_writes=sum(life.inherited_writes > 0 for life in deaths),
        first_write_delays=[life.first_write - life.first_injury for life in collected if life.first_write is not None])


class Report(BaseModel):
    files_verified: int
    worlds: list[Timing]


def analyze(directory: Path, manifest: Path) -> Report:
    verified = verify_manifest(directory, manifest)
    results = []
    for variant in ('legacy', 'survival'):
        store = TapeStore(directory / variant)
        for key in store.worlds:
            path, world = store.world(*key)
            changes = {(r.tick, r.individual) for line in (path / 'updates.jsonl').read_text().splitlines()
                       if (r := UpdateRecord.model_validate_json(line)).changed}
            result = describe(frames(path), changes, store.header(*key).initial_positions,
                              variant=variant, seed=world.seed, arm=world.arm)
            if (sum(p.deaths for p in result.periods) != world.deaths
                    or sum(p.exhaustion + p.deaths for p in result.periods) != world.exhausted
                    or sum(p.active_steps for p in result.periods) != world.active_individual_steps
                    or sum(p.writes for p in result.periods) != sum(world.writes)):
                raise ValueError('分段事件与已有完整汇总不符')
            results.append(result)
    return Report(files_verified=verified, worlds=results)
