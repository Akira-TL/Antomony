"""固定连续数据的只读验收接口，不创建模型或训练会话。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from mathhackson.training.foraging.disturbance import DisturbedColony
from .continuous import Arm, Execution, WorldResult
from .online_actor import UpdateRecord

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "logs/continuous-adaptation/20260926T123655-2"
RESULT = ROOT / "analysis/continuous-adaptation/summary.json"


class WeightPoint(BaseModel):
    tick: int
    values: list[float]


class WeightGroup(BaseModel):
    name: str
    shape: list[int]
    frozen: bool
    points: list[WeightPoint]


class Header(BaseModel):
    result: WorldResult
    food: list[float]
    initial_positions: list[list[float]]
    initial_headings: list[float]
    nest_radius: float
    signal_radius: float
    contact_radius: float
    source_strength: float


class TapeStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.execution = Execution.model_validate_json((directory / "execution.json").read_text())
        if not self.execution.completed_at or self.execution.smoke:
            raise ValueError("验收只读取已完成的正式批次")
        rows = [WorldResult.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
        self.worlds = {(r.condition, r.seed, r.arm): r for r in rows}
        if len(self.worlds) != len(rows):
            raise ValueError("世界身份重复")

    def world(self, condition: str, seed: int, arm: Arm) -> tuple[Path, WorldResult]:
        result = self.worlds.get((condition, seed, arm))
        if result is None:
            raise HTTPException(404, "没有这项登记世界")
        return self.directory / f"{condition}-{seed}-{arm}", result

    def header(self, condition: str, seed: int, arm: Arm) -> Header:
        _, result = self.world(condition, seed, arm)
        config = next(c.disturbance for c in self.execution.plan.conditions if c.name == condition)
        env = DisturbedColony(seed, self.execution.plan.environment, config)
        return Header(result=result, food=env.food.tolist(), initial_positions=[a.position.tolist() for a in env.ants],
            initial_headings=[a.heading for a in env.ants], nest_radius=self.execution.plan.environment.nest_signal_radius,
            signal_radius=config.signal_radius, contact_radius=config.contact_radius, source_strength=config.signal_strength)

    def weights(self, condition: str, seed: int, arm: Arm, individual: int) -> list[WeightGroup]:
        directory, result = self.world(condition, seed, arm)
        if not 0 <= individual < self.execution.plan.environment.ants:
            raise HTTPException(404, "个体不存在")
        if arm == "rules":
            return []
        groups: list[WeightGroup] = []
        files = [("policy", directory / f"tick-0000-ant-{individual:02d}.npz"),
                 ("motor", ROOT / self.execution.plan.motor)]
        if arm != "mlp":
            files.append(("decision", directory / f"decision-ant-{individual:02d}.npz"))
        for prefix, path in files:
            with np.load(path, allow_pickle=False) as saved:
                for name in saved.files:
                    value = saved[name]
                    if value.ndim == 0 or value.dtype.kind != "f":
                        continue
                    groups.append(WeightGroup(name=f"{prefix}.{name}", shape=list(value.shape), frozen=True,
                        points=[WeightPoint(tick=0, values=value.flatten().astype(float).tolist()),
                                WeightPoint(tick=result.steps, values=value.flatten().astype(float).tolist())]))
        if arm != "mlp":
            records = [UpdateRecord.model_validate_json(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            selected = [r for r in records if r.individual == individual]
            points = [WeightPoint(tick=0, values=[0.] * 45)]
            points.extend(WeightPoint(tick=r.tick, values=r.after) for r in selected)
            groups.insert(0, WeightGroup(name="direction.offset", shape=[9, 5], frozen=arm == "skip", points=points))
        return groups


@lru_cache(maxsize=1)
def store() -> TapeStore:
    return TapeStore(DATA)


app = FastAPI(title="连续对照只读验收", docs_url=None, redoc_url=None)


@app.get("/api/health")
def health() -> bool:
    store()
    return True


@app.get("/api/acceptance/catalog")
def catalog() -> FileResponse:
    return FileResponse(RESULT, media_type="application/json")


@app.get("/api/acceptance/{condition}/{seed}/{arm}/header")
def header(condition: str, seed: int, arm: Arm) -> Header:
    return store().header(condition, seed, arm)


@app.get("/api/acceptance/{condition}/{seed}/{arm}/trace")
def trace(condition: str, seed: int, arm: Arm) -> FileResponse:
    directory, _ = store().world(condition, seed, arm)
    return FileResponse(directory / "trajectory.jsonl.gz", media_type="application/x-ndjson", headers={"Content-Encoding": "gzip"})


@app.get("/api/acceptance/{condition}/{seed}/{arm}/updates")
def updates(condition: str, seed: int, arm: Arm) -> FileResponse:
    directory, _ = store().world(condition, seed, arm)
    return FileResponse(directory / "updates.jsonl", media_type="application/x-ndjson")


@app.get("/api/acceptance/{condition}/{seed}/{arm}/weights/{individual}")
def weights(condition: str, seed: int, arm: Arm, individual: int) -> list[WeightGroup]:
    return store().weights(condition, seed, arm, individual)


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse("/acceptance.html")


if (ROOT / "web/dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web/dist"), name="acceptance-ui")
