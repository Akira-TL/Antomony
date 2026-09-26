"""逐帧记录与当前运行的压缩回看索引，不覆盖既有记录。"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import subprocess
import zlib

from pydantic import BaseModel

from mathhackson.training.comparison.continuous import ContinuousPlan
from mathhackson.training.comparison.online_actor import UpdateRecord
from mathhackson.training.comparison.run import SourceRecord
from .protocol import Control, Frame, Edit, GroupKey, SessionConfig


class Execution(BaseModel):
    source_commit: str
    started_at: str
    config: SessionConfig
    plan: ContinuousPlan
    sources: list[SourceRecord]
    qualification: str = "工程验收；固定接受，不代表学会更新时机或通过效果验证"


class Intervention(BaseModel):
    tick: int
    edit: Edit


class GroupUpdate(BaseModel):
    group: GroupKey
    record: UpdateRecord


class ControlEvent(BaseModel):
    tick: int
    command: Control


class RecordStore:
    def __init__(self, directory: Path, config: SessionConfig, plan: ContinuousPlan) -> None:
        paths = {Path(plan.motor), *[plan.policy_path(i) for i in range(config.ants)],
                 *[plan.gate_path(i) for i in range(config.ants)], *[plan.mlp_path(i) for i in range(config.ants)]}
        execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            started_at=datetime.now(timezone.utc).isoformat(), config=config, plan=plan,
            sources=[SourceRecord(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)])
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        (directory / "execution.json").write_text(execution.model_dump_json(indent=2))
        self.frames = gzip.open(directory / "frames.jsonl.gz", "xt")
        self.updates = (directory / "updates.jsonl").open("x")
        self.interventions = (directory / "interventions.jsonl").open("x")
        self.controls = (directory / "controls.jsonl").open("x")
        self.history: list[bytes] = []

    def append(self, frame: Frame) -> None:
        if frame.tick != len(self.history):
            raise ValueError("运行记录时序不连续")
        raw = frame.model_dump_json()
        self.history.append(zlib.compress(raw.encode(), level=1))
        self.frames.write(raw + "\n")
        if frame.tick % 16 == 0:
            self.flush()

    def replay(self, tick: int) -> Frame:
        if not 0 <= tick < len(self.history):
            raise ValueError("回看时点尚未发生")
        return Frame.model_validate_json(zlib.decompress(self.history[tick]))

    def intervention(self, tick: int, edit: Edit) -> None:
        self.interventions.write(Intervention(tick=tick, edit=edit).model_dump_json() + "\n")
        self.flush()

    def update(self, group: GroupKey, record: UpdateRecord) -> None:
        self.updates.write(GroupUpdate(group=group, record=record).model_dump_json() + "\n")

    def control(self, tick: int, command: Control) -> None:
        self.controls.write(ControlEvent(tick=tick, command=command).model_dump_json() + "\n")
        self.flush()

    def flush(self) -> None:
        for stream in (self.frames, self.updates, self.interventions, self.controls):
            stream.flush()

    def close(self) -> None:
        for stream in (self.frames, self.updates, self.interventions, self.controls):
            stream.close()
