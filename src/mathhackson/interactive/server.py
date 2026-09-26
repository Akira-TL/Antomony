"""本地三组实时验收服务；同一把锁保护时钟、编辑和状态读取。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4
import zipfile
import zlib

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from .protocol import Control, Edit, Frame, GroupKey, ParameterView, Preview, SessionConfig
from .session import LiveSession

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "logs" / "interactive"


class Download(BaseModel):
    url: str


class Engine:
    def __init__(self, session: LiveSession) -> None:
        self.session = session
        self.lock = asyncio.Lock()
        self.error: str | None = None
        self.stopping = False
        self.view = session.frame()
        self.downloads: dict[str, Path] = {}

    def refresh(self) -> Frame:
        self.view = self.session.frame().model_copy(update={"error": self.error})
        return self.view

    def ensure_ready(self) -> None:
        if self.error:
            raise HTTPException(409, "本轮异常已停止，请保存记录并开始新一轮")

    def fail(self) -> None:
        logging.exception("实时验收停止，保留已有记录")
        self.session.paused = True
        self.error = "执行异常，已停止；请保存记录并开始新一轮"
        self.view = self.view.model_copy(update={"paused": True, "error": self.error})

    async def run(self) -> None:
        while not self.stopping:
            start = time.perf_counter()
            async with self.lock:
                if not self.session.paused and not self.error:
                    try:
                        await asyncio.to_thread(self.session.advance)
                        self.refresh()
                    except Exception:
                        self.fail()
                rate = self.session.rate
            await asyncio.sleep(max(.005, .1 / rate - (time.perf_counter() - start)))

    def export(self) -> Download:
        if not self.error:
            self.session.checkpoint()
        self.session.records.flush()
        name = f"{self.session.run_id}-step-{self.session.tick}-{uuid4().hex[:8]}.zip"
        folder = RUNS / "exports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for source in self.session.records.directory.rglob("*"):
                if source.is_file() and source.name != "frames.jsonl.gz":
                    archive.write(source, source.relative_to(self.session.records.directory))
            # 运行中的gzip文件尚未写入尾部，导出压缩索引中的完整逐步记录。
            with archive.open("frames.jsonl", "w") as stream:
                for raw in self.session.records.history:
                    stream.write(zlib.decompress(raw) + b"\n")
        self.downloads[name] = path
        return Download(url=f"/api/download/{name}")


def new_session(config: SessionConfig) -> LiveSession:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return LiveSession(config, RUNS / f"{stamp}-{uuid4().hex[:8]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(1)
    app.state.engine = Engine(new_session(SessionConfig()))
    task = asyncio.create_task(app.state.engine.run())
    try:
        yield
    finally:
        engine = app.state.engine
        engine.stopping = True
        await task
        if engine.error:
            engine.session.records.close()
        else:
            engine.session.close()


app = FastAPI(lifespan=lifespan)


def current(request: Request, *, write: bool = False) -> Engine:
    if write:
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "仅允许同源操作")
    return request.app.state.engine


@app.get("/api/health")
async def health() -> bool:
    return True


@app.get("/api/state")
async def state(request: Request) -> Frame:
    return current(request).view


@app.get("/api/parameters")
async def parameters(request: Request, group: GroupKey = "adaptive", individual: int = Query(default=0, ge=0, le=31)) -> ParameterView:
    engine = current(request)
    async with engine.lock:
        try:
            return engine.session.parameters(group, individual)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error


@app.get("/api/replay")
async def replay(request: Request, tick: int = Query(ge=0)) -> Frame:
    engine = current(request)
    async with engine.lock:
        try:
            return engine.session.records.replay(tick)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error


@app.post("/api/preview")
async def preview(request: Request, edit: Edit) -> Preview:
    engine = current(request, write=True)
    async with engine.lock:
        engine.ensure_ready()
        return engine.session.preview(edit)


@app.post("/api/edit")
async def edit_world(request: Request, edit: Edit) -> Preview:
    engine = current(request, write=True)
    async with engine.lock:
        engine.ensure_ready()
        result = engine.session.edit(edit)
        engine.refresh()
        return result


@app.post("/api/control")
async def control(request: Request, command: Control) -> Frame:
    engine = current(request, write=True)
    async with engine.lock:
        engine.ensure_ready()
        try:
            await asyncio.to_thread(engine.session.control, command)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except Exception:
            engine.fail()
            raise HTTPException(500, engine.error)
        return engine.refresh()


@app.post("/api/reset")
async def reset(request: Request, config: SessionConfig) -> Frame:
    engine = current(request, write=True)
    async with engine.lock:
        replacement = await asyncio.to_thread(new_session, config)
        if engine.error:
            engine.session.records.close()
        else:
            engine.session.close()
        engine.session, engine.error = replacement, None
        return engine.refresh()


@app.post("/api/export")
async def export(request: Request) -> Download:
    engine = current(request, write=True)
    async with engine.lock:
        return await asyncio.to_thread(engine.export)


@app.get("/api/download/{name}")
async def download(request: Request, name: str) -> FileResponse:
    path = current(request).downloads.get(name)
    if path is None:
        raise HTTPException(404, "导出记录不存在")
    return FileResponse(path, filename=name, media_type="application/zip")


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse("/interactive.html")


if (ROOT / "web/dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web/dist"), name="interactive-ui")
