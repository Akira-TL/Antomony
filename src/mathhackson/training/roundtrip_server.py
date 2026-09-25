"""仅本机的往返训练验收服务；与已有单蚁训练进程隔离。"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .roundtrip_session import RoundTripSession
from .schemas import RoundTripCommand, RoundTripState

ROOT = Path(__file__).resolve().parents[3]
FOUNDATION = ROOT / "checkpoints" / "recurrent" / "foundation-episode-002570.npz"
session: RoundTripSession | None = None


def current() -> RoundTripSession:
    if session is None:
        raise HTTPException(503, "往返训练尚未启动")
    return session


async def run() -> None:
    while True:
        active = current()
        if not active.paused:
            try:
                active.step()
            except Exception:
                logging.exception("往返训练已暂停")
                active.error = "训练异常，已暂停；请检查服务日志"
                active.paused = True
        await asyncio.sleep(.1 / active.speed)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global session
    session = RoundTripSession(ROOT / "logs" / "roundtrip-training", FOUNDATION)
    task = asyncio.create_task(run())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    session.finish(False)


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health() -> bool:
    return session is not None


@app.get("/api/roundtrip/state")
async def state() -> RoundTripState:
    return current().state()


@app.post("/api/roundtrip/command")
async def command(body: RoundTripCommand, request: Request) -> RoundTripState:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "仅允许同源操作")
    try:
        current().command(body)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return current().state()


@app.get("/api/roundtrip/checkpoints/{episode}")
async def checkpoint(episode: int) -> FileResponse:
    path = current().directory / f"episode-{episode:06d}.npz"
    if episode < 1 or not path.is_file():
        raise HTTPException(404, "记录不存在")
    return FileResponse(path, filename=path.name)


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse("/roundtrip.html")


if (ROOT / "web" / "dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist"), name="roundtrip-ui")
