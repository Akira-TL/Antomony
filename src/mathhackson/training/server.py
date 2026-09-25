"""单蚁训练服务；默认暂停，与原演示服务隔离。"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (Command, RecurrentCommand, RecurrentParameterHistory,
                      RecurrentState, State, WeightTrace)
from .session import TrainingSession
from .recurrent_session import RecurrentSession

ROOT = Path(__file__).resolve().parents[3]
session: TrainingSession | None = None
recurrent_session: RecurrentSession | None = None


def current() -> TrainingSession:
    if session is None:
        raise HTTPException(503, "训练服务尚未启动")
    return session


def current_recurrent() -> RecurrentSession:
    if recurrent_session is None:
        raise HTTPException(503, "循环记忆试验尚未启动")
    return recurrent_session


async def run() -> None:
    while True:
        active = current()
        if not active.paused:
            try:
                active.step()
            except Exception:
                logging.exception("单蚁训练已暂停")
                active.error = "训练异常，已暂停；请检查 logs/training-server-*.log"
                active.paused = True
        await asyncio.sleep(.1 / active.speed)


async def run_recurrent() -> None:
    while True:
        active = current_recurrent()
        if not active.paused:
            try:
                active.step()
            except Exception:
                logging.exception("循环记忆试验已暂停")
                active.error = "训练异常，已暂停；请检查训练服务日志"
                active.paused = True
        await asyncio.sleep(.1 / active.speed)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global session, recurrent_session
    session = TrainingSession(ROOT / "logs" / "training")
    recurrent_session = RecurrentSession(ROOT / "logs" / "recurrent-training")
    task = asyncio.create_task(run())
    recurrent_task = asyncio.create_task(run_recurrent())
    yield
    task.cancel()
    recurrent_task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    with suppress(asyncio.CancelledError):
        await recurrent_task
    session.finish("服务停止，未执行外部更新", False)
    recurrent_session.finish("服务停止，未执行外部更新", False)


app = FastAPI(lifespan=lifespan)


@app.get("/api/training/state")
async def state() -> State:
    return current().state()


@app.get("/api/recurrent/state")
async def recurrent_state() -> RecurrentState:
    return current_recurrent().state()


@app.get("/api/recurrent/parameter-history")
async def recurrent_parameter_history() -> RecurrentParameterHistory:
    return current_recurrent().parameter_history()


@app.post("/api/recurrent/command")
async def recurrent_command(body: RecurrentCommand, request: Request) -> RecurrentState:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "仅允许同源操作")
    try:
        current_recurrent().command(body)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return current_recurrent().state()


@app.get("/api/training/trace")
async def trace(row: int = Query(ge=0, le=38), column: int = Query(ge=0, le=15)) -> WeightTrace:
    return current().weight_trace(row, column)


@app.get("/api/health")
async def health() -> bool:
    return session is not None


@app.post("/api/training/command")
async def command(body: Command, request: Request) -> State:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "仅允许同源操作")
    try:
        current().command(body)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return current().state()


@app.get("/api/training/checkpoints/{episode}")
async def checkpoint(episode: int) -> FileResponse:
    path = current().directory / f"episode-{episode:06d}.npz"
    if episode < 1 or not path.is_file():
        raise HTTPException(404, "记录不存在")
    return FileResponse(path, filename=path.name)


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse("/training.html")


if (ROOT / "web" / "dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist"), name="training-ui")
