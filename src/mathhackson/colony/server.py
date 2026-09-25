"""仅本机的仿真服务：单个时钟、类型化命令、真实状态快照。"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from .world import World
from .protocol import Command, Frame, apply_command, snapshot

class Engine:
    def __init__(self) -> None:
        self.world: World|None=None
        self.commands:asyncio.Queue[Command]=asyncio.Queue(maxsize=64)

    async def run(self) -> None:
        self.world=World()
        while True:
            start=asyncio.get_running_loop().time()
            while not self.commands.empty():
                command=self.commands.get_nowait()
                self.world=apply_command(self.world,command)
            if not self.world.paused: self.world.tick()
            elapsed=asyncio.get_running_loop().time()-start
            await asyncio.sleep(max(.003,self.world.dt/self.world.rate-elapsed))

engine=Engine()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    task=asyncio.create_task(engine.run())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError): await task

app=FastAPI(title='MathHackson 独立神经蚁群',lifespan=lifespan,docs_url=None,redoc_url=None)

class Health(BaseModel):
    ready: bool
    tick: int

@app.get('/api/health')
async def health() -> Health:
    return Health(ready=engine.world is not None,tick=engine.world.tick_count if engine.world else 0)

@app.get('/api/state')
async def state() -> Frame|None:
    return snapshot(engine.world) if engine.world else None

@app.get('/api/export')
async def export() -> Response:
    if engine.world is None: return Response('{}',status_code=503,media_type='application/json')
    return Response(snapshot(engine.world).model_dump_json(),media_type='application/json',headers={'Content-Disposition':'attachment; filename="colony-snapshot.json"'})

@app.websocket('/ws')
async def connection(ws: WebSocket) -> None:
    origin=ws.headers.get('origin')
    if origin and urlparse(origin).netloc!=ws.headers.get('host'):
        await ws.close(code=1008); return
    await ws.accept()
    async def send() -> None:
        while True:
            if engine.world is not None: await ws.send_text(snapshot(engine.world).model_dump_json())
            await asyncio.sleep(.1)
    sender=asyncio.create_task(send())
    try:
        while True:
            text=await ws.receive_text()
            if len(text)>4096:
                await ws.close(code=1009); break
            try:
                command=Command.model_validate_json(text)
                engine.commands.put_nowait(command)
            except (ValidationError,asyncio.QueueFull):
                if engine.world: engine.world.event('error','无效操作或命令过快，未执行')
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError,WebSocketDisconnect,RuntimeError): await sender

static=Path(__file__).resolve().parents[3]/'web'/'dist'
if static.is_dir(): app.mount('/',StaticFiles(directory=static,html=True),name='web')
