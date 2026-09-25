"""仅本机的仿真服务：单个时钟、类型化命令、真实状态快照。"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import numpy as np
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from .world import World
from .control.session import ComparisonSession, ComparisonFrame
from .display.telemetry import NetworkView, network_view
from .protocol import Command, Frame, apply_command, snapshot

class Engine:
    def __init__(self) -> None:
        self.session: ComparisonSession|None=None
        self.commands:asyncio.Queue[Command]=asyncio.Queue(maxsize=64)

    @property
    def world(self) -> World | None:
        return self.session.world if self.session else None

    async def run(self) -> None:
        self.session=ComparisonSession()
        while True:
            start=asyncio.get_running_loop().time()
            while not self.commands.empty():
                command=self.commands.get_nowait()
                self.session.dispatch(command)
            self.session.tick()
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
async def state() -> ComparisonFrame|None:
    return engine.session.frame() if engine.session else None

class WallPreview(BaseModel):
    valid: bool
    message: str
    displaced: int = 0

@app.get('/api/wall-preview')
async def wall_preview(x: float = Query(ge=-14,le=14), y: float = Query(ge=-10,le=10), hx: float = Query(default=.4,ge=.25,le=4), hy: float = Query(default=2.,ge=.25,le=4), angle: float = Query(default=0.,ge=-1000,le=1000)) -> WallPreview:
    world=engine.world
    if world is None: return WallPreview(valid=False,message='模型尚未就绪')
    if engine.session and engine.session.reference:
        other,_,message=engine.session.reference.plan_wall(x,y,hx,hy,angle)
        if other is None: return WallPreview(valid=False,message='普通侧：'+message)
    wall,positions,message=world.plan_wall(x,y,hx,hy,angle)
    if wall is None or positions is None: return WallPreview(valid=False,message=message)
    count=sum(float(np.linalg.norm(p-a.position))>1e-6 for p,a in zip(positions,world.ants,strict=True))
    return WallPreview(valid=True,message='可放置' if not count else f'可放置；将就近移开 {count} 只个体',displaced=count)

@app.get('/api/food-preview')
async def food_preview(x: float = Query(ge=-14,le=14), y: float = Query(ge=-10,le=10)) -> WallPreview:
    if engine.world is None:
        return WallPreview(valid=False,message='模型尚未就绪')
    if engine.session and engine.session.reference:
        valid,message=engine.session.reference.plan_food(x,y)
        if not valid: return WallPreview(valid=False,message='普通侧：'+message)
    valid,message=engine.world.plan_food(x,y)
    return WallPreview(valid=valid,message=message)

@app.get('/api/network')
async def network(ant: int = Query(default=0,ge=0,le=63)) -> NetworkView | None:
    world=engine.world
    if world is None or ant>=len(world.ants): return None
    return network_view(world,ant)

@app.get('/api/export')
async def export() -> Response:
    if engine.world is None: return Response('{}',status_code=503,media_type='application/json')
    return Response(engine.session.frame().model_dump_json(),media_type='application/json',headers={'Content-Disposition':'attachment; filename="colony-snapshot.json"'})

@app.websocket('/ws')
async def connection(ws: WebSocket) -> None:
    origin=ws.headers.get('origin')
    if origin and urlparse(origin).netloc!=ws.headers.get('host'):
        await ws.close(code=1008); return
    await ws.accept()
    async def send() -> None:
        while True:
            if engine.session is not None: await ws.send_text(engine.session.frame().model_dump_json())
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
