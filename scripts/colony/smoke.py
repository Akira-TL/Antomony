"""验证真实本地服务与命令路径；不是学习效果实验。"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
from websockets.sync.client import connect
from websockets.typing import Origin
from pydantic import BaseModel
from mathhackson.colony.protocol import Command,Frame

BASE='http://127.0.0.1:8765'

def state() -> Frame:
    with urlopen(BASE+'/api/state',timeout=5) as response:
        return Frame.model_validate_json(response.read())

def wait_until(test:Callable[[Frame],bool]) -> Frame:
    end=time.monotonic()+6
    while time.monotonic()<end:
        value=state()
        if test(value): return value
        time.sleep(.08)
    raise RuntimeError('服务状态未在期限内达到预期')

class Result(BaseModel):
    passed: list[str]
    final_seed: int
    final_ant_count: int
    final_tick: int


def main() -> None:
    checks:list[str]=[]
    with connect('ws://127.0.0.1:8765/ws',origin=Origin(BASE),max_size=2**22,max_queue=None) as ws:
        def send(c:Command) -> None:
            ws.send(c.model_dump_json())
        current=state()
        if not current.paused: send(Command(kind='pause'))
        current=wait_until(lambda s:s.paused)
        tick=current.tick
        send(Command(kind='step')); current=wait_until(lambda s:s.tick==tick+1)
        checks.append('暂停后单步只推进一次')
        send(Command(kind='learning',value=0)); current=wait_until(lambda s:all(a.frozen for a in s.ants))
        fingerprints=[a.fingerprint for a in current.ants]; counts=[a.updates for a in current.ants]; tick=current.tick
        send(Command(kind='step')); current=wait_until(lambda s:s.tick==tick+1)
        assert fingerprints==[a.fingerprint for a in current.ants]
        assert counts==[a.updates for a in current.ants]
        checks.append('冻结全体保持每只参数和更新次数')
        send(Command(kind='freeze-ant',ant=1)); current=wait_until(lambda s:not s.ants[1].frozen)
        tick=current.tick; send(Command(kind='step')); current=wait_until(lambda s:s.tick==tick+1)
        assert current.ants[1].updates==counts[1]+1
        assert all(a.fingerprint==fingerprints[a.id] for a in current.ants if a.id!=1)
        checks.append('单独恢复个体不修改其他模型')
        before=len(current.walls)
        send(Command(kind='wall',x=-10,y=0)); time.sleep(.2)
        assert len(state().walls)==before
        checks.append('墙体覆盖巢穴被拒绝')
        placed:tuple[float,float]|None=None
        for x,y in [(0.,-7.),(5.,0.),(-6.,7.),(-6.,-7.)]:
            send(Command(kind='wall',x=x,y=y,hx=.3,hy=.6)); time.sleep(.2)
            if len(state().walls)==before+1:
                placed=(x,y); break
        assert placed is not None
        send(Command(kind='erase',x=placed[0],y=placed[1])); wait_until(lambda s:len(s.walls)==before)
        checks.append('新增和删除真实墙体')
        fingerprints=[a.fingerprint for a in state().ants]
        send(Command(kind='scent',x=0,y=0)); time.sleep(.2)
        assert fingerprints==[a.fingerprint for a in state().ants]
        checks.append('环境干预不直接写入神经参数')
        send(Command(kind='reset',seed=42,count=32)); current=wait_until(lambda s:s.seed==42 and len(s.ants)==32 and s.tick<30 and not s.paused)
        checks.append('重新独立预热后恢复现场')
    result=Result(passed=checks,final_seed=current.seed,final_ant_count=len(current.ants),final_tick=current.tick)
    Path('logs/colony-smoke.json').write_text(result.model_dump_json(indent=2),encoding='utf-8')
    print(result.model_dump_json(indent=2))

if __name__=='__main__': main()
