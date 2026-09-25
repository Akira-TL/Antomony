"""固定配置的开发试跑：独立世界比较，完整报告，不作确认性结论。"""
from __future__ import annotations
from pathlib import Path
from statistics import median
from time import perf_counter
from pydantic import BaseModel
from mathhackson.colony.world import World

class Configuration(BaseModel):
    purpose: str='工程可行性试跑；不是确认性效果研究'
    seeds: tuple[int,...]=(101,102,103)
    ants: int=12
    ticks: int=400
    warmup: int=220
    force_start: int=100
    force: float=.8

class Run(BaseModel):
    seed: int
    online: bool
    delivered: int
    contacts_per_100_actions: float
    prediction_mse: float
    changed_region_mse: float
    neural_updates: int
    tick_ms_median: float
    elapsed_seconds: float

class Report(BaseModel):
    configuration: Configuration
    runs: list[Run]


def run(config:Configuration,seed:int,online:bool) -> Run:
    start=perf_counter(); world=World(seed,config.ants,config.warmup)
    for ant in world.ants: ant.brain.frozen=not online
    times:list[float]=[]; previous_error=0.; previous_samples=0
    for tick in range(config.ticks):
        if tick==config.force_start:
            world.wind=config.force
            previous_error=world.error_sum; previous_samples=world.samples
        world.tick(); times.append(world.last_ms)
    return Run(seed=seed,online=online,delivered=sum(a.delivered for a in world.ants),contacts_per_100_actions=100*world.contact_count/world.samples,prediction_mse=world.error_sum/world.samples,changed_region_mse=(world.error_sum-previous_error)/(world.samples-previous_samples),neural_updates=sum(a.brain.updates for a in world.ants),tick_ms_median=median(times),elapsed_seconds=perf_counter()-start)


def main() -> None:
    config=Configuration(); path=Path('logs/colony-development-benchmark.json')
    report=Report(configuration=config,runs=[])
    # 配置在首个结果之前落盘；每组保留独立环境，不修改现场服务。
    path.write_text(report.model_dump_json(indent=2),encoding='utf-8')
    for seed in config.seeds:
        for online in (False,True):
            result=run(config,seed,online); report.runs.append(result)
            path.write_text(report.model_dump_json(indent=2),encoding='utf-8')
            print(result.model_dump_json(),flush=True)
    print('开发试跑完成；全部结果保留，不挑选种子。',flush=True)

if __name__=='__main__': main()
