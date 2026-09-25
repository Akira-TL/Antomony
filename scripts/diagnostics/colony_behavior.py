"""固定场景工程回归：记录聚集、转向、搬运和碰撞，不宣称学习增益。"""
from pathlib import Path
from statistics import median
import numpy as np
from pydantic import BaseModel
from mathhackson.colony.world import World


class Sample(BaseModel):
    tick: int
    delivered: int
    peak_neighbors_within_1_5: int
    radius_of_gyration: float
    contacts_per_100: float


class Report(BaseModel):
    purpose: str='工程回归；不是神经学习效果对照'
    seed: int=42
    ants: int=32
    ticks: int=600
    samples: list[Sample]=[]
    wall_result: str='未执行'
    max_heading_change: float=0.
    max_overlap: float=0.
    decisions: int=0
    updates: int=0
    median_tick_ms: float=0.


def main() -> None:
    report=Report()
    out=Path('logs/colony-behavior-regression.json')
    out.write_text(report.model_dump_json(indent=2),encoding='utf-8')
    world=World(report.seed,report.ants)
    times=[]
    for tick in range(report.ticks):
        if tick==150:
            world.field.spray(np.asarray([1.,0.],np.float32))
        if tick==350:
            report.wall_result=world.add_wall(0.,0.,.4,2.)
        previous=np.asarray([a.heading for a in world.ants])
        world.tick(); times.append(world.last_ms)
        headings=np.asarray([a.heading for a in world.ants])
        change=np.abs(np.arctan2(np.sin(headings-previous),np.cos(headings-previous)))
        report.max_heading_change=max(report.max_heading_change,float(change.max()))
        points=np.stack([a.position for a in world.ants])
        distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
        np.fill_diagonal(distances,np.inf)
        report.max_overlap=max(report.max_overlap,float(max(0.,2*world.radius-distances.min())))
        assert np.all(np.isfinite(points))
        assert all(not w.overlaps(a.position,world.radius-.001) for w in world.walls for a in world.ants)
        if (tick+1)%100==0:
            sample=Sample(tick=tick+1,delivered=sum(a.delivered for a in world.ants),
                peak_neighbors_within_1_5=int((distances<1.5).sum(axis=1).max()+1),
                radius_of_gyration=float(np.sqrt(np.mean(np.sum((points-points.mean(axis=0))**2,axis=1)))),
                contacts_per_100=100*world.contact_count/world.samples)
            report.samples.append(sample)
            out.write_text(report.model_dump_json(indent=2),encoding='utf-8')
            print(sample.model_dump_json(),flush=True)
    report.decisions=sum(a.decisions for a in world.ants)
    report.updates=sum(a.brain.updates for a in world.ants)
    report.median_tick_ms=median(times)
    out.write_text(report.model_dump_json(indent=2),encoding='utf-8')
    assert report.max_heading_change<=.42001
    assert report.max_overlap<.01
    assert report.updates==report.ants*report.ticks
    print(report.model_dump_json(indent=2),flush=True)


if __name__=='__main__':
    main()
