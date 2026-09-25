"""所选个体最近一次真实前向计算的部分连线；不生成装饰性神经活动。"""
from __future__ import annotations
from pydantic import BaseModel
from ..world import World
from ..brain import Brain


class ConnectionView(BaseModel):
    layer: int
    source: int
    target: int
    weight: float
    contribution: float
    update: float


class NetworkView(BaseModel):
    tick: int
    ant: int
    paused: bool
    frozen: bool
    updates: int
    layers: list[list[float]]
    widths: list[int]
    connections: list[ConnectionView]


def network_view(world: World, ant_id: int) -> NetworkView | None:
    ant=world.ants[ant_id]
    brain=ant.brain
    if not isinstance(brain,Brain):
        return None
    indices=[list(range(8)),[0,4,8,12,16,20],[0,3,6,9,12,15],list(range(3))]
    activations=[ant.inputs,brain.first_hidden,brain.hidden,brain.output]
    matrices=[brain.w1,brain.w2,brain.inference_head]
    edges=[]
    for layer,matrix in enumerate(matrices):
        for source,i in enumerate(indices[layer]):
            for target,j in enumerate(indices[layer+1]):
                weight=float(matrix[i,j])
                delta=float(brain.last_head_delta[i,j]) if layer==2 and not brain.frozen else 0.
                edges.append(ConnectionView(layer=layer,source=source,target=target,weight=weight,
                    contribution=float(activations[layer][i])*weight,update=delta))
    return NetworkView(tick=world.tick_count,ant=ant_id,paused=world.paused,frozen=brain.frozen,updates=brain.updates,
        layers=[[float(values[i]) for i in ids] for values,ids in zip(activations,indices,strict=True)],
        widths=[8,24,16,3],connections=edges)
