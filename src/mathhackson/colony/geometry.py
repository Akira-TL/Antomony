"""二维圆形接触；物理约束不承担路径规划。"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float32]

@dataclass(frozen=True)
class Wall:
    id: int
    x: float
    y: float
    hx: float
    hy: float

    def overlaps(self, p: Array, radius: float) -> bool:
        qx = max(abs(float(p[0])-self.x)-self.hx, 0.0)
        qy = max(abs(float(p[1])-self.y)-self.hy, 0.0)
        return qx*qx+qy*qy < radius*radius-1e-8

    def project(self, p: Array, radius: float) -> Array:
        q = np.clip(p, [self.x-self.hx,self.y-self.hy], [self.x+self.hx,self.y+self.hy])
        delta = p-q
        d = float(np.linalg.norm(delta))
        if d >= radius:
            return p
        if d > 1e-8:
            return np.asarray(q+delta*(radius+1e-5)/d, dtype=np.float32)
        gaps = [p[0]-(self.x-self.hx), self.x+self.hx-p[0], p[1]-(self.y-self.hy), self.y+self.hy-p[1]]
        k = int(np.argmin(gaps)); result = p.copy()
        if k == 0: result[0] = self.x-self.hx-radius-1e-5
        elif k == 1: result[0] = self.x+self.hx+radius+1e-5
        elif k == 2: result[1] = self.y-self.hy-radius-1e-5
        else: result[1] = self.y+self.hy+radius+1e-5
        return result


def unit(angle: float) -> Array:
    return np.asarray([math.cos(angle),math.sin(angle)],dtype=np.float32)


def ray_distance(p: Array, direction: Array, walls: list[Wall], half: Array, radius: float, reach: float = 2.4) -> float:
    """保守膨胀矩形距离射线，仅观察传感半径内的障碍。"""
    result = reach
    for axis in range(2):
        if abs(float(direction[axis])) > 1e-8:
            edge = math.copysign(float(half[axis]-radius),float(direction[axis]))
            t = float((edge-p[axis])/direction[axis])
            if t >= 0: result = min(result,t)
    for w in walls:
        lo=np.asarray([w.x-w.hx-radius,w.y-w.hy-radius]); hi=np.asarray([w.x+w.hx+radius,w.y+w.hy+radius])
        near,far = 0.0,reach
        for axis in range(2):
            if abs(float(direction[axis])) < 1e-8:
                if p[axis] < lo[axis] or p[axis] > hi[axis]: far=-1; break
            else:
                a=float((lo[axis]-p[axis])/direction[axis]); b=float((hi[axis]-p[axis])/direction[axis])
                near=max(near,min(a,b)); far=min(far,max(a,b))
        if far >= near and far >= 0: result=min(result,near)
    return max(0.0,result)


def project_static(p: Array, walls: list[Wall], half: Array, radius: float) -> Array:
    p=np.clip(p,-half+radius,half-radius).astype(np.float32)
    for w in walls: p=w.project(p,radius)
    return np.clip(p,-half+radius,half-radius).astype(np.float32)


def relocate_for_wall(positions: Array, wall: Wall, walls: list[Wall], half: Array, radius: float) -> Array | None:
    """编辑器事务：就近安置被新墙覆盖的圆，不改变模型或执行寻路。"""
    result = positions.copy()
    affected = [i for i, p in enumerate(positions) if wall.overlaps(p, radius+.02)]
    settled = [i for i in range(len(positions)) if i not in affected]
    margin = radius+.015
    for i in affected:
        p = positions[i]
        candidates = [wall.project(p, margin)]
        for step in range(17):
            for offset in ({0.} if step == 0 else {-step*(2*radius+.04), step*(2*radius+.04)}):
                along_y = float(np.clip(p[1], wall.y-wall.hy, wall.y+wall.hy))+offset
                along_x = float(np.clip(p[0], wall.x-wall.hx, wall.x+wall.hx))+offset
                candidates.extend(np.asarray(q, np.float32) for q in (
                    (wall.x-wall.hx-margin, along_y), (wall.x+wall.hx+margin, along_y),
                    (along_x, wall.y-wall.hy-margin), (along_x, wall.y+wall.hy+margin)))
        candidates.sort(key=lambda q: float(np.sum((q-p)**2)))
        for q in candidates:
            if np.any(np.abs(q) > half-margin): continue
            if any(w.overlaps(q, radius+.008) for w in [*walls, wall]): continue
            if any(float(np.linalg.norm(q-result[j])) < 2*radius+.008 for j in settled): continue
            result[i] = q
            settled.append(i)
            break
        else:
            return None
    return result


def move_discs(positions: Array, displacements: Array, radius: float, walls: list[Wall], half: Array) -> tuple[Array,NDArray[np.bool_]]:
    result=positions.copy(); contacts=np.zeros(len(result),dtype=np.bool_)
    steps=max(1,int(math.ceil(float(np.linalg.norm(displacements,axis=1).max(initial=0))/(radius*.45))))
    for _ in range(steps):
        result += displacements/steps
        for _iteration in range(6):
            changed=False
            for i in range(len(result)):
                old=result[i].copy(); result[i]=project_static(result[i],walls,half,radius)
                touched=bool(np.linalg.norm(result[i]-old)>1e-6)
                contacts[i] |= touched; changed |= touched
            for i in range(len(result)):
                for j in range(i):
                    delta=result[i]-result[j]
                    if abs(float(delta[0]))>=2*radius or abs(float(delta[1]))>=2*radius:
                        continue
                    dist=float(np.linalg.norm(delta))
                    if dist < radius*2-1e-6:
                        normal=delta/dist if dist>1e-7 else unit((i+j)*2.399)
                        push=normal*((2*radius-dist)*.5+1e-5)
                        result[i]+=push; result[j]-=push; contacts[i]=contacts[j]=True
                        changed=True
            if not changed:
                break
        for i in range(len(result)): result[i]=project_static(result[i],walls,half,radius)
    return result,contacts
