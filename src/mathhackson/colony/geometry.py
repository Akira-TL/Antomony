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
    angle: float = 0.

    def local_vector(self, v: Array) -> Array:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.asarray([c*v[0]+s*v[1], -s*v[0]+c*v[1]], np.float32)

    def local(self, p: Array) -> Array:
        return self.local_vector(p-np.asarray([self.x, self.y], np.float32))

    def global_point(self, p: Array) -> Array:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.asarray([self.x+c*p[0]-s*p[1], self.y+s*p[0]+c*p[1]], np.float32)

    @property
    def extent(self) -> Array:
        c, s = abs(math.cos(self.angle)), abs(math.sin(self.angle))
        return np.asarray([c*self.hx+s*self.hy, s*self.hx+c*self.hy], np.float32)

    def overlaps(self, p: Array, radius: float) -> bool:
        q = self.local(p)
        dx, dy = max(abs(float(q[0]))-self.hx, 0.), max(abs(float(q[1]))-self.hy, 0.)
        return dx*dx+dy*dy < radius*radius-1e-8

    def intersects(self, other: Wall) -> bool:
        # 四个局部主轴上的分离轴检验，不用外接矩形误拒绝斜墙。
        axes = [unit(self.angle), unit(self.angle+math.pi/2),
                unit(other.angle), unit(other.angle+math.pi/2)]
        delta = np.asarray([other.x-self.x, other.y-self.y], np.float32)
        for axis in axes:
            ra = self.hx*abs(float(axis@axes[0]))+self.hy*abs(float(axis@axes[1]))
            rb = other.hx*abs(float(axis@axes[2]))+other.hy*abs(float(axis@axes[3]))
            if abs(float(delta@axis)) >= ra+rb-1e-6:
                return False
        return True

    def project(self, p: Array, radius: float) -> Array:
        if not self.overlaps(p, radius):
            return p
        local = self.local(p)
        half = np.asarray([self.hx, self.hy], np.float32)
        closest = np.clip(local, -half, half)
        delta = local-closest
        distance = float(np.linalg.norm(delta))
        if distance > 1e-8:
            return self.global_point(closest+delta*(radius+1e-5)/distance)
        axis = int(np.argmin(half-np.abs(local)))
        closest[axis] = math.copysign(float(half[axis]+radius+1e-5), float(local[axis]))
        return self.global_point(closest)


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
        local_p=w.local(p); local_d=w.local_vector(direction)
        hi=np.asarray([w.hx+radius,w.hy+radius]); lo=-hi
        near,far = 0.0,reach
        for axis in range(2):
            if abs(float(local_d[axis])) < 1e-8:
                if local_p[axis] < lo[axis] or local_p[axis] > hi[axis]: far=-1; break
            else:
                a=float((lo[axis]-local_p[axis])/local_d[axis]); b=float((hi[axis]-local_p[axis])/local_d[axis])
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
        local = wall.local(p)
        candidates = [wall.project(p, margin)]
        for step in range(17):
            for offset in ({0.} if step == 0 else {-step*(2*radius+.04), step*(2*radius+.04)}):
                along_y = float(np.clip(local[1], -wall.hy, wall.hy))+offset
                along_x = float(np.clip(local[0], -wall.hx, wall.hx))+offset
                candidates.extend(wall.global_point(np.asarray(q, np.float32)) for q in (
                    (-wall.hx-margin, along_y), (wall.hx+margin, along_y),
                    (along_x, -wall.hy-margin), (along_x, wall.hy+margin)))
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
