"""Deterministic checks on gait, pose and the teaching comparison."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from communication.video.scenes.ant_drawing import Ant
from communication.video.production.toy_motion import rollout

ant=Ant(scale=1.)
ant.pose_at([0.,0.,0.],0.,0.)
legs0=[leg.get_points().copy() for leg in ant.legs]
body0=ant.shell[0].get_points().copy()
ant.pose_at([0.,0.,0.],0.,math.pi/2)
assert len(ant.legs)==6
assert all(np.max(abs(a-b.get_points()))>.10 for a,b in zip(legs0,ant.legs))
assert np.allclose(body0,ant.shell[0].get_points(),atol=1e-5)
head0=ant.shell[-1].get_center().copy()
ant.pose_at([1.3,-.7,0.],0.,math.pi/2)
assert np.allclose(ant.shell[-1].get_center()-head0,[1.3,-.7,0.],atol=1e-5)
print('PASS: six legs articulate relative to the body; pose translation preserves geometry.')

motions=[rollout(24.,name,'mixed') for name in ('fixed','always','selective')]
for motion in motions:
 assert motion.samples[-1,1]-motion.samples[0,1]>10
 assert motion.samples[-1,4]>motion.samples[0,4]+30
 assert np.isfinite(motion.samples).all()
assert np.array_equal(motions[0].samples[:,6],motions[1].samples[:,6])
assert np.array_equal(motions[0].samples[:,6],motions[2].samples[:,6])
assert np.all(motions[0].samples[:,5]==0)
assert np.max(abs(motions[0].xy-motions[1].xy))>.2
print('PASS: continuous travel and gait phase, identical pushes, frozen control, distinct computed paths.')
