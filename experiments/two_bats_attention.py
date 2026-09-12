"""Look down: the asymmetry between two minds is attention, not physics.

Two bats in a room localize each other unequally (two_bats_3d.py):
A reads B to ~12 cm, B reads A to ~65 cm. The gap looks like a law of
range and geometry. It is not. B reads A badly because B is flying
its patrol with its head aimed along its own flight, roughly level,
while A is far below and off to the side -- the steep, off-boresight
direction to A is exactly where the two-cone elevation solve is worst
conditioned.

Point B's head AT A -- let it look down at the thing it wants to know
about -- and the error collapses from ~71 cm to ~8 cm, the same floor
A enjoys. The asymmetry of mutual awareness was a choice about where
to look, and pointing the sensor closes it. Which is the head-tilt
result (elevation from rolling the head) turned social: you know whom
you attend to, and the gap to another mind is the gap you decline to
look across.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.two_bats_3d import A_ORBIT, orbit, localize
from experiments.head_tilt import unit


def b_pose(k, n=48):
    return orbit((0, 0, 0), 1.5, 1.4, 2.2, k, n)     # (pos, tangent heading)


def run(seed=2, n=48):
    rs = np.random.default_rng(seed)
    true_a = np.array([A_ORBIT(k) for k in range(n)])
    tangent, aimed = [], []
    for k in range(n):
        pos, head_tan = b_pose(k)
        a = A_ORBIT(k)
        p1 = localize(pos, head_tan, a, rs)          # head along flight
        p2 = localize(pos, unit(a - pos), a, rs)     # head pointed at A
        if p1 is not None:
            tangent.append(p1)
        if p2 is not None:
            aimed.append(p2)
    return dict(true_a=true_a, tangent=np.array(tangent),
                aimed=np.array(aimed),
                b_path=np.array([b_pose(k)[0] for k in range(n)]))


def med_err(recon, truth):
    return np.median([np.min(np.linalg.norm(truth - p, axis=1))
                      for p in recon]) if len(recon) else float("nan")


if __name__ == "__main__":
    r = run()
    et = med_err(r["tangent"], r["true_a"]) * 100
    ea = med_err(r["aimed"], r["true_a"]) * 100
    print(f"B head along its flight tangent : {et:5.1f} cm  (inattentive)")
    print(f"B head pointed at A (looks down): {ea:5.1f} cm  (attending)")
    print(f"\nattention closes the gap {et/ea:.1f}x. The asymmetry was a "
          f"choice about where to look.")
