#!/usr/bin/env python3
"""Night hunt: the dark panel, flipped.

The 3D moth hunt (moth_hunt_3d.py) ended on a result that is
backwards for a nocturnal animal: climbing moth, ears 0/6, ears+eyes
6/6, DARK 0/6 -- the bat needed light to hear UP. Real bats hunt at
midnight. Head tilt (head_tilt.py) supplies the missing piece: roll
the head, chirp twice, intersect the two interaural cones, and
elevation comes from the ears alone. No photons required.

Four conditions on the same climbing moth, same ear model (~1 deg
bearing noise, 4.3 m sonar), eyes as in moth_hunt_3d (55 deg
half-FOV, 20 m, lit only):

    level ears, dark          -- the cone of confusion: escape
    level ears + eyes, LIT    -- vision supplies the vertical: catch
    level ears + eyes, DARK   -- the old embarrassment: escape
    TILTED ears, dark         -- the flip: catch, lights off

The level-ear conditions keep the honest pin (heading z forced to 0
every step -- see head_tilt.py's docstring for the drift bug that
makes this non-negotiable).
"""
import numpy as np

from head_tilt import ROLL, M_NOISE, baseline_axis, solve_dir, unit

SONAR = 4.3
CATCH = 0.7
FOV = np.deg2rad(55.0)


def hunt_night(tilt_head=False, use_eyes=False, lit=False,
               moth_speed=0.19, seed=0, max_steps=200):
    r = np.random.default_rng(seed)
    bat = np.array([0.0, 0.0, 0.0])
    heading = unit(np.array([1.0, 0.0, 0.0]))
    moth = np.array([2.2, 1.2, 0.3])
    STEP = 0.34
    bp, mp = [bat.copy()], [moth.copy()]
    for _ in range(max_steps):
        v = moth - bat
        d = np.linalg.norm(v)
        if d < CATCH:
            return np.array(bp), np.array(mp), True
        desired = None
        if use_eyes and lit and unit(v) @ heading > np.cos(FOV) and d < 20:
            desired = unit(v + r.normal(0, 0.01, 3))       # full 3D fix
        if desired is None and d < SONAR:
            true_dir = unit(v)
            if tilt_head:
                a1 = baseline_axis(heading, +ROLL)
                a2 = baseline_axis(heading, -ROLL)
                m1 = a1 @ true_dir + r.normal(0, M_NOISE)
                m2 = a2 @ true_dir + r.normal(0, M_NOISE)
                desired = solve_dir(a1, m1, a2, m2, heading)
            else:
                a = baseline_axis(heading, 0.0)            # level ears:
                m = a @ true_dir + r.normal(0, M_NOISE)    # azimuth only
                ang = np.arcsin(np.clip(m, -1, 1))
                fwd = unit(np.array([heading[0], heading[1], 0.0]))
                ca, sa = np.cos(-ang), np.sin(-ang)
                desired = np.array([ca * fwd[0] - sa * fwd[1],
                                    sa * fwd[0] + ca * fwd[1], 0.0])
        heading = (unit(heading + r.normal(0, 0.5, 3)) if desired is None
                   else unit(0.6 * heading + 0.4 * desired))
        if not tilt_head and not (use_eyes and lit):
            heading[2] = 0.0                # the pin: level ears, no eyes,
            heading = unit(heading)         # nothing licenses a climb
        bat = bat + STEP * heading
        away = unit(moth - bat) * 0.5 + np.array([0, 0, 1.0])
        moth = moth + moth_speed * unit(away + r.normal(0, 0.25, 3))
        bp.append(bat.copy()); mp.append(moth.copy())
    return np.array(bp), np.array(mp), False


CONDITIONS = [
    ("level ears, dark",        dict(tilt_head=False, use_eyes=False, lit=False)),
    ("level ears + eyes, LIT",  dict(tilt_head=False, use_eyes=True,  lit=True)),
    ("level ears + eyes, DARK", dict(tilt_head=False, use_eyes=True,  lit=False)),
    ("TILTED ears, dark",       dict(tilt_head=True,  use_eyes=False, lit=False)),
]


def campaign(n=10):
    out = {}
    for name, kw in CONDITIONS:
        runs = [hunt_night(seed=s, **kw) for s in range(n)]
        out[name] = runs
        print(f"{name:26s} {sum(r[2] for r in runs)}/{n}")
    return out


if __name__ == "__main__":
    campaign(10)
