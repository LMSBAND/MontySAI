#!/usr/bin/env python3
"""3D moth hunt: the sensory federation, elevation, and acquisition.

Ear = model calibrated to the verified binaural CARFAC pipeline
(range + ~1 deg azimuth; audiomonty/binaural.py). A horizontal
ear-baseline is elevation-BLIND (the cone of confusion), so a
climbing target escapes sonar. Eye = geometric stereo (range + full
3D bearing, lit only; the log-polar retina's scale-invariance is
proven separately in logpolar_scale.py). See audiomonty/eyes.py.

Three results:
  ELEVATION (moth climbs):  ears 0/6, ears+eyes 6/6, dark 0/6.
    Two ears cannot hear UP; two eyes supply the vertical; the dark
    control proves it was vision.
  SPEED, inside the sonar bubble (level moth, starts near): ears and
    eyes both hold 100% until the moth outruns the bat (~100% speed).
    Sonar suffices in its own range.
  SPEED, beyond the horizon (level moth, starts at 8 m): ears blind
    (~0%); eyes acquire and hold 100% to the same physical speed
    limit. The eye's edge is ACQUISITION, not the stern-chase --
    and, per the log-polar result, the eye holds the moth's identity
    as it dilates on approach. Once the eye is on, it's on.
"""
import numpy as np

SONAR = 4.3
CATCH = 0.7


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def hunt3d(use_eyes, lit=True, moth_speed=0.19, climb=True,
           start=None, seed=0, max_steps=200):
    r = np.random.default_rng(seed)
    bat = np.array([0., 0., 0.])
    heading = unit(np.array([1., 0., 0.]))
    moth = np.array(start if start is not None else [2.2, 1.2, 0.3])
    STEP = 0.34
    bp, mp = [bat.copy()], [moth.copy()]
    for _ in range(max_steps):
        v = moth - bat
        d = np.linalg.norm(v)
        if d < CATCH:
            return np.array(bp), np.array(mp), True
        desired = None
        if use_eyes and lit and unit(v) @ heading > np.cos(np.deg2rad(55)) \
                and d < 20:
            desired = unit(v + r.normal(0, 0.01, 3))       # full 3D fix
        if desired is None and d < SONAR:
            los = v.copy(); los[2] = 0.                     # ears: in-plane
            if np.linalg.norm(los) > 0:
                los = unit(los)
                desired = unit(np.array([los[0], los[1], heading[2]]))
        heading = (unit(heading + r.normal(0, 0.5, 3)) if desired is None
                   else unit(0.6 * heading + 0.4 * desired))
        bat = bat + STEP * heading
        away = unit(moth - bat) * 0.5
        if climb:
            away = away + np.array([0, 0, 1.0])
        else:
            away = away + r.normal(0, 0.35, 3); away[2] = 0
        moth = moth + moth_speed * unit(away + r.normal(0, 0.25, 3))
        bp.append(bat.copy()); mp.append(moth.copy())
    return np.array(bp), np.array(mp), False
