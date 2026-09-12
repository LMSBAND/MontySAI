#!/usr/bin/env python3
"""Head tilt: elevation from ears alone, by moving the head.

A horizontal ear-baseline is elevation-blind -- the interaural range
difference fixes a CONE around the baseline axis (the cone of
confusion), and every direction on that cone reads identical. The 3D
hunt (moth_hunt_3d.py) showed it: climbing moth, ears 0/6.

But the cone is attached to the HEAD. Roll the head and the baseline
axis rolls with it; a second chirp draws a second cone; two cones
through the same moth intersect in (generically) two directions, and
only one of them is in front of you. Elevation recovered with ears
alone -- no eyes, no new physics, just the sensorimotor loop: move
the sensor, intersect the readings.

    chirp i:  a_i . dir = m_i          (a_i = baseline axis, unit)
    |dir| = 1, dir forward             -> solve_dir()

Measurement model calibrated to the verified binaural pipeline
(audiomonty/binaural.py: ~1 deg worst-case bearing over +/-75 deg,
so noise on m = sin(bearing) is ~0.017).

The honest control, and the bug it fixes: the "fixed head" bat must
be PINNED level (heading z forced to 0 every step). An early draft
let a stray z-component survive unit() renormalization, drift 0.1 ->
0.97, and fake 8/8 catches on a climbing moth. Level flight +
in-plane ears = elevation-blind, and the split must show it BEFORE
any figure gets drawn: fixed ~0/N, tilt catching.

Bryan's framing: pointing the head is the same act that aims the
eye's field of view and the ear's gain -- active sensing is one
gesture serving every sensor on the skull.
"""
import numpy as np

SONAR = 4.3
CATCH = 0.7
ROLL = np.deg2rad(40.0)
M_NOISE = 0.017          # sin-domain noise ~ 1 deg bearing error


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def baseline_axis(heading, roll):
    """Ear-baseline axis for a head pointed along `heading`, rolled
    by `roll` about it. Roll 0 = level baseline (horizontal perp)."""
    e = np.cross([0.0, 0.0, 1.0], heading)
    if np.linalg.norm(e) < 1e-9:          # looking straight up/down
        e = np.array([0.0, 1.0, 0.0])
    e = unit(e)
    u = unit(np.cross(heading, e))
    return np.cos(roll) * e + np.sin(roll) * u


def solve_dir(a1, m1, a2, m2, forward):
    """Intersect two interaural cones: a1.dir=m1, a2.dir=m2, |dir|=1.

    Both axes are perpendicular to `forward`, so with A = [a1; a2;
    forward] the third coordinate c = dir.forward is free; |dir|=1
    is quadratic in c. Two roots = the two cone intersections; keep
    the forward one."""
    A = np.stack([a1, a2, forward])
    if abs(np.linalg.det(A)) < 1e-9:
        return None                        # axes parallel: one cone twice
    Ai = np.linalg.inv(A)
    p = Ai @ np.array([m1, m2, 0.0])
    q = Ai @ np.array([0.0, 0.0, 1.0])
    aa = q @ q
    bb = 2.0 * (p @ q)
    cc = p @ p - 1.0
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0:                           # noise pushed cones apart
        return None
    cands = []
    for s in (1.0, -1.0):
        c = (-bb + s * np.sqrt(disc)) / (2.0 * aa)
        v = p + c * q
        if v @ forward > 0:
            cands.append(unit(v))
    if not cands:
        return None
    return max(cands, key=lambda v: v @ forward)


def hunt_tilt(tilt_head, moth_speed=0.19, seed=0, max_steps=200):
    """One hunt. tilt_head=False: level flight, heading z PINNED to 0
    (the honest elevation-blind control). tilt_head=True: two chirps
    per step at roll +/-ROLL, elevation from solve_dir."""
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
        if d < SONAR:
            true_dir = unit(v)
            if tilt_head:
                a1 = baseline_axis(heading, +ROLL)
                a2 = baseline_axis(heading, -ROLL)
                m1 = a1 @ true_dir + r.normal(0, M_NOISE)
                m2 = a2 @ true_dir + r.normal(0, M_NOISE)
                desired = solve_dir(a1, m1, a2, m2, heading)
            else:
                a = baseline_axis(heading, 0.0)   # level baseline:
                m = a @ true_dir + r.normal(0, M_NOISE)
                los = v.copy()
                los[2] = 0.0                      # azimuth only --
                if np.linalg.norm(los) > 0:       # the cone collapses
                    ang = np.arcsin(np.clip(m, -1, 1))
                    fwd = unit(np.array([heading[0], heading[1], 0.0]))
                    ca, sa = np.cos(-ang), np.sin(-ang)
                    desired = np.array([ca * fwd[0] - sa * fwd[1],
                                        sa * fwd[0] + ca * fwd[1], 0.0])
        heading = (unit(heading + r.normal(0, 0.5, 3)) if desired is None
                   else unit(0.6 * heading + 0.4 * desired))
        if not tilt_head:
            heading[2] = 0.0                      # THE PIN: level flight,
            heading = unit(heading)               # every single step
        bat = bat + STEP * heading
        away = unit(moth - bat) * 0.5 + np.array([0, 0, 1.0])  # climber
        moth = moth + moth_speed * unit(away + r.normal(0, 0.25, 3))
        bp.append(bat.copy()); mp.append(moth.copy())
    return np.array(bp), np.array(mp), False


def campaign(n=10):
    out = {}
    for name, tilt in [("fixed", False), ("tilt", True)]:
        runs = [hunt_tilt(tilt, seed=s) for s in range(n)]
        out[name] = runs
        wins = sum(r[2] for r in runs)
        print(f"{name:5s} head: {wins}/{n} catches "
              f"({[('Y' if r[2] else '.') for r in runs]})")
    return out


if __name__ == "__main__":
    # solve_dir self-check against planted directions, before any hunt
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(500):
        h = unit(rng.normal(size=3))
        tgt = unit(h + 0.8 * rng.normal(size=3))
        if tgt @ h <= 0.05:
            continue
        a1 = baseline_axis(h, +ROLL)
        a2 = baseline_axis(h, -ROLL)
        got = solve_dir(a1, a1 @ tgt, a2, a2 @ tgt, h)
        if got is not None:
            errs.append(np.degrees(np.arccos(np.clip(got @ tgt, -1, 1))))
    print(f"solve_dir noiseless: n={len(errs)} "
          f"worst={max(errs):.3f} deg median={np.median(errs):.3f} deg")
    campaign(10)
