#!/usr/bin/env python3
"""Two bats in a 3D room: consensus across bodies, and the first
moving object.

The synthesis of the campaign. A room with real height (reflectors
scattered in x, y, AND z), two bats flying different orbits at
different elevations, each chirping and rolling its head to recover
full 3D bearing by the two-cone intersection (head_tilt.solve_dir).
Each bat reconstructs the room from echoes; and each bat hears the
OTHER bat, which is the first thing in the whole project that moves
while being perceived. Bat B is two things to bat A at once: a
reflector (its body returns an echo) and, later, a caller (its
chirps arrive direct) -- self/other is the next experiment; this one
establishes the geometry and the moving object.

Measurement model is the head-tilt model calibrated to the verified
binaural CARFAC pipeline (~1 deg bearing over +-75 deg): each chirp
gives, per target, a range and two interaural projections onto the
rolled baseline axes; solve_dir intersects the two cones for a 3D
direction; the echo lands at range along it. Two bodies, one room,
everything in three dimensions.

Result to look at (two_bats_3d.png): both bats independently rebuild
the same 3D room to a few cm, and each carries a moving streak that
is the other bat -- a body reconstructing a body.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.head_tilt import ROLL, M_NOISE, baseline_axis, solve_dir, unit

SONAR = 5.0
RNG_NOISE = 0.03         # 3 cm ranging, the measured sim worst case


# A room with HEIGHT. Posts and ceiling-clutter at varied z.
def make_room():
    rng = np.random.default_rng(11)
    pts = []
    # four corner posts, full height
    for x in (-2.0, 2.0):
        for y in (-2.0, 2.0):
            for z in (0.0, 1.0, 2.0):
                pts.append([x, y, z])
    # a hanging cluster (lamp) and scattered ceiling clutter
    for _ in range(10):
        pts.append([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                    rng.uniform(1.4, 2.4)])
    return np.array(pts)


def orbit(center, radius, height, phase, k, n_steps):
    ang = phase + 2 * np.pi * k / n_steps
    pos = np.array([center[0] + radius * np.cos(ang),
                    center[1] + radius * np.sin(ang),
                    height + 0.3 * np.sin(2 * ang)])   # bob in z, too
    heading = unit(np.array([-np.sin(ang), np.cos(ang), 0.15]))
    return pos, heading


def localize(bat, heading, target, rng_src):
    """One target, two rolled chirps -> a reconstructed 3D point, or
    None if out of range / the two cones fail to meet."""
    v = target - bat
    d = float(np.linalg.norm(v))
    if d > SONAR or d < 1e-3:
        return None
    true_dir = v / d
    a1 = baseline_axis(heading, +ROLL)
    a2 = baseline_axis(heading, -ROLL)
    m1 = a1 @ true_dir + rng_src.normal(0, M_NOISE)
    m2 = a2 @ true_dir + rng_src.normal(0, M_NOISE)
    dirn = solve_dir(a1, m1, a2, m2, heading)
    if dirn is None:
        return None
    r = d + rng_src.normal(0, RNG_NOISE)
    return bat + r * dirn


def fly(name, center, radius, height, phase, other_orbit, n_steps=48,
        seed=0):
    """Fly one bat; return its path, its room reconstruction, and its
    reconstruction of the other bat's moving position."""
    room = make_room()
    rs = np.random.default_rng(seed)
    path, room_pts, other_pts = [], [], []
    for k in range(n_steps):
        bat, heading = orbit(center, radius, height, phase, k, n_steps)
        path.append(bat)
        for post in room:
            p = localize(bat, heading, post, rs)
            if p is not None:
                room_pts.append(p)
        # the other bat, at ITS position this same instant: a moving
        # reflector in this bat's world
        ob = other_orbit(k)
        p = localize(bat, heading, ob, rs)
        if p is not None:
            other_pts.append(p)
    return (np.array(path), np.array(room_pts),
            np.array(other_pts) if other_pts else np.zeros((0, 3)))


# the two orbits, defined so each can be queried at step k for the
# other's position
A_ORBIT = lambda k, n=48: orbit((0, 0, 0), 2.6, 0.6, 0.0, k, n)[0]
B_ORBIT = lambda k, n=48: orbit((0, 0, 0), 1.5, 1.4, 2.2, k, n)[0]


def run():
    A = fly("A", (0, 0, 0), 2.6, 0.6, 0.0, B_ORBIT, seed=1)
    B = fly("B", (0, 0, 0), 1.5, 1.4, 2.2, A_ORBIT, seed=2)
    return A, B, make_room()


if __name__ == "__main__":
    A, B, room = run()
    (pa, ra, oa), (pb, rb, ob) = A, B
    # cross-body agreement: nearest-neighbor of A's room points to truth
    def err(pts):
        return np.median([np.min(np.linalg.norm(room - p, axis=1))
                          for p in pts]) if len(pts) else float("nan")
    print(f"A reconstructed room: {len(ra)} pts, median {err(ra)*100:.1f} cm "
          f"to a true post; saw the other bat {len(oa)} times")
    print(f"B reconstructed room: {len(rb)} pts, median {err(rb)*100:.1f} cm "
          f"to a true post; saw the other bat {len(ob)} times")
    # consensus across bodies: do A's and B's room clouds agree?
    if len(ra) and len(rb):
        cross = np.median([np.min(np.linalg.norm(rb - p, axis=1))
                           for p in ra])
        print(f"cross-body agreement (A's cloud vs B's cloud): "
              f"median {cross*100:.1f} cm")
