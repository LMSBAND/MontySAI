"""A body is a moving measurement: A finds a door because B flew
through it.

The cheapest social learning, and it needs no message. A wall with a
gap divides the room. Bat A patrols its own side and maps the wall
from echoes, but a gap returns no echo, and a missing echo is
ambiguous: it could be an opening, or a bearing A never sampled (the
sensed-absence problem of Section 5, now in space). A alone cannot
certify the door is passable.

Bat B flies through the door. A already carries B as an object
(two_bats_3d.py), so A watches B cross, and every point B occupies
is certified empty the way an echo at range r certifies the disc
inside it empty: a body is proof of free space. Where B's trail
pierces the wall plane, there is a door, and A learns it without
ever going there. Learning from another's success -- B got through,
so there is a way through -- by physics alone, no teaching, no
signal.

The wall knowledge comes from A's own echoes; the passability of the
gap comes from B's body. Neither bat has the whole story; the room
is understood only across the two of them.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

WALL_X = 0.0
DOOR_Y = (0.4, 1.2)                 # true gap in the wall
DOOR_CENTER = np.mean(DOOR_Y)       # 0.8
SONAR = 5.0
OBS_NOISE = 0.05


def wall():
    """A full-height wall of posts at x=0, with a gap (the door)."""
    pts = []
    for y in np.arange(-2.0, 2.01, 0.3):
        if DOOR_Y[0] <= y <= DOOR_Y[1]:
            continue
        for z in (0.0, 0.7, 1.4, 2.1):
            pts.append([WALL_X, y, z])
    return np.array(pts)


def a_patrol(n=40):
    """A flies an arc on its own side (x < 0), never crossing."""
    t = np.linspace(0, 1, n)
    ang = np.pi * (0.25 + 0.5 * t)              # sweep facing the wall
    return np.column_stack([-1.6 + 0.4 * np.cos(4 * np.pi * t),
                            -2.0 + 4.0 * t,
                            0.7 + 0.4 * np.sin(2 * np.pi * t)])


def b_through_door(n=40):
    """B crosses from A's side to the far side, through the gap."""
    t = np.linspace(0, 1, n)
    x = -1.8 + 3.6 * t                          # crosses x=0 at t=0.5
    y = DOOR_CENTER + 0.15 * np.sin(2 * np.pi * t)
    z = 1.0 + 0.2 * t
    return np.column_stack([x, y, z])


def run(seed=0):
    rs = np.random.default_rng(seed)
    W = wall()
    A = a_patrol()
    B = b_through_door()

    # A maps the wall from echoes (posts in range, with noise)
    wall_recon = []
    for bat in A:
        for post in W:
            if np.linalg.norm(post - bat) < SONAR:
                wall_recon.append(post + rs.normal(0, OBS_NOISE, 3))
    wall_recon = np.array(wall_recon)

    # A reconstructs B's crossing; each point is certified free space
    b_recon = np.array([p + rs.normal(0, OBS_NOISE, 3) for p in B])

    # the door: where B's certified-free trail pierces the wall plane
    cross = []
    for i in range(len(b_recon) - 1):
        x0, x1 = b_recon[i, 0], b_recon[i + 1, 0]
        if (x0 - WALL_X) * (x1 - WALL_X) < 0:            # straddles x=0
            f = (WALL_X - x0) / (x1 - x0)
            cross.append(b_recon[i] + f * (b_recon[i + 1] - b_recon[i]))
    door = np.mean(cross, axis=0) if cross else None
    return dict(W=W, A=A, B=B, wall_recon=wall_recon, b_recon=b_recon,
                door=door)


if __name__ == "__main__":
    r = run()
    d = r["door"]
    print(f"wall posts A mapped from its own echoes: {len(r['wall_recon'])}")
    print(f"A never crossed the wall (max x on patrol: "
          f"{r['A'][:,0].max():+.2f})")
    if d is not None:
        print(f"door inferred from B's crossing at y = {d[1]:.2f} m, "
              f"z = {d[2]:.2f} m  (true door center y = {DOOR_CENTER:.2f})")
        print(f"  error: {abs(d[1]-DOOR_CENTER)*100:.1f} cm, from a body "
              f"A watched pass through a wall it never touched")
