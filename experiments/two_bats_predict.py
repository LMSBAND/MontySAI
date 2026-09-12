#!/usr/bin/env python3
"""Watching, and learning: one bat builds a model of another and
predicts it through an occlusion.

Reconstructing the other bat (two_bats_3d.py) is watching. Learning
is what happens next: bat A observes bat B for a while, fits a model
of B's motion from the reconstructed positions alone, and then B
passes behind clutter and goes silent. During the silence A has no
echo of B at all -- the sensed-absence of Section 5, now about
another agent. The question is whether A can say where B is while it
cannot hear it, and whether A is right when B reappears.

This is the smallest thing that counts as learning ABOUT another
body: object permanence. A holds a belief about a thing it cannot
currently perceive, and that belief is either confirmed or refuted
when perception returns. Two conditions:

  B stays on its orbit  -> A predicts the reappearance to a few cm.
  B breaks its pattern   -> A's prediction misses, and the size of
                            the miss is exactly the cost of having
                            modelled another agent as a clockwork.

The second condition is the honest limit of modelling a mind: it
works precisely as long as the other stays predictable, and a free
agent is under no obligation to. That gap is where a model of an
object ends and a model of an agent would have to begin.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.two_bats_3d import make_room

OBS_NOISE = 0.05         # what A's 3D localization of B costs (few cm)
N = 60
OCCLUDE = (38, 50)       # steps B is behind clutter and silent


def true_B(k, break_pattern=False):
    """B's true position at step k. If break_pattern, B reverses its
    turn and drops height the moment it goes dark at OCCLUDE[0]."""
    n = N
    if not break_pattern or k < OCCLUDE[0]:
        ang = 2 * np.pi * k / n
        z = 1.4 + 0.3 * np.sin(2 * ang)
    else:
        k0 = OCCLUDE[0]
        ang = 2 * np.pi * k0 / n - 2 * np.pi * (k - k0) / n   # reversed
        z = 1.4 + 0.3 * np.sin(2 * (2 * np.pi * k0 / n)) \
            - 0.6 * (k - k0) / n                              # diving
    return np.array([1.5 * np.cos(ang), 1.5 * np.sin(ang), z])


def fit_orbit(pts, ks):
    """Learn a circular orbit (Kasa algebraic fit) + angular rate +
    height law from A's reconstructed B positions. Returns a
    predictor pos(k)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    R = np.sqrt(c + cx ** 2 + cy ** 2)
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    # angle = a0 + omega*k ; z = z0 + zc*cos(2*ang)+zs*sin(2*ang)
    G = np.column_stack([np.ones_like(ks), ks])
    a0, omega = np.linalg.lstsq(G, ang, rcond=None)[0]
    Z = np.column_stack([np.ones_like(ks), np.cos(2 * ang),
                         np.sin(2 * ang)])
    z0, zc, zs = np.linalg.lstsq(Z, pts[:, 2], rcond=None)[0]

    def pos(k):
        a = a0 + omega * k
        return np.array([cx + R * np.cos(a), cy + R * np.sin(a),
                         z0 + zc * np.cos(2 * a) + zs * np.sin(2 * a)])
    return pos


def run(break_pattern, seed=0):
    rs = np.random.default_rng(seed)
    obs_k = [k for k in range(N) if k < OCCLUDE[0] or k >= OCCLUDE[1]]
    seen_before = [k for k in obs_k if k < OCCLUDE[0]]
    recon = {k: true_B(k, break_pattern) + rs.normal(0, OBS_NOISE, 3)
             for k in obs_k}
    pts = np.array([recon[k] for k in seen_before])
    pred = fit_orbit(pts, np.array(seen_before, float))
    gap = list(range(OCCLUDE[0], OCCLUDE[1]))
    predicted = np.array([pred(k) for k in gap])
    truth_gap = np.array([true_B(k, break_pattern) for k in gap])
    reappear_err = float(np.linalg.norm(pred(OCCLUDE[1])
                                        - true_B(OCCLUDE[1], break_pattern)))
    return dict(recon=recon, pred=pred, predicted=predicted,
                truth_gap=truth_gap, gap=gap, seen_before=seen_before,
                obs_k=obs_k, reappear_err=reappear_err)


if __name__ == "__main__":
    for bp, label in [(False, "B stays on orbit"), (True, "B breaks pattern")]:
        r = run(bp)
        gap_err = np.median(np.linalg.norm(r["predicted"] - r["truth_gap"],
                                           axis=1))
        print(f"{label:20s}: median error through the blind window "
              f"{gap_err*100:5.1f} cm; at reappearance "
              f"{r['reappear_err']*100:5.1f} cm")
