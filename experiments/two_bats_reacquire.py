"""Losing the track, and earning it back: how long sonar takes to
re-acquire a body that broke its pattern.

Object permanence (two_bats_predict.py) holds only while the other
stays predictable. When B breaks its pattern, A's model points at
empty air, and A must fall back on sonar to find B again -- sweeping
its beam across the room until an echo lands on the thing that
moved. This is the intermediary the story needed: the model does not
merely fail, it fails and then RECOVERS, and the recovery has a cost
in chirps that grows with how badly the model was wrong.

The mechanism is a search. On reappearance A aims its beam where the
model predicted B and finds nothing there; it then scans outward,
alternating left and right, one step per chirp, while B keeps
moving. Re-acquisition happens on the chirp whose beam finally falls
on B. The sharper B's break, the farther the true bearing sits from
the failed prediction, and the more chirps the sweep costs. The
number is the price of having trusted a model of a mind a moment too
long.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SONAR = 6.0
BEAM_HALF = np.deg2rad(12.0)        # sonar beam half-width
SCAN_STEP = np.deg2rad(11.0)        # spiral growth per chirp
A_POS = np.array([-1.6, -1.8, 0.9])  # A hovers and searches
PREDICTED = np.array([1.4, 0.2, 1.3])  # where A's failed model points
# the direction B actually bolted, scaled by how hard it broke
BREAK_DIR = np.array([-0.5, 1.0, 0.35])
BREAK_DIR = BREAK_DIR / np.linalg.norm(BREAK_DIR)
MAX_CHIRPS = 60


def b_after_break(k, severity):
    """B reappears offset from A's failed prediction by an amount that
    grows with how hard it broke (the model-miss of two_bats_predict
    scaled up), then drifts on. Bigger break = farther from where A is
    looking = longer to find again."""
    reappear = PREDICTED + 0.6 * severity * BREAK_DIR
    drift = np.array([0.06, 0.05, -0.02]) * k * (0.5 + 0.2 * severity)
    return reappear + drift


def bearing(frm, to):
    v = to - frm
    return np.arctan2(v[1], v[0]), np.arctan2(v[2], np.hypot(v[0], v[1]))


GOLDEN = np.deg2rad(137.508)        # spiral angular increment


def reacquire(severity, predicted_bearing, seed=0):
    """A spiral-scans outward from the failed prediction, covering
    azimuth AND elevation, until its beam lands on B (which keeps
    moving). Returns (n_chirps, beam_dirs_tried, catch_pos, b_path)."""
    az0, el0 = predicted_bearing
    b_path, tried = [], []
    for k in range(MAX_CHIRPS):
        rad = SCAN_STEP * np.sqrt(k)              # widening spiral
        theta = k * GOLDEN
        az = az0 + rad * np.cos(theta)
        el = el0 + rad * np.sin(theta)
        tried.append((az, el))
        b = b_after_break(k, severity)
        b_path.append(b)
        b_az, b_el = bearing(A_POS, b)
        d = np.linalg.norm(b - A_POS)
        daz = (az - b_az + np.pi) % (2 * np.pi) - np.pi
        if abs(daz) < BEAM_HALF and abs(el - b_el) < BEAM_HALF \
                and d < SONAR:
            return k + 1, tried, b, np.array(b_path)
    return MAX_CHIRPS, tried, None, np.array(b_path)


SEVERITIES = [("gentle break", 1.0), ("hard break", 2.2),
              ("violent break", 3.6)]


def run():
    out = {}
    for name, sev in SEVERITIES:
        pred_bearing = bearing(A_POS, PREDICTED)
        n, tried, catch, bpath = reacquire(sev, pred_bearing)
        travel = (np.linalg.norm(catch - bpath[0]) if catch is not None
                  else float("nan"))
        out[name] = dict(sev=sev, n=n, tried=tried, catch=catch,
                         bpath=bpath, pred=PREDICTED, travel=travel)
    return out


if __name__ == "__main__":
    for name, r in run().items():
        print(f"{name:15s} (severity {r['sev']:.1f}): re-acquired in "
              f"{r['n']:2d} chirps, B travelled {r['travel']*100:.0f} cm "
              f"before A's sonar found it again")
