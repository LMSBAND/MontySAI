"""Self versus other: two bats learn to stop talking over each other.

Every experiment so far let A hear only its own echoes. Put a second
caller in the room and A faces the first genuinely social problem:
some of the energy coming back is A's echo, and some is B's call
arriving direct. If A cannot tell them apart, B's call lands in A's
map as a phantom object -- a ghost at a range nothing is at. This is
the self/other distinction in its rawest form: to not be fooled, A
must model that a second voice exists and is not its own.

Real bats solve it with the jamming-avoidance response: when two
chirp near the same frequency they shift apart, each away from the
other, until their calls no longer overlap. We do not code the
divergence. We give each bat one local rule -- 'if I hear energy in
my own band I cannot account for, edge my frequency away from it' --
and let the separation emerge. Two bats starting on the same
frequency should drive their voices apart on their own, and the
ghosts in each map should clear as they do.

Control: lock the frequencies and the ghosts never clear. The
self/other problem is solved by behavior, and the behavior is
emergent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.two_bats_3d import make_room, orbit

F0 = 40.0               # kHz, both bats start here
BAND = 3.0              # kHz: calls this close jam each other
SHIFT = 0.8             # kHz edged away per jammed chirp
FMIN, FMAX = 32.0, 56.0
N = 50


def positions(k):
    a = orbit((0, 0, 0), 2.6, 0.6, 0.0, k, N)[0]
    b = orbit((0, 0, 0), 1.5, 1.4, 2.2, k, N)[0]
    return a, b


def run(jar=True, seed=0):
    """jar=True: bats may shift frequency (emergent divergence).
    jar=False: frequencies locked (the control)."""
    rs = np.random.default_rng(seed)
    fa, fb = F0 + 0.1, F0 - 0.1          # a hair apart to break symmetry
    room = make_room()
    hist = []                            # (k, fa, fb, jammed, ghosts)
    for k in range(N):
        pa, pb = positions(k)
        jammed = abs(fa - fb) < BAND
        # when jammed, B's direct call plants a ghost in A's map at a
        # range set by the call's arrival, not any real reflector
        ghosts = []
        if jammed:
            d = np.linalg.norm(pb - pa)
            ghost_r = d * rs.uniform(0.4, 0.9)      # phantom range
            dirn = (pb - pa) / (d + 1e-9)
            ghosts.append(pa + ghost_r * dirn)
        hist.append((k, fa, fb, jammed, ghosts))
        if jar and jammed:
            # each edges away from the other, symmetric divergence
            fa = np.clip(fa + SHIFT * np.sign(fa - fb), FMIN, FMAX)
            fb = np.clip(fb + SHIFT * np.sign(fb - fa), FMIN, FMAX)
    return dict(hist=hist, room=room, fa=fa, fb=fb)


if __name__ == "__main__":
    for jar, label in [(True, "JAR on (bats may shift)"),
                       (False, "JAR off (frequencies locked)")]:
        r = run(jar)
        jam = sum(1 for h in r["hist"] if h[3])
        ghosts = sum(len(h[4]) for h in r["hist"])
        # when did the last jam happen?
        last = max([h[0] for h in r["hist"] if h[3]], default=-1)
        print(f"{label:32s}: {jam:2d}/{N} chirps jammed, {ghosts:2d} "
              f"ghosts; last jam at step {last:2d}; "
              f"final gap {abs(r['fa']-r['fb']):.1f} kHz")
