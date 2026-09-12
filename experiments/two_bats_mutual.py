"""When both bats look at each other, the asymmetry closes.

two_bats_3d.py flew both bats blind, heads aimed along their own
paths, and their awareness of each other was lopsided (12 cm vs
65 cm). two_bats_attention.py showed the gap was attention. Here both
bats point their heads at each other every chirp -- mutual attention
-- and each locates the other to the same tight floor. Awareness
becomes symmetric the moment it is mutual.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.two_bats_3d import orbit, localize
from experiments.head_tilt import unit

N = 48


def poses(k):
    a = orbit((0, 0, 0), 2.6, 0.6, 0.0, k, N)[0]
    b = orbit((0, 0, 0), 1.5, 1.4, 2.2, k, N)[0]
    return a, b


def run(seed=3):
    rs = np.random.default_rng(seed)
    pa = np.array([poses(k)[0] for k in range(N)])
    pb = np.array([poses(k)[1] for k in range(N)])
    a_hears_b, b_hears_a = [], []
    for k in range(N):
        A, B = poses(k)
        p = localize(A, unit(B - A), B, rs)      # A looks at B
        q = localize(B, unit(A - B), A, rs)      # B looks at A
        if p is not None:
            a_hears_b.append(p)
        if q is not None:
            b_hears_a.append(q)
    return dict(pa=pa, pb=pb, a_hears_b=np.array(a_hears_b),
                b_hears_a=np.array(b_hears_a))


def med_err(recon, truth):
    return np.median([np.min(np.linalg.norm(truth - p, axis=1))
                      for p in recon]) if len(recon) else float("nan")


if __name__ == "__main__":
    r = run()
    eab = med_err(r["a_hears_b"], r["pb"]) * 100
    eba = med_err(r["b_hears_a"], r["pa"]) * 100
    print(f"A looks at B, locates it to {eab:.1f} cm")
    print(f"B looks at A, locates it to {eba:.1f} cm")
    print(f"gap: {abs(eab - eba):.1f} cm -- symmetric once attention is "
          f"mutual")
