"""Folie a deux: the demolition test for consensus.

The whole humility result (MontyConsensus, the veto, the bats agreeing
to 8.9 cm) rests on one unstated premise: that the witnesses err
INDEPENDENTLY. Independent errors cancel when you average, and
disagreement between witnesses honestly tracks how wrong they are. So
the machine trusts agreement.

This test breaks that premise on purpose. N observers estimate a
target's direction; each carries a bearing error. In the INDEPENDENT
condition the errors are drawn per observer. In the CORRELATED
condition they share a common bias (a miscalibration they all inherit
-- the same training data, the same broken assumption, the same
cultural prior) plus a little private noise.

Two things are measured: the consensus error (how far the averaged
estimate lands from the truth) and the disagreement (how far the
observers land from each other). The prediction that would sink the
humility result: under correlation, the consensus error does NOT fall
with N, AND the disagreement COLLAPSES -- so the observers agree most
precisely at the exact moment they are all wrong together. A
consensus rule that reads agreement as confidence is not just
unhelped by correlation; it is actively fooled into maximal
confidence on a false answer.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SIGMA = 0.25            # bearing error scale (radians-ish, in a plane)
TARGET = np.array([4.0, 0.0, 1.5])


def estimates(n, correlated, rs):
    """Each observer sits at the origin-ish and estimates the target
    direction with error; return their estimated target POINTS at the
    true range."""
    r = np.linalg.norm(TARGET)
    true_dir = TARGET / r
    if correlated:
        shared = rs.normal(0, SIGMA, 3)                # common bias
        errs = shared + rs.normal(0, SIGMA * 0.25, (n, 3))
    else:
        errs = rs.normal(0, SIGMA, (n, 3))             # independent
    pts = []
    for e in errs:
        d = true_dir + e
        d = d / np.linalg.norm(d)
        pts.append(r * d)
    return np.array(pts)


def trial(n, correlated, seed):
    rs = np.random.default_rng(seed)
    pts = estimates(n, correlated, rs)
    consensus = pts.mean(axis=0)
    cons_err = np.linalg.norm(consensus - TARGET)
    # disagreement: mean pairwise distance among observers
    dis = np.mean([np.linalg.norm(pts[i] - pts[j])
                   for i in range(n) for j in range(i + 1, n)])
    return cons_err, dis, pts


def sweep(correlated, ns=(2, 4, 8, 16, 32), reps=200):
    out = {}
    for n in ns:
        ce, dg = [], []
        for s in range(reps):
            c, d, _ = trial(n, correlated, s)
            ce.append(c); dg.append(d)
        out[n] = (np.median(ce), np.median(dg))
    return out


if __name__ == "__main__":
    ind = sweep(False)
    cor = sweep(True)
    print("N   | independent errors      | correlated errors")
    print("    | consensus  disagreement | consensus  disagreement")
    for n in ind:
        ci, di = ind[n]; cc, dc = cor[n]
        print(f"{n:3d} |   {ci*100:5.1f} cm    {di*100:5.1f} cm   |   "
              f"{cc*100:5.1f} cm    {dc*100:5.1f} cm")
    print("\nIndependent: consensus error falls with N, disagreement "
          "tracks it (honest).")
    print("Correlated:  consensus error plateaus, disagreement COLLAPSES "
          "-- the crowd agrees most")
    print("precisely when it is wrong together. Agreement-as-confidence "
          "is fooled.")
