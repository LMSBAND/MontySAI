"""One animal learns another animal's moves.

Object permanence (two_bats_predict.py) fails when B breaks its
pattern: A's model points at empty air, reappearance error ~2.9 m.
But A stores paths. If B breaks the SAME way twice, the dodge can be
learned -- stored as a branch on B's object, B's orbit plus B's
dodge, one animal with two routes. The pre-registered prediction
(from the other Fable): the second time B runs the learned dodge,
A's reappearance error drops from ~2.9 m toward the ~6 cm permanence
floor, because the dodge is now a predicted displacement instead of
a surprise.

The control is the whole experiment. A NOVEL dodge, never seen,
must still land near 2.9 m. If it does, A learned B's specific
repertoire, not "breaks in general." If a novel dodge is also
predicted well, something is leaking (A would be cheating by seeing
the future, not remembering the past).

This is a model of another agent's behavior learned by watching --
the same object A already builds for the room and the moth, now
holding a second creature's habits. If it holds, the sentence writes
itself: one animal learned another animal's moves.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DT = 1.0
FLOOR = 0.06            # the permanence floor from two_bats_predict
OBS_NOISE = 0.05


def orbit(k):
    ang = 2 * np.pi * k / 40.0
    return np.array([1.5 * np.cos(ang), 1.5 * np.sin(ang),
                     1.4 + 0.3 * np.sin(2 * ang)])


def dodge(name):
    """A fixed evasive maneuver as a displacement sequence off the
    orbit -- B's 'move'. Two named moves plus a novel one."""
    moves = {
        "juke":  [np.array([0.0, 0.0, 0.0]),
                  np.array([-0.4, 0.5, 0.2]),
                  np.array([-0.9, 0.9, 0.5]),
                  np.array([-1.3, 1.0, 0.7]),
                  np.array([-1.5, 0.8, 0.6])],
        "dive":  [np.array([0.0, 0.0, 0.0]),
                  np.array([0.3, -0.2, -0.5]),
                  np.array([0.5, -0.3, -1.0]),
                  np.array([0.6, -0.3, -1.3]),
                  np.array([0.6, -0.2, -1.4])],
        "novel": [np.array([0.0, 0.0, 0.0]),
                  np.array([0.6, 0.6, -0.1]),
                  np.array([1.2, 0.9, 0.3]),
                  np.array([1.6, 0.8, 0.8]),
                  np.array([1.7, 0.4, 1.1])],
    }
    return moves[name]


def episode(move_name, break_at=30):
    """B flies the orbit, then at break_at runs the named dodge. Returns
    the true B path."""
    path = []
    for k in range(break_at):
        path.append(orbit(k))
    base = orbit(break_at - 1)
    for disp in dodge(move_name):
        path.append(base + disp)
    return np.array(path)


RECOGNIZE_TOL = 0.5        # onset-match radius (m) to call it a known move


class BModel:
    """A's model of B: the orbit (always) plus a library of learned
    dodges. A recognizes a dodge as it begins by matching its onset
    (first two displacements) to the closest stored move; a match
    predicts the rest of that move, an unrecognized onset falls back
    to straight extrapolation (the permanence model that fails)."""

    def __init__(self):
        self.library = []          # list of (onset_signature, full_seq)

    def learn(self, observed_dodge):
        """Store a dodge as A saw it, through localization noise."""
        sig = np.concatenate([observed_dodge[0], observed_dodge[1]])
        self.library.append((sig, [d.copy() for d in observed_dodge]))

    def _recognize(self, onset_obs):
        if len(onset_obs) < 2 or not self.library:
            return None
        sig = np.concatenate([onset_obs[0], onset_obs[1]])
        best, bd = None, np.inf
        for stored_sig, seq in self.library:
            d = np.linalg.norm(sig - stored_sig)
            if d < bd:
                best, bd = seq, d
        return best if bd < RECOGNIZE_TOL else None

    def predict_break(self, base, onset_obs, n):
        seq = self._recognize(onset_obs)
        if seq is not None:                     # a move A has seen before
            return [base + seq[min(i, len(seq) - 1)]
                    for i in range(len(onset_obs), len(onset_obs) + n)]
        v = onset_obs[-1] - onset_obs[-2] if len(onset_obs) >= 2 \
            else np.zeros(3)
        last = base + onset_obs[-1]
        return [last + v * (i + 1) for i in range(n)]


def reappearance_error(model, move_name, seed=0, onset_seen=2, gap=3):
    """A watches the first `onset_seen` steps of B's break, predicts the
    next `gap`, and we measure the error at the reappearance step."""
    rs = np.random.default_rng(seed)
    path = episode(move_name)
    bk = 30
    base = path[bk - 1]
    onset = [path[bk + i] - base + rs.normal(0, OBS_NOISE, 3)
             for i in range(onset_seen)]
    pred = model.predict_break(base, onset, gap)
    true_reappear = path[bk + onset_seen + gap - 1]
    return float(np.linalg.norm(pred[-1] - true_reappear))


def observe_dodge(move_name, seed):
    """A watches B run the move once, through localization noise."""
    rs = np.random.default_rng(seed)
    return [d + rs.normal(0, OBS_NOISE, 3) for d in dodge(move_name)]


def summary(seeds=range(20)):
    before = {m: [] for m in ("juke", "dive", "novel")}
    after = {m: [] for m in ("juke", "dive", "novel")}
    for s in seeds:
        A = BModel()
        for mv in ("juke", "dive", "novel"):
            before[mv].append(reappearance_error(A, mv, seed=s))
        A.learn(observe_dodge("juke", s))
        A.learn(observe_dodge("dive", s))
        for mv in ("juke", "dive", "novel"):
            after[mv].append(reappearance_error(A, mv, seed=s + 100))
    return before, after


if __name__ == "__main__":
    before, after = summary()
    print("reappearance error, median over 20 runs (cm):\n")
    print(f"{'move':8s} {'before':>10s} {'after':>10s}   status")
    for mv in ("juke", "dive", "novel"):
        b = np.median(before[mv]) * 100
        a = np.median(after[mv]) * 100
        tag = "LEARNED" if mv in ("juke", "dive") else "NOVEL (control)"
        print(f"{mv:8s} {b:10.1f} {a:10.1f}   {tag}")
    print("\nprediction (pre-registered): learned moves collapse toward "
          "the ~6 cm\npermanence floor; the novel control stays high. "
          "One animal learned\nanother animal's moves.")
