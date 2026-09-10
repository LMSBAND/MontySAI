"""Verdict anatomy: decompose WHY a take matched or broke a memory.

Grew out of the hollow-body Minor Swing session (2026-09-10), where
"no match, closest: Minor Swing" needed dissecting and the answer was
three separate mechanisms at once. This module makes that dissection
one call, because the Ibanez round wants the same cuts:

  percept_stream(path)     take -> [x, y, pitch] percepts via the SM
  best_shift(stream, g)    the single translation that maximizes fit
                           (transposition, measured as one number)
  fit_fraction(stream, g)  how much of a stream lands on the memory
  decompose(stream, g)     each non-fitting percept classified:
      'octave'    pitch ~ 2x/3x a memory node (estimator cousin)
      'centroid'  same pitch, place moved >3 channels (the box)
      'novel'     no relative at any octave (genuinely new notes)
      'feature'   right place, right pitch -- broke on features alone

Predictions on record for the solid-body session: octave-stacks
survive the guitar swap (estimator, not box), centroid-drift shrinks
(box gone), novel notes hold exactly (they are the music).
"""
from __future__ import annotations

import numpy as np

from .audio_sm import AudioSM
from .voices import WavVoice

SR = 44100.0
STEP_S = 0.25
MATCH_M = 0.01          # the LM's match radius
PITCH_TOL_HZ = 15.0     # the LM's pitch tolerance
CENTS_NEAR = 40.0
CENTS_FAR = 60.0
CH_SCALE = 0.004


def percept_stream(path: str, secs: float = 30.0) -> np.ndarray:
    """-> [n, 3] of (x, y, pitch_hz) for the first `secs` of a take."""
    v = WavVoice(path)
    sm = AudioSM()
    n = int(STEP_S * SR)
    out = []
    for k in range(int(secs / STEP_S)):
        t = (np.arange(n) + k * n) / SR
        msg = sm.step(None, {"waveform": v(t)})
        if msg.pass_message:
            out.append([msg.location[0], msg.location[1],
                        msg.non_morphological_features["pitch_hz"]])
    return np.array(out)


def graph_arrays(graph):
    """-> (node positions [n,3], node pitches [n]) from a stored graph."""
    nodes = graph.pos.numpy()
    fm = graph.feature_mapping
    pitches = graph.x.numpy()[:, fm["pitch_hz"][0]:fm["pitch_hz"][1]].ravel()
    return nodes, pitches


def fit_fraction(stream: np.ndarray, nodes: np.ndarray,
                 shift_x: float = 0.0) -> float:
    pts = stream[:, :3].copy()
    pts[:, 0] -= shift_x
    pts[:, 2] = 0            # compare in (x, y) plane
    nd = nodes.copy()
    nd[:, 2] = 0
    d = np.linalg.norm(nd[None, :, :] - pts[:, None, :], axis=2)
    return float((d.min(axis=1) <= MATCH_M).mean())


def best_shift(stream: np.ndarray, nodes: np.ndarray,
               span: float = 0.05) -> tuple[float, float]:
    """-> (shift_x meters, fit at that shift). One translation only --
    the transposition axis."""
    shifts = np.linspace(-span, span, int(span / 0.00025) * 2 + 1)
    fits = [fit_fraction(stream, nodes, s) for s in shifts]
    i = int(np.argmax(fits))
    return float(shifts[i]), float(fits[i])


def decompose(stream: np.ndarray, nodes: np.ndarray,
              node_pitches: np.ndarray) -> dict:
    """Classify every percept. -> dict of lists of stream rows."""
    out = {"fit": [], "octave": [], "centroid": [], "novel": [],
           "feature": []}
    for row in stream:
        x, y, bp = row
        d = np.linalg.norm(nodes[:, :2] - [x, y], axis=1)
        j = int(d.argmin())
        if d[j] <= MATCH_M and abs(bp - node_pitches[j]) <= PITCH_TOL_HZ:
            out["fit"].append(row)
            continue
        r1 = np.min(np.abs(1200 * np.log2(bp / node_pitches)))
        r2 = np.min(np.abs(1200 * np.log2(bp / (2 * node_pitches))))
        r3 = np.min(np.abs(1200 * np.log2(bp / (3 * node_pitches))))
        jp = int(np.argmin(np.abs(1200 * np.log2(bp / node_pitches))))
        dy = (y - nodes[jp, 1]) / CH_SCALE
        if min(r2, r3) < CENTS_NEAR and r1 > CENTS_FAR:
            out["octave"].append(row)
        elif r1 < CENTS_NEAR and abs(dy) > 3:
            out["centroid"].append(row)
        elif r1 < CENTS_NEAR:
            out["feature"].append(row)
        else:
            out["novel"].append(row)
    return {k: np.array(v) if v else np.empty((0, 3))
            for k, v in out.items()}


def summarize(name: str, dec: dict) -> str:
    n = sum(len(v) for v in dec.values())
    parts = [f"{k} {len(v)}" for k, v in dec.items()]
    return f"{name}: {n} percepts -- " + ", ".join(parts)
