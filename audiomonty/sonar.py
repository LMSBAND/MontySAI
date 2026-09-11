"""Sonar mode: reading ranges off the SAI's lag axis.

The melody tracker asks "what is the ONE period here?" and its
first-peak rule stops at the call's own 1 ms lobe -- measured 12x
taller than a 2 m echo. Sonar asks a different question of the SAME
image: "at which lags does energy stand above the floor?" Every
answer is a target, and its range is c * lag / 2.

Two rules, both borrowed from actual bats:

  BLANKING -- no ranging inside the emission's own lobe. Real bats
  cannot hear during their own call and shorten calls on approach;
  ours excludes lag < blank_s. With a 1 ms call, 2 ms of blanking
  sets the minimum range at 34 cm.

  ALL PEAKS, NOT FIRST -- a melody has one period; a room has many
  walls. Local maxima above median + k*MAD (robust to the marginal's
  DC-blocker undershoot), each refined by the same parabola the
  pitch tracker earned its 1.4 cents with. k_mad=2.5, measured: the
  first cut used 8 and demanded a threshold three times taller than
  a clean 2 m echo ridge -- statistical machismo, corrected against
  the data.

  KNOWN BIAS, calibratable: the NAP envelope lags the waveform by
  the hair-cell smoothing plus the call duration, which shifts the
  ridge ~0.5 ms late = ~+9 cm at all ranges. Every physical sonar
  carries a processing-latency calibration; ours is measured by the
  sweep below and left IN the readings here so the raw instrument
  is what's on display.
"""
from __future__ import annotations

import numpy as np

from . import sai_ref as R

SPEED_OF_SOUND = 343.0


def gated_marginal(frames: list) -> np.ndarray:
    """EMISSION GATING, the third rule borrowed from bats: process
    echoes call-by-call. The SAI's silence-fallback -- the honest
    forgetting that keeps melodies current -- erases a sparse echo
    within ~4 frames of the call (measured: ridge at +164, ashes by
    +1.3, ten frames later). A bat does not read its sonar between
    calls; neither do we. Take the frame where the ranging window
    holds the most energy: that is the frame the call just lit."""
    lo, hi = R.LO, R.HI
    best, best_e = None, -np.inf
    for f in frames:
        mg = f.sum(axis=0)
        e = float(mg[lo:hi + 1].max())
        if e > best_e:
            best_e, best = e, mg
    return best


def echo_ranges(mg: np.ndarray, sample_rate: float = float(R.SR),
                blank_s: float = 0.002, k_mad: float = 2.5,
                max_targets: int = 6) -> list[dict]:
    """-> [{'range_m', 'lag_s', 'strength'}] sorted near to far."""
    lo = max(R.LO, int(blank_s * sample_rate))
    hi = R.HI
    seg = mg[lo:hi + 1]
    base = float(np.median(seg))
    mad = float(np.median(np.abs(seg - base))) or 1e-12
    thresh = base + k_mad * mad

    out = []
    for i in range(lo + 1, hi):
        if mg[i] > thresh and mg[i] > mg[i - 1] and mg[i] >= mg[i + 1]:
            ym, y0, yp = mg[i - 1], mg[i], mg[i + 1]
            den = ym - 2 * y0 + yp
            d = 0.0 if den == 0 else max(-0.5, min(0.5,
                                                   0.5 * (ym - yp) / den))
            lag_s = (i + d) / sample_rate
            out.append({"range_m": SPEED_OF_SOUND * lag_s / 2.0,
                        "lag_s": lag_s,
                        "strength": float((y0 - base) / mad)})
    # adjacent bins of one ridge -> keep the strongest within 0.1 m
    out.sort(key=lambda e: e["range_m"])
    merged = []
    for e in out:
        if merged and e["range_m"] - merged[-1]["range_m"] < 0.1:
            if e["strength"] > merged[-1]["strength"]:
                merged[-1] = e
        else:
            merged.append(e)
    merged.sort(key=lambda e: -e["strength"])
    merged = merged[:max_targets]
    merged.sort(key=lambda e: e["range_m"])
    return merged
