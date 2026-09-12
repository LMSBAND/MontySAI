#!/usr/bin/env python3
"""Real sonar: a real click, a real room, a tape measure, the model ear.

Bryan's rig (2026-09-12): SM57 lying on top of a studio monitor,
pointed the same way; a click track through the monitor; a tape
measure. Four takes in ~/Desktop/chords/bats/:

    click_100towall.wav    wall at 110 in = 2.794 m, path clear
    click_withme_at_23.wav Bryan standing 23 in = 0.584 m in front
    click_with_hand_at1.wav a hand 1 in from the mic
    click_roomnoise.wav    the room alone, no clicks

Pipeline identical to the simulated bat: CARFAC+SAI (Ear), one
gated_marginal per click, echo_ranges. Speaker and mic are
colocated, so range = c*lag/2 exactly as in the sim's monostatic
geometry.

READINGS (strength-weighted 10 cm histogram over 40 clicks):
    wall take:  tower at 2.9 m (tape 2.794 + the known envelope-
                latency bias) -- the WALL, on real hardware.
    me at 23":  towers 0.5-0.6 m (tape 0.584); wall strength drops
                287 -> 121 -- a body is both an echo and a shadow.
    hand at 1": 0.025 m is below the 34 cm blanking floor (the
                instrument's stated minimum range, from its own call
                lobe) so the hand itself is UNRANGEABLE -- but the
                wall collapses to ~81 (occlusion) and a 0.5 m tower
                appears: the ear ranges the arm's owner standing
                behind the hand.
    room noise: no tower above 56. No emission, no ranges.
    all click takes: the fixed ~0.3-0.45 m artifact family known
                from the sim, now reproduced on a real click.

Lesson relearned mid-analysis (look before theorising): band
PRESENCE is not evidence -- echo_ranges' MAD threshold adapts, so
even pure room noise yields in-band peaks. The discriminators are
strength and CONCENTRATION; and a mis-centered "tower window" gave
a backwards occlusion ratio until the histogram was actually drawn.
"""
import json
import wave

import numpy as np

from audiomonty.ear import Ear
from audiomonty.sonar import echo_ranges, gated_marginal

TAKES = "/home/bryan/Desktop/chords/bats"
WALL_M = 110 * 0.0254
BRYAN_M = 23 * 0.0254
FILES = ["click_100towall", "click_withme_at_23",
         "click_with_hand_at1", "click_roomnoise"]


def load(path):
    w = wave.open(path)
    sr, n, ch, sw = (w.getframerate(), w.getnframes(),
                     w.getnchannels(), w.getsampwidth())
    raw = w.readframes(n)
    assert sw == 3, "these takes are 24-bit"
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3 * ch)[:, :3]
    d = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)
         | (a[:, 2].astype(np.int32) << 16))
    d = (d << 8) >> 8                     # sign-extend 24-bit
    return sr, d.astype(float) / (2 ** 23)


def click_times(d, sr):
    peaks = np.where(np.abs(d) > 0.5 * np.abs(d).max())[0]
    cl = [int(peaks[0])]
    for p in peaks[1:]:
        if p - cl[-1] > 0.2 * sr:
            cl.append(int(p))
    return cl


def range_all(name, max_clicks=40):
    sr, d = load(f"{TAKES}/{name}.wav")
    if name == "click_roomnoise":         # no clicks: same cadence anyway
        cl = list(range(sr, len(d) - sr // 2, sr // 4))[:max_clicks]
    else:
        cl = click_times(d, sr)[:max_clicks]
    ear = Ear(float(sr))
    take = []
    for c in cl:
        chunk = d[max(c - sr // 200, 0): c + sr // 5]
        frames = ear.hear(chunk)
        if frames:
            t = echo_ranges(gated_marginal(frames), float(sr))
            take.append([(r["range_m"], r["strength"]) for r in t])
    return take


def strength_hist(take, edges):
    h = np.zeros(len(edges) - 1)
    for t in take:
        for r, s in t:
            i = np.searchsorted(edges, r) - 1
            if 0 <= i < len(h):
                h[i] += s
    return h


if __name__ == "__main__":
    edges = np.arange(0.3, 4.6, 0.1)
    out = {}
    for name in FILES:
        take = range_all(name)
        out[name] = take
        h = strength_hist(take, edges)
        top = sorted(np.argsort(h)[::-1][:4])
        print(f"{name:24s} clicks={len(take):2d} peaks: "
              + ", ".join(f"{edges[i]:.1f}m={h[i]:.0f}" for i in top))
    json.dump(out, open("data/real_sonar_ranges.json", "w"))
    print("truth: wall", round(WALL_M, 3), "m; Bryan", round(BRYAN_M, 3),
          "m; hand 0.025 m (below the 0.34 m blanking floor)")
