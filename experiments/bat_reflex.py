#!/usr/bin/env python3
"""The blind flinch: obstacle avoidance from the SAI marginal alone.

Three-line brainstem: chirp; if the nearest echo ridge is under
NEAR, veer; else relax heading toward +x. Always step -- the first
version turned in place and pirouetted forever at (2.49, 1.40); a
flinch that freezes is a statue. No map, no Monty, no vision.

Result on the post-wall world: nine flinches, zero collisions,
escaped the field in 39 steps -- by ROUNDING THE WALL'S END, not
threading the gap. Honest finding: a mono ear gives range without
bearing, so it avoids everything but cannot see a doorway. Blind-
cane navigation. Threading gaps needs bearing (a second ear) or a
room memory (Monty). Reflex first, cortex second.

Run from tbp.monty: uv run python ../experiments/bat_reflex.py
"""
import numpy as np

from audiomonty.acoustics import AcousticScene, SoundSource, Reflector
from audiomonty.voices import BatCall
from audiomonty.ear import Ear
from audiomonty.sonar import echo_ranges, gated_marginal

SR = 44100
STEP, TURN, NEAR = 0.22, np.deg2rad(38), 0.55


def run():
    scene = AcousticScene(SR)
    for y in np.arange(-2.0, 2.01, 0.25):
        if 0.2 <= y <= 1.4:
            continue                       # the gap it will never see
        scene.add_reflector(Reflector(f"p{y:.2f}",
                                      np.array([3.0, y, 0.0])))
    src = SoundSource("call", np.zeros(3), BatCall(), level=0.6,
                      self_atten=0.1)
    scene.add(src)
    ear = Ear(SR)

    pos, heading = np.array([0.0, -1.2]), 0.0
    path = [pos.copy()]
    for step in range(200):
        src.position = np.array([pos[0], pos[1], 0.0])
        fr = []
        for _ in range(2):
            fr += ear.hear(scene.render(np.array([pos[0], pos[1], 0.0]),
                                        int(0.25 * SR)))
        e = echo_ranges(gated_marginal(fr))
        nearest = min((t["range_m"] for t in e), default=99.0)
        if nearest < NEAR:
            heading += TURN
        else:
            heading *= 0.8
        pos = pos + STEP * np.array([np.cos(heading), np.sin(heading)])
        path.append(pos.copy())
        if pos[0] > 3.6:
            print(f"escaped at step {step}: ({pos[0]:.2f}, {pos[1]:.2f})")
            break
    return np.array(path)


if __name__ == "__main__":
    run()
