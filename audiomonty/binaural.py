"""The second ear: bearing from the range difference.

One ear gives range without direction -- measured three times over
(the doorway, the gap, the fleeing moth that walked away from a
blind Brownian hunter). Two ears fix it with arithmetic a bat does
in brainstem: the same echo arrives via two path lengths, and

    delta_r = r_right - r_left ~= (baseline / 2) * sin(bearing)

so bearing = asin(2 * delta_r / baseline), signed. The factor of 2
is not a fudge: each ear reports (mouth->moth + moth->ear)/2, since
the echo path runs mouth to moth to ear and echo_ranges halves the
round trip. The shared mouth leg cancels in the difference but
leaves every ear-range halved -- monostatic-source, bistatic-
receiver geometry. Derived, then confirmed: 60 deg read 25.5 before
the 2x, and asin(sin(60)/2)=25.6. With the SAI's lag
resolution (~8 mm of path per sample) and a 20 cm baseline, bearing
resolves to a few degrees. No new physics: the scene already renders
at any listener position; a head is two renders and a subtraction.

Each ear is a full independent CARFAC+SAI (as in the skull); the
mouth sits between them.
"""
from __future__ import annotations

import numpy as np

from .acoustics import AcousticScene
from .ear import Ear
from .sonar import echo_ranges, gated_marginal

SR = 44100


class BinauralBat:
    def __init__(self, baseline: float = 0.20,
                 artifact: float = 0.55) -> None:
        self.baseline = baseline
        self.artifact = artifact
        self.left = Ear(SR)
        self.right = Ear(SR)

    def ear_positions(self, pos: np.ndarray, heading: float):
        perp = np.array([-np.sin(heading), np.cos(heading)])
        b2 = self.baseline / 2.0
        return pos + b2 * perp, pos - b2 * perp     # left, right

    def listen(self, scene: AcousticScene, pos: np.ndarray,
               heading: float, chunk_s: float = 0.25):
        """One chirp, two ears -> (range_m, bearing_rad) of the
        nearest honest target, or (None, None).

        NOTE the clock: both ears must hear the SAME sound, so the
        scene clock is rewound between the two renders -- the two
        ears share a skull, not a timeline.
        """
        pl, pr = self.ear_positions(pos, heading)
        n = int(chunk_s * SR)
        t0 = scene._clock
        wl = scene.render(np.array([pl[0], pl[1], 0.0]), n)
        scene._clock = t0
        wr = scene.render(np.array([pr[0], pr[1], 0.0]), n)
        rl = self._nearest(self.left, wl)
        rr = self._nearest(self.right, wr)
        if rl is None or rr is None:
            return None, None
        rng = 0.5 * (rl + rr)
        s = np.clip(2.0 * (rr - rl) / self.baseline, -1.0, 1.0)
        return rng, float(np.arcsin(s))   # + = target to the LEFT

    def _nearest(self, ear: Ear, wave: np.ndarray):
        fr = ear.hear(wave)
        if not fr:
            return None
        e = echo_ranges(gated_marginal(fr))
        targets = [t["range_m"] for t in e if t["range_m"] > self.artifact]
        return min(targets, default=None)
