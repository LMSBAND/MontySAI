"""Two eyes: stereo localization, the long-range complement to sonar.

Honest scope: the log-polar RETINA (scale-invariance, Test-1-for-eyes)
is proven separately in experiments/logpolar_scale.py. HERE the eyes'
job is navigation -- bearing and range to a target -- so they are
modelled geometrically, the way a point-mass models a body: two
apertures a baseline apart, each seeing the target's bearing; the
disparity gives range.

The complementarity is the whole point, and it is real physics:
  - stereo range error GROWS with distance (disparity shrinks as
    baseline/range, so far targets pin poorly) -- eyes are sharp on
    bearing far out but vague on range far out.
  - eyes need LIGHT and a forward field of view, and see far
    (no 4.3 m horizon).
  - the ear (binaural CARFAC) is the mirror: superb range up close,
    works in the dark, but deaf past ~4.3 m.
So: eyes acquire and vector from afar; ears close the kill. Each
covers exactly the other's blind spot -- which is why an animal has
both.
"""
from __future__ import annotations

import numpy as np


class Eyes:
    def __init__(self, baseline=0.06, fov_deg=70.0, horizon=20.0,
                 ang_res_deg=0.4, lit=True, rng=None):
        self.baseline = baseline
        self.fov = np.deg2rad(fov_deg)
        self.horizon = horizon
        self.ang_res = np.deg2rad(ang_res_deg)   # angular quantization
        self.lit = lit
        self.rng = rng or np.random.default_rng(0)

    def look(self, pos, heading, target):
        """-> (range, bearing) to target, or (None, None). Bearing is
        relative to heading, + = target to the left."""
        if not self.lit:
            return None, None
        v = np.asarray(target) - np.asarray(pos)
        d = float(np.hypot(v[0], v[1]))
        if d > self.horizon or d < 1e-3:
            return None, None
        world_ang = np.arctan2(v[1], v[0])
        bearing = (world_ang - heading + np.pi) % (2*np.pi) - np.pi
        if abs(bearing) > self.fov / 2:
            return None, None            # out of the field of view
        # bearing: sharp, small noise
        b = bearing + self.rng.normal(0, self.ang_res)
        # stereo range from disparity, quantized by angular resolution
        # -> error grows with distance (the honest limit of stereo)
        disparity = self.baseline / d               # small-angle
        dq = max(disparity + self.rng.normal(0, self.ang_res),
                 self.ang_res)          # can't resolve below one step
        r = self.baseline / dq
        return r, b
