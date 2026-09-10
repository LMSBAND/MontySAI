"""The ear as a streaming object: waveform in, SAI frames out.

CARFAC (carfac_ref, the google/carfac NumPy reference -- the same code
lms_sai.jsfx is verified against) runs in streaming mode: its state
rides in the CarfacParams object across calls, so the AGC and hair
cells stay warm between Monty steps exactly as they would listening to
one continuous world. The SAI layer is sai_ref's arithmetic made
stateful: an input ring per channel, one stabilized image, one frame
out per completed hop.

The frame is the sensor's raw observation. Feature extraction belongs
to the sensor module, not here -- the ear hears, it does not opine.
"""
from __future__ import annotations

import numpy as np

from . import carfac_ref as C
from . import sai_ref as R


class Ear:
    """One cochlea plus one stabilized auditory image, streaming."""

    def __init__(self, sample_rate: float) -> None:
        if int(sample_rate) != R.SR:
            raise ValueError(
                f"sai_ref is built for {R.SR} Hz, got {sample_rate}. "
                "Resample the scene, not the ear.")
        self.cfp = C.design_carfac(fs=sample_rate)
        C.carfac_init(self.cfp)
        self.n_ch = self.cfp.ears[0].car_coeffs.n_ch
        if self.n_ch != len(R.poles):
            raise AssertionError(
                f"carfac says {self.n_ch} channels, sai_ref says "
                f"{len(R.poles)} -- pole ladders have diverged")
        self._ib = np.zeros((self.n_ch, R.BUF))
        self._img = np.zeros((self.n_ch, R.S))
        self._pending = np.zeros((self.n_ch, 0))

    @property
    def image(self) -> np.ndarray:
        """The current stabilized image [n_ch, lags]. A live view --
        copy before mutating."""
        return self._img

    def hear(self, waveform: np.ndarray) -> list[np.ndarray]:
        """Feed a mono chunk; return one SAI frame per completed hop.

        Frames are copies, safe to keep. An empty list just means the
        chunk didn't cross a hop boundary yet -- the samples are not
        lost, they are pending.
        """
        naps, _, _, _, _ = C.run_segment(self.cfp, waveform[:, None])
        nap = naps[:, :, 0].T                       # [ch, time]
        self._pending = np.concatenate([self._pending, nap], axis=1)

        frames = []
        while self._pending.shape[1] >= R.HOP:
            seg, self._pending = (self._pending[:, :R.HOP],
                                  self._pending[:, R.HOP:])
            self._ib[:, :R.BUF - R.HOP] = self._ib[:, R.HOP:]
            self._ib[:, R.BUF - R.HOP:] = seg
            self._stabilize()
            frames.append(self._img.copy())
        return frames

    def _stabilize(self) -> None:
        """cpp/sai.cc StabilizeSegment, one frame, over the ring."""
        for i in range(self.n_ch):
            row = self._ib[i]
            for w in range(R.NT):
                wo = int(w * R.W * 0.5)
                prod = row[R.WRS + wo:R.WRS + wo + R.W] * R.win
                tt = int(prod.argmax())
                pv = prod[tt]
                if pv <= 0:
                    pv, tt = R.WPV, R.WPK
                tt += wo
                a = (0.025 + pv) / (0.5 + pv)
                self._img[i] = (self._img[i] * (1 - a)
                                + a * row[tt + R.ORS:tt + R.ORS + R.S])
