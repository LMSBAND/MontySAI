"""Lesioned ears: hearing loss as configuration.

Every lesion here is a knob on the published CARFAC model or an
honest statement about neural output -- nothing ad hoc. Predictions
for each were registered BEFORE any run, against the human hearing-
loss literature: see docs/damaged_ear_predictions.md.

  dead_band=(lo_hz, hi_hz)   L1  Moore's dead region: no functioning
                                 IHC/neurons over a place range = no
                                 neural output; NAP rows in the band
                                 are zeroed.
  undamping_scale=0.5        L2  OHC loss: the cochlear amplifier's
                                 authority (zr_coeffs) scaled down.
  ihc_tau_scale=4.0          L3  TFS loss: hair-cell smoothing taus
                                 raised; the polarity corner drops
                                 and fine structure blurs.
  keep_channels=8            L4  spectral starvation: only N evenly
                                 spaced places report (CI-flavored;
                                 note this is sparse place sampling,
                                 not an N-band vocoder).
  open_loop=True             L5  recruitment: the AGC's compression
                                 removed.

Caveat, stated up front: the AudioSM's timbre vector is computed
from the RAW observation (CameraSM precedent), so it does NOT pass
through the lesioned ear. Lesion results therefore measure the
location/salience/pitch pathway; a fully lesioned percept would
need timbre-through-the-ear, which is future work and would only
make lesions MORE damaging than reported here.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import carfac_ref as C
from . import sai_ref as R
from .ear import Ear


class LesionedEar(Ear):
    def __init__(self, sample_rate: float,
                 dead_band: tuple | None = None,
                 undamping_scale: float = 1.0,
                 ihc_tau_scale: float = 1.0,
                 keep_channels: int | None = None,
                 open_loop: bool = False,
                 efferent_suppress: tuple | None = None) -> None:
        """efferent_suppress=(f_lo, f_hi, scale): the MOC pathway --
        the brain's wire back into the cochlea, which works by
        suppressing outer-hair-cell gain over a place range. Same
        physical knob as OHC loss; different politics: a lesion is
        fate, an efferent is a decision. This is how attention
        reaches the sensor in actual ears."""
        super().__init__(sample_rate)
        self._open_loop = bool(open_loop)

        if ihc_tau_scale != 1.0:
            # rebuild the whole ear with slowed hair cells -- the tau
            # is a design parameter, so redesign, don't poke state
            ihc = C.IhcTwoCapParams(
                tau_lpf=0.000080 * ihc_tau_scale,
                tau1_in=0.000200 * ihc_tau_scale,
                tau1_out=0.000500 * ihc_tau_scale,
            )
            self.cfp = C.design_carfac(fs=sample_rate, ihc_params=ihc)
            C.carfac_init(self.cfp)

        if undamping_scale != 1.0:
            cc = self.cfp.ears[0].car_coeffs
            cc.zr_coeffs = cc.zr_coeffs * float(undamping_scale)
            # state was seeded from the healthy zr; reseed
            self.cfp.ears[0].car_state.zb_memory = cc.zr_coeffs.copy()

        if efferent_suppress is not None:
            f_lo, f_hi, scale = efferent_suppress
            sel = (R.poles >= float(f_lo)) & (R.poles <= float(f_hi))
            cc = self.cfp.ears[0].car_coeffs
            cc.zr_coeffs = np.where(sel, cc.zr_coeffs * float(scale),
                                    cc.zr_coeffs)
            self.cfp.ears[0].car_state.zb_memory = cc.zr_coeffs.copy()

        self._mask = np.ones(self.n_ch, dtype=bool)
        if dead_band is not None:
            lo, hi = float(dead_band[0]), float(dead_band[1])
            self._mask &= ~((R.poles >= lo) & (R.poles <= hi))
        if keep_channels is not None and keep_channels < self.n_ch:
            keep = np.zeros(self.n_ch, dtype=bool)
            idx = np.linspace(0, self.n_ch - 1, int(keep_channels))
            keep[np.round(idx).astype(int)] = True
            self._mask &= keep

    def hear(self, waveform: np.ndarray) -> list[np.ndarray]:
        naps, _, _, _, _ = C.run_segment(self.cfp, waveform[:, None],
                                         open_loop=self._open_loop)
        nap = naps[:, :, 0].T
        nap[~self._mask, :] = 0.0            # the dead places say nothing
        self.nap_mean = np.maximum(nap, 0.0).mean(axis=1)
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
