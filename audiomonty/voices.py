"""Voice factories, Hydra-friendly AND picklable.

Each voice is a small class instance whose __call__ maps a time array
[s] to samples, so a yaml can say

    voices:
      mug: {_target_: audiomonty.voices.Drone}
      banana: {_target_: audiomonty.voices.Siren}

and -- the part closures could not do -- torch.save can pickle the
experiment config that carries them. Monty saves its whole config
beside the trained model; a voice that cannot be pickled poisons the
save of everything the run learned.

The zoo is deliberately small and deliberately distinct IN THE
AUDITORY FRAME: a cluster, a path, and a texture -- see sm_selftest
for their measured signatures.

Lowercase aliases (drone, siren, bird) are kept for existing callers.
"""
from __future__ import annotations

import numpy as np

HARMONICS = (1.0, 0.5, 0.33, 0.25, 0.2)
_HNORM = sum(HARMONICS)


class Drone:
    """A harmonic hum with slow vibrato -- a CLUSTER in auditory
    space, not a mathematical point. The first pretraining run found
    the distinction the hard way: a perfectly steady tone lands every
    percept on one node, and a one-node graph has no edges -- to
    Monty it has no extent, no surface to move over. Real voices
    shimmer; shimmer is extent."""

    def __init__(self, freq: float = 147.0, vibrato_hz: float = 4.0,
                 vibrato_cents: float = 35.0) -> None:
        self.freq = freq
        self.vibrato_hz = vibrato_hz
        self.vibrato_cents = vibrato_cents

    def __call__(self, t: np.ndarray) -> np.ndarray:
        depth = 2.0 ** (self.vibrato_cents / 1200.0) - 1.0
        phase = 2 * np.pi * self.freq * (
            t - depth * np.cos(2 * np.pi * self.vibrato_hz * t)
            / (2 * np.pi * self.vibrato_hz))
        out = np.zeros_like(t)
        for k, a in enumerate(HARMONICS, start=1):
            out += a * np.sin(k * phase)
        return out / _HNORM


class Siren:
    """A slow harmonic glide inside the tracker's range: an object
    whose identity IS its trajectory."""

    def __init__(self, f0: float = 110.0, f1: float = 220.0,
                 period_s: float = 3.0) -> None:
        self.f0 = f0
        self.f1 = f1
        self.period_s = period_s

    def __call__(self, t: np.ndarray) -> np.ndarray:
        u = 0.5 - 0.5 * np.cos(2 * np.pi * t / self.period_s)   # 0..1..0
        k = np.log(self.f1 / self.f0)
        inst = self.f0 * np.exp(k * u)
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 0))
        phase = 2 * np.pi * np.cumsum(inst * dt)
        return (np.sin(phase) + 0.5 * np.sin(2 * phase)
                + 0.33 * np.sin(3 * phase)) / 1.83


class Bird:
    """A fast high chirp: above the polarity corner, weakly pitched,
    mostly refused by the sensor -- kept as the honest hard case."""

    def __init__(self, f0: float = 2000.0, f1: float = 4000.0,
                 period_s: float = 0.5, duty: float = 0.4) -> None:
        self.f0 = f0
        self.f1 = f1
        self.period_s = period_s
        self.duty = duty

    def __call__(self, t: np.ndarray) -> np.ndarray:
        ph = np.mod(t, self.period_s) / self.period_s
        active = ph < self.duty
        u = np.where(active, ph / self.duty, 0.0)
        k = np.log(self.f1 / self.f0)
        phase = (2 * np.pi * self.f0 * self.duty * self.period_s
                 * (np.exp(k * u) - 1) / k)
        return np.where(active, np.sin(phase) * np.sin(np.pi * u), 0.0)


# aliases for callers written against the factory-function API
drone = Drone
siren = Siren
bird = Bird
