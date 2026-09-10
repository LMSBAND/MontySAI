"""The acoustic layer: sound sources in space, a listener with a pose.

This is the physics the environment owes the ear, and nothing more:
each source emits a waveform, the listener hears the sum with
per-source gain and propagation delay set by geometry. Mono for now --
one cochlea, matching lms_sai.jsfx's deliberate choice. Binaural is a
later, separate object (see the SAI plugin's note on why a mono
downmix must never be dressed up as two ears).

Everything is deterministic and sample-accurate: a source's phase
continues across render calls, and moving the listener changes gain
and delay smoothly. Ground truth is always recoverable -- that is the
entire advantage of doing this in Python instead of capturing a game
engine's audio bus.

Units: positions in meters, MuJoCo world frame. Gain follows 1/r with
a floor at REF_DIST so a listener standing inside a source does not
get infinite loudness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

SPEED_OF_SOUND = 343.0   # m/s, dry air, 20 C
REF_DIST = 0.25          # gain plateau radius, meters


def tone(freq: float, harmonics: Sequence[float] = (1.0,)) -> Callable:
    """A steady complex tone: harmonic k at relative amplitude
    harmonics[k-1]. The simplest possible auditory 'object'."""
    def gen(t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t)
        for k, a in enumerate(harmonics, start=1):
            out += a * np.sin(2 * np.pi * freq * k * t)
        return out / max(sum(abs(a) for a in harmonics), 1e-9)
    return gen


def chirp_call(f0: float, f1: float, period_s: float, duty: float = 0.4
               ) -> Callable:
    """A repeating up-chirp -- a cartoon bird. Distinct from a tone in
    exactly the way the SAI cares about: its structure lives in the
    envelope and the trajectory, not in a stationary spectrum."""
    def gen(t: np.ndarray) -> np.ndarray:
        ph = np.mod(t, period_s) / period_s          # 0..1 within call
        active = ph < duty
        u = np.where(active, ph / duty, 0.0)         # 0..1 within chirp
        inst = f0 * (f1 / f0) ** u                   # exponential glide
        # integrate frequency for phase; per-sample cumsum is overkill
        # for a cartoon -- use the closed form of the exp glide
        k = np.log(f1 / f0)
        phase = 2 * np.pi * f0 * duty * period_s * (np.exp(k * u) - 1) / k
        return np.where(active, np.sin(phase) * np.sin(np.pi * u), 0.0)
    return gen


@dataclass
class SoundSource:
    """A voice attached to a place (usually an object's position)."""
    name: str
    position: np.ndarray                 # (3,) world frame
    generator: Callable                  # t [s] array -> waveform array
    level: float = 1.0


@dataclass
class AcousticScene:
    """All sources, and the one method that matters: what does a
    listener at this pose hear for the next n samples?"""
    sample_rate: float
    sources: list[SoundSource] = field(default_factory=list)
    _clock: int = 0                      # samples rendered so far

    def add(self, source: SoundSource) -> None:
        self.sources.append(source)

    def remove_all(self) -> None:
        self.sources.clear()

    def render(self, listener_pos: np.ndarray, n: int) -> np.ndarray:
        """Mono waveform at the listener for the next n samples.

        The clock advances so successive calls are one continuous
        stream -- the ear downstream carries state (AGC, hair cell)
        and feeding it disjoint snippets would reset nothing yet mean
        nothing. Propagation delay is applied per source, so a source
        10 m away is heard 29 ms in the past, which matters the moment
        two sources share a pitch but not a place.
        """
        # t from INTEGER sample indices, so a stream rendered in chunks
        # is bit-identical to the same stream rendered in one call --
        # (clock + i) / sr is the same float no matter how the calls
        # were sliced, while t0 + i/sr is not.
        t = (self._clock + np.arange(n)) / self.sample_rate
        out = np.zeros(n)
        for s in self.sources:
            d = float(np.linalg.norm(np.asarray(listener_pos) - s.position))
            gain = s.level * REF_DIST / max(d, REF_DIST)
            delay = d / SPEED_OF_SOUND
            out += gain * s.generator(t - delay)
        self._clock += n
        return out
