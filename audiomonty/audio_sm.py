"""AudioSM: the ear as a Monty sensor module.

Monty's cortical messaging protocol wants ONE feature-at-location per
step, with a pose. That constraint, taken seriously, decides what an
auditory object IS here: the SAI's most salient point traces a path
through (place, lag) space as the sound evolves, and the object is
that trajectory with features along it. A steady drone is a single
point revisited; a chirp is a curve; a riff is a shape. Monty already
learns objects as graphs of feature-at-location visits -- we are
handing it a space where sounds are such graphs natively.

THE AUDITORY LOCATION. location = (x, y, z) with
    x = log2(lag / 1 ms) * LAG_SCALE     (octaves of period, scaled)
    y = channel * CH_SCALE               (place on the membrane)
    z = 0                                (the frame is flat, for now)
Log-lag because pitch is perceived in ratios -- an octave is the same
distance everywhere, which makes the same melodic shape at two
transpositions two SIMILAR graphs, not two unrelated ones. The scales
put typical call trajectories in the same few-centimeter range Monty's
learning modules are tuned for on YCB objects.

THE POSE. pose_vectors rows: a fixed 'normal' out of the auditory
plane, then the local ridge orientation in-plane and its orthogonal.
pose_fully_defined=False -- the honest statement that, like a point
on a sphere where curvature directions are arbitrary, the auditory
frame's in-plane orientation is weak evidence. The LM already knows
what to do with that flag.

WHAT MOVEMENT MEANS (the RECON.md question, answered by construction):
time. The stream strobes; successive frames are the saccades. Agent
pose modulates level and (later) direction, and rides along in the
non-morphological features, but displacement in the object's own frame
comes from the sound's evolution, not the walker's.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)

from . import sai_ref as R
from .audio_env import AUDIO_SENSOR_ID
from .ear import Ear

# THE MAGNIFICATION OF AUDITORY SPACE. Monty's machinery carries YCB
# assumptions in its constants: graph dedup at 1 mm, match distance
# 1 cm, max graph size 30 cm. An auditory frame has no intrinsic
# meters, so we are free to print it at whatever scale makes those
# constants sensible -- and obliged to, as the first pretraining run
# proved: at 2 cm/octave a vibrato cluster was 0.6 mm wide, the 1 mm
# dedup swallowed it whole, and the drone collapsed back into the
# single point the vibrato existed to prevent. At 10 cm/octave a
# 35-cent vibrato is a 3 mm cluster and a one-octave siren is a 10 cm
# path: object-sized, by construction.
LAG_SCALE = 0.1       # meters per octave of lag
CH_SCALE = 0.004      # meters per cochlear channel
SALIENCE_FLOOR = 1.6  # peak must beat this x the mean marginal to count


class AudioSM:
    """SensorModule turning waveform observations into auditory percepts.

    Satisfies tbp.monty.frameworks.models.abstract_monty_classes.
    SensorModule by structure (step / update_state / reset /
    state_dict); registered as an ABC virtual subclass would also work,
    but Monty type-checks by duck typing at runtime.
    """

    def __init__(
        self,
        sensor_module_id: str = AUDIO_SENSOR_ID,
        sample_rate: float = 44100.0,
        save_raw_obs: bool = False,
        lesion: dict | None = None,
        timbre_source: str = "raw",
    ) -> None:
        """lesion: kwargs for lesions.LesionedEar (dead_band,
        undamping_scale, ihc_tau_scale, keep_channels, open_loop).
        None = healthy ear. See docs/damaged_ear_predictions.md.

        timbre_source: "raw" (harmonic bands from the raw
        observation -- CameraSM precedent, fine for healthy-ear
        work) or "ear" (harmonic-place sampling of the NAP, so the
        timbre pathway passes through any lesion). LESION STUDIES
        MUST USE "ear": with "raw", the LM has a side channel to
        the healthy signal and damage is masked -- measured, not
        hypothetical (tau x8 read correct with raw, confused
        without). Post-AGC timbre is weaker (the AGC equalizes
        spectra); that weakness is the ear's truth, not a bug."""
        self.sensor_module_id = sensor_module_id
        self.features = ["pitch_hz", "pitch_semis", "level", "salience",
                         "place_centroid", "timbre"]
        self.save_raw_obs = save_raw_obs
        self._sample_rate = sample_rate
        self._lesion = dict(lesion) if lesion else None
        self._timbre_source = timbre_source
        self._ear = self._make_ear()
        self.state: SensorState | None = None
        self.is_exploring = False
        self.processed_obs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _make_ear(self):
        if self._lesion:
            from .lesions import LesionedEar
            return LesionedEar(self._sample_rate, **self._lesion)
        return Ear(self._sample_rate)

    def reset(self) -> None:
        self._ear = self._make_ear()
        self.processed_obs = []
        self.is_exploring = False

    def update_state(self, agent: AgentState) -> None:
        # The ear rides the agent, unrotated: a mono ear has no gaze.
        self.state = SensorState(
            position=agent.position, rotation=agent.rotation
        )

    def state_dict(self):
        return {"processed_observations": self.processed_obs}

    def propose_goals(self):
        return []

    # ------------------------------------------------------------------
    def step(self, ctx, observation, motor_only_step: bool = False
             ) -> Message:
        wave = observation["waveform"]
        frames = self._ear.hear(np.asarray(wave, dtype=np.float64))
        img = frames[-1] if frames else self._ear.image

        percept = self._percept_from_image(img, wave)
        if motor_only_step:
            percept.pass_message = False
        if not self.is_exploring:
            self.processed_obs.append({
                "location": None if percept.location is None
                else percept.location.tolist(),
                "features": dict(percept.non_morphological_features),
                "confidence": percept.confidence,
                "pass_message": percept.pass_message,
            })
        return percept

    # ------------------------------------------------------------------
    def _percept_from_image(self, img: np.ndarray, wave: np.ndarray
                            ) -> Message:
        mg = img.sum(axis=0)
        lo, hi = R.LO, R.HI
        seg = mg[lo:hi + 1]
        mean = float(seg.mean()) if seg.size else 0.0
        lag = R.first_peak(mg)

        salient = (
            lag is not None
            and mean > 0
            and mg[int(round(lag))] > SALIENCE_FLOOR * mean
        )
        if not salient:
            # Silence, or noise with no stable period: a message that
            # says so. No location, no confidence, not passed. The LM
            # never hears about sounds the ear could not vouch for.
            return Message(
                location=None,
                morphological_features={},
                non_morphological_features={},
                confidence=0.0,
                pass_message=False,
                sender_id=self.sensor_module_id,
                sender_type="SM",
                process_features_in_lm=False,
            )

        # which place on the membrane carries this period: the salience-
        # weighted centroid of per-channel energy at the winning lag
        li = int(round(lag))
        col = np.maximum(img[:, li], 0.0)
        ch_centroid = (
            float((col * np.arange(len(col))).sum() / col.sum())
            if col.sum() > 0 else float(len(col) / 2)
        )

        x = np.log2((lag / self._sample_rate) / 0.001) * LAG_SCALE
        y = ch_centroid * CH_SCALE
        location = np.array([x, y, 0.0])

        # local ridge orientation in the (lag, channel) plane, from the
        # image gradient around the salient point -- the analog of
        # curvature directions on a surface patch
        g_lag, g_ch = _local_gradient(img, ch_centroid, li)
        in_plane = np.array([g_lag, g_ch, 0.0])
        n = np.linalg.norm(in_plane)
        dir1 = in_plane / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        dir2 = np.cross([0.0, 0.0, 1.0], dir1)
        pose_vectors = np.vstack([[0.0, 0.0, 1.0], dir1, dir2])

        pitch = self._sample_rate / lag
        m = 69 + 12 * np.log2(pitch / 440.0)
        level = float(np.sqrt(np.mean(np.square(wave))))
        salience = float(mg[li] / (mean if mean > 0 else 1.0))

        # TIMBRE: the vertical of the SAI, encoded like the finger
        # encodes texture at a spot (Bryan's observation). Two design
        # rules, learned the hard way and from the other Fable's memo:
        # compress to a few PHYSICAL numbers (raw slices drown the
        # path features in noise -- same reason you don't feed a
        # compressor the raw FFT), and average over the note's
        # sustain, not the strobe instant (or some nodes wear the
        # attack and call it the horn). So: the mean NAP of this
        # step's whole chunk, sampled AT the membrane places where
        # harmonics 2..5 of the detected pitch live, as log-ratios to
        # the fundamental's place. Four numbers. Level cancels in the
        # ratio; pitch already has its own axis. The first binned-
        # column attempt failed the selftest flat (silent channels'
        # strobe-fallback floor + the AGC equalizing) -- sampling at
        # harmonic places asks only the witnesses that matter.
        if self._timbre_source == "ear":
            timbre = _timbre_harmonics_ear(self._ear.nap_mean, pitch)
        else:
            timbre = _timbre_harmonics(np.asarray(wave, float), pitch,
                                       self._sample_rate)

        return Message(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "pitch_hz": float(pitch),
                # pitch on a LOG scale, so a flat LM tolerance is the
                # same number of CENTS in every octave. The Hz feature
                # with a 15 Hz tolerance was 24 cents at C5 and 190 at
                # C3 -- semitone-blind exactly where guitars live.
                # Found by the duration-confound check, 2026-09-10.
                "pitch_semis": float(m),
                "level": level,
                "salience": salience,
                "place_centroid": ch_centroid,
                "timbre": timbre,
            },
            confidence=float(min(salience / (2 * SALIENCE_FLOOR), 1.0)),
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )


TIMBRE_HARMONICS = (2, 3, 4, 5)


def _timbre_harmonics_ear(nap_mean: np.ndarray, pitch: float
                          ) -> np.ndarray:
    """Timbre THROUGH the ear: log10 ratios of mean NAP at harmonic
    k's membrane place vs the fundamental's place. Weaker than the
    raw version (the AGC equalizes spectra -- measured 0.095
    separation vs 2.157 raw on the same tones) but it passes through
    lesions, which is mandatory for damaged-ear work."""
    from . import sai_ref as R

    def at(freq):
        i = int(np.argmin(np.abs(R.poles - freq)))
        i0, i1 = max(i - 1, 0), min(i + 2, len(nap_mean))
        return float(nap_mean[i0:i1].max())
    h1 = at(pitch)
    if h1 <= 1e-12:
        return np.zeros(len(TIMBRE_HARMONICS))
    return np.array([
        np.clip(np.log10(max(at(k * pitch), 1e-12) / h1), -2.0, 1.0)
        for k in TIMBRE_HARMONICS
    ])


def _timbre_harmonics(wave: np.ndarray, pitch: float, sr: float
                      ) -> np.ndarray:
    """log10 of (harmonic k's band energy / the fundamental's), from
    the step's RAW waveform, k = 2..5. Raw on purpose, and with
    precedent: CameraSM computes curvature from raw depth, not from a
    neural code -- the ear supplies WHERE (pitch, place), the raw
    observation supplies WHAT. The NAP-based attempt measured flat:
    the AGC equalizes spectra at full level and un-equalizes them at
    low level, which is its job and timbre's ruin. Bands are +-4%
    around k*pitch so vibrato stays inside its own harmonic; ratios
    cancel level exactly; clip [-2, 1] because a harmonic 100x down
    and an absent one are the same fact."""
    spec = np.abs(np.fft.rfft(wave))
    fbin = sr / len(wave)

    def band(freq):
        b0 = int(freq * 0.96 / fbin)
        b1 = int(freq * 1.04 / fbin) + 1
        return float(spec[max(b0, 0):min(b1, len(spec))].sum())

    h1 = band(pitch)
    if h1 <= 1e-12:
        return np.zeros(len(TIMBRE_HARMONICS))
    return np.array([
        np.clip(np.log10(max(band(k * pitch), 1e-12) / h1), -2.0, 1.0)
        for k in TIMBRE_HARMONICS
    ])


def _local_gradient(img: np.ndarray, ch: float, lag_i: int
                    ) -> tuple[float, float]:
    """Central-difference gradient of the image at the salient point,
    averaged over a small window so one noisy cell cannot set the
    orientation of the percept."""
    c = int(round(ch))
    c0, c1 = max(c - 2, 1), min(c + 2, img.shape[0] - 2)
    l0, l1 = max(lag_i - 2, 1), min(lag_i + 2, img.shape[1] - 2)
    patch = img[c0 - 1:c1 + 2, l0 - 1:l1 + 2]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 1.0, 0.0
    g_ch = float((patch[2:, 1:-1] - patch[:-2, 1:-1]).mean())
    g_lag = float((patch[1:-1, 2:] - patch[1:-1, :-2]).mean())
    return g_lag, g_ch
