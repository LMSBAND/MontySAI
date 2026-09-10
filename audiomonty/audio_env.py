"""A sounding world: any Monty environment, wrapped so objects can sing.

Composition, not inheritance -- AudioEnvironment wraps any
SimulatedObjectEnvironment (MuJoCoSimulator today, anything tomorrow)
and adds one thing: an acoustic scene whose sources are attached to
objects, rendered at the listening agent's pose every step and
delivered as an extra sensor observation.

The inner simulator never knows. Monty's Observations type is a plain
nested dict at runtime, so the audio arrives exactly the way camera
patches do: observations[agent_id]["audio"] = {"waveform": ...}.
The waveform is the RAW observation -- the cochlea and the SAI live in
the sensor module (the ear is a sensor, not a fact about the world).

Time: each step renders `step_seconds` of audio at the post-action
pose. The scene's clock is continuous across steps, so the ear
downstream hears one unbroken world, not a pile of snippets. This is
the discrete-step approximation: pose changes are staircase, not
glide. Fine for learning; revisit if doppler ever matters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .acoustics import AcousticScene, SoundSource

AUDIO_SENSOR_ID = "audio"


class AudioEnvironment:
    """Wraps a SimulatedObjectEnvironment; objects named in `voices`
    emit sound from wherever they are placed."""

    def __init__(
        self,
        inner,
        voices: dict[str, Callable],
        listener_agent_id: str,
        sample_rate: float = 44100.0,
        step_seconds: float = 0.25,
    ) -> None:
        """Args:
            inner: the wrapped environment (e.g. MuJoCoSimulator).
            voices: object name -> waveform generator (t array -> samples).
                Objects not named here are silent, exactly like the
                visual world where most things have no voice.
            listener_agent_id: whose pose is the ear's pose.
            sample_rate: must match the ear downstream (44100).
            step_seconds: seconds of audio rendered per step.
        """
        self._inner = inner
        self._voices = dict(voices)
        self._listener = listener_agent_id
        self._scene = AcousticScene(sample_rate)
        self._n_step = int(round(step_seconds * sample_rate))

    # ---- the audio addition ------------------------------------------
    def _observe_audio(self, observations, states):
        state = states[self._listener]
        pos = np.asarray(state.position, dtype=float)
        wave = self._scene.render(pos, self._n_step)
        observations[self._listener][AUDIO_SENSOR_ID] = {
            "waveform": wave,
            "listener_position": pos,
        }
        return observations

    # ---- Environment protocol, by delegation -------------------------
    def step(self, actions):
        obs, states = self._inner.step(actions)
        return self._observe_audio(obs, states), states

    def reset(self):
        obs, states = self._inner.reset()
        return self._observe_audio(obs, states), states

    def close(self):
        self._inner.close()

    # ---- ObjectEnvironment protocol ----------------------------------
    def add_object(self, name, position=(0.0, 0.0, 0.0), *args, **kwargs):
        info = self._inner.add_object(name, position, *args, **kwargs)
        gen = self._voices.get(name)
        if gen is not None:
            self._scene.add(SoundSource(
                name=f"{name}#{info.object_id}",
                position=np.asarray(position, dtype=float),
                generator=gen,
            ))
        return info

    def remove_all_objects(self):
        self._inner.remove_all_objects()
        self._scene.remove_all()
