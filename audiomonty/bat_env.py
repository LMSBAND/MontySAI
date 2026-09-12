"""Rooms as objects: an environment where the creatures are PLACES.

The bat campaign trained a working sonar (echo_ranges), two ears
(bearing from range difference), reflex escapes, and a truth-free
map -- and then never introduced the bat to Monty. This closes that
loop. Here an "object" is a ROOM LAYOUT: add_object("the_yard")
builds a fence with a door and a scatter of posts out of Reflectors;
the agent flies a patrol through it, chirping; the observation is
what two ears hear. Episode structure, samplers, everything else is
the stock harness -- OneObjectPerEpisodeInterface cycles rooms the
way it cycles mugs.

Motion is REAL here. The other senses fabricated extent (the sound's
evolution, an internal saccade); the bat's percept stream changes
because the bat MOVES, which is the sensorimotor loop the theory is
named for. The flight is a scripted patrol (a circle inside the
room); actions from the motor policy are absorbed, as with the
NoopAgent.

The pose is honest and it is the experiment: the bat's pose rides
along in the observation (perfect dead reckoning, stated as such --
IMU drift is a queued robot problem, not a today problem). The
sensor module prints echo locations in the frame of the bat's
STARTING pose, so the same room entered at a different point yields
a rigidly transformed cloud. If Monty recognizes the room anyway,
its detected pose IS the bat's position in the room: localization
as a byproduct of recognition.
"""
from __future__ import annotations

import numpy as np

from .acoustics import AcousticScene, SoundSource
from .acoustics import Reflector
from .voices import BatCall

SONAR_SENSOR_ID = "sonar"
SR = 44100.0
BASELINE = 0.20          # ear separation, meters (the second-ear rig)
STEP_M = 0.34            # bat stride per chirp
CHUNK_S = 0.25           # listening window per chirp


def _fence(x, y0, y1, gap_lo, gap_hi, spacing=0.35):
    posts = []
    y = y0
    while y <= y1 + 1e-9:
        if not (gap_lo <= y <= gap_hi):
            posts.append((x, y))
        y += spacing
    return posts


# Room layouts, in meters. the_yard is the blind-cartography world:
# a post fence along x=3 with a door at y~0.8, plus interior posts.
ROOMS = {
    "the_yard": _fence(3.0, -1.0, 3.0, 0.65, 0.95) + [
        (1.0, -0.4), (0.6, 2.2), (2.2, 2.6),
    ],
    "the_grove": [
        (0.4, 0.4), (1.2, 2.4), (2.0, 0.2), (2.8, 1.6),
        (0.2, 1.8), (2.6, -0.6), (1.6, 1.2), (3.2, 2.8),
    ],
    # the foil: a corridor the bat never learned
    "the_hall": _fence(0.0, -1.0, 3.0, 99, 99) + _fence(2.4, -1.0, 3.0,
                                                        99, 99),
}

PATROL_CENTER = np.array([1.5, 1.0])
PATROL_RADIUS = 1.1


class BatRoomEnvironment:
    """Environment protocol by duck type (step/reset/close/add_object/
    remove_all_objects), same as AudioEnvironment. Rooms are objects;
    the patrol supplies motion; two ear waveforms plus the bat's pose
    are the observation."""

    def __init__(self, listener_agent_id: str = "agent_id_0",
                 sample_rate: float = SR,
                 start_phase_deg: float = 0.0,
                 patrol_radius: float = PATROL_RADIUS,
                 **_ignored) -> None:
        self._agent = listener_agent_id
        self._sr = float(sample_rate)
        self._phase0 = np.deg2rad(start_phase_deg)
        self._radius = patrol_radius
        self._scene = AcousticScene(self._sr)
        self._call = SoundSource("call", np.zeros(3), BatCall(),
                                 level=0.6, self_atten=0.1)
        self._scene.add(self._call)
        self._k = 0

    # ---- flight ------------------------------------------------------
    def _pose(self, k: int):
        ang = self._phase0 + 2 * np.pi * (k * STEP_M) \
            / (2 * np.pi * self._radius)
        pos = PATROL_CENTER + self._radius * np.array(
            [np.cos(ang), np.sin(ang)])
        heading = ang + np.pi / 2.0          # tangent, counterclockwise
        return pos, heading

    # ---- observation -------------------------------------------------
    def _observe(self):
        pos, heading = self._pose(self._k)
        self._call.position = np.array([pos[0], pos[1], 0.0])
        perp = np.array([-np.sin(heading), np.cos(heading)])
        pl = pos + (BASELINE / 2) * perp
        pr = pos - (BASELINE / 2) * perp
        n = int(CHUNK_S * self._sr)
        t0 = self._scene._clock
        wl = self._scene.render(np.array([pl[0], pl[1], 0.0]), n)
        self._scene._clock = t0              # two ears, one skull
        wr = self._scene.render(np.array([pr[0], pr[1], 0.0]), n)
        obs = {self._agent: {SONAR_SENSOR_ID: {
            "wave_l": wl, "wave_r": wr,
            "position": pos.copy(), "heading": float(heading),
        }}}
        states = {self._agent: _AgentState(
            position=np.array([pos[0], pos[1], 0.0]), rotation=heading)}
        return obs, states

    # ---- Environment protocol ----------------------------------------
    def step(self, actions):
        self._k += 1                          # actions absorbed; fly on
        return self._observe()

    def reset(self):
        self._k = 0
        return self._observe()

    def close(self):
        pass

    def add_object(self, name, position=(0.0, 0.0, 0.0), *a, **kw):
        posts = ROOMS[name]
        for i, (x, y) in enumerate(posts):
            self._scene.add_reflector(Reflector(
                name=f"{name}_{i}", position=np.array([x, y, 0.0]),
                reflectivity=0.9))
        return _ObjectInfo(name=name, object_id=0)

    def remove_all_objects(self):
        self._scene.reflectors.clear()
        self._scene._clock = 0                # new episode, new take
        self._k = 0


class _AgentState:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation


class _ObjectInfo:
    def __init__(self, name, object_id):
        self.name = name
        self.object_id = object_id
