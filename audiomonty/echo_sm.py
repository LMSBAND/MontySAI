"""EchoSM: echolocation as a Monty sensor module.

The fourth sense, same recipe. Two independent CARFAC+SAI ears hear
one chirp; ranges come from echo_ranges on each ear's gated image;
bearing comes from the range difference (the second-ear result,
verified to ~1 degree); and the percept's location is the echoing
thing's position in the frame of the bat's STARTING pose:

    location = R(-h0) . (p_world - p0) * ROOM_SCALE

Ego-start coordinates are the honest choice and the experiment
itself. In world coordinates a room would present identical clouds
from any entry point and recognition would be trivial. In ego-start
coordinates a different entry presents the same room rigidly
rotated and translated -- exactly the transformation the evidence
LM's pose search exists to solve. If it recognizes the room, its
detected pose is the bat's situation in it: localization as a
byproduct of recognition, which is the thousand-brains claim
stated at the scale of architecture.

Dead reckoning is perfect here and declared: the pose rides in the
observation. Drift is a robot's problem, queued, not smuggled.

ROOM_SCALE prints meters of room at 5 cm per meter, so a 5 m room
is a 25 cm graph -- inside the YCB-tuned constants (1 mm dedup =
2 cm of world; 1 cm match distance = 20 cm of world, a fair sonar
tolerance against the measured 3.3 cm worst-case ranging error).

One feature: echo strength above the scene's own floor (the MAD
z-score echo_ranges already computes), log-scaled. Identity is
carried by the geometry, as with touch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)

from .bat_env import BASELINE, SONAR_SENSOR_ID
from .ear import Ear
from .sonar import echo_ranges, gated_marginal

SR = 44100.0
ROOM_SCALE = 0.05        # meters of graph per meter of room
ARTIFACT_M = 0.55        # near-field family, measured in sim and on
                         # the SM57 takes; nothing honest lives below
HORIZON_M = 6.0
PAIR_TOL_M = 0.6         # left/right ranges must agree this well to
                         # be the same echo


class EchoSM:
    """SensorModule turning two ear waveforms into room percepts."""

    def __init__(self, sensor_module_id: str = SONAR_SENSOR_ID,
                 deaf: bool = False,
                 save_raw_obs: bool = False) -> None:
        """deaf=True is the control: no percepts, honestly."""
        self.sensor_module_id = sensor_module_id
        self.features = ["strength"]
        self.deaf = deaf
        self.save_raw_obs = save_raw_obs
        self.state: SensorState | None = None
        self.is_exploring = False
        self.processed_obs: list[dict[str, Any]] = []
        self._left = Ear(SR)
        self._right = Ear(SR)
        self._origin: np.ndarray | None = None
        self._h0: float | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._left = Ear(SR)
        self._right = Ear(SR)
        self._origin = None
        self._h0 = None
        self.processed_obs = []
        self.is_exploring = False

    def update_state(self, agent: AgentState) -> None:
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
        percept = self._percept(observation)
        if motor_only_step:
            percept.pass_message = False
        if not self.is_exploring:
            self.processed_obs.append({
                "location": None if percept.location is None
                else percept.location.tolist(),
                "features": dict(percept.non_morphological_features),
                "pass_message": percept.pass_message,
            })
        return percept

    def _no_percept(self) -> Message:
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

    def _nearest_honest(self, ear: Ear, wave: np.ndarray):
        frames = ear.hear(np.asarray(wave, dtype=np.float64))
        if not frames:
            return None
        out = [(t["range_m"], t["strength"])
               for t in echo_ranges(gated_marginal(frames))
               if ARTIFACT_M < t["range_m"] < HORIZON_M]
        return min(out, default=None)

    def _percept(self, observation) -> Message:
        if self.deaf:
            return self._no_percept()
        pos = np.asarray(observation["position"], dtype=float)
        heading = float(observation["heading"])
        if self._origin is None:              # first chirp fixes the frame
            self._origin = pos.copy()
            self._h0 = heading
        left = self._nearest_honest(self._left, observation["wave_l"])
        right = self._nearest_honest(self._right, observation["wave_r"])
        if left is None or right is None:
            return self._no_percept()         # the room said nothing usable
        (rl, sl), (rr, sr) = left, right
        if abs(rl - rr) > PAIR_TOL_M:
            return self._no_percept()         # two ears, two different
                                              # echoes: not one target
        rng = 0.5 * (rl + rr)
        s = np.clip(2.0 * (rr - rl) / BASELINE, -1.0, 1.0)
        bearing = float(np.arcsin(s))         # + = target to the left

        world = pos + rng * np.array([np.cos(heading + bearing),
                                      np.sin(heading + bearing)])
        d = world - self._origin
        c, sn = np.cos(-self._h0), np.sin(-self._h0)
        ego = np.array([c * d[0] - sn * d[1],
                        sn * d[0] + c * d[1], 0.0]) * ROOM_SCALE

        ray = np.array([np.cos(heading + bearing),
                        np.sin(heading + bearing), 0.0])
        dir1 = np.array([c * ray[0] - sn * ray[1],
                         sn * ray[0] + c * ray[1], 0.0])
        pose_vectors = np.vstack([[0.0, 0.0, 1.0], dir1,
                                  np.cross([0.0, 0.0, 1.0], dir1)])

        return Message(
            location=ego,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "strength": float(np.log10(max(0.5 * (sl + sr), 1e-3))),
            },
            confidence=1.0,
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )
