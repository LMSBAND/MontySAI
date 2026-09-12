"""CamSM: Monty's own eye, transplanted onto our skull -- the control
arm of the eye-vs-eye comparison.

This is a faithful stand-in for the reference CameraSM
(sensor_modules.py: features at 3D locations from an RGBD patch --
surface normal + curvature poses, HSV color), given exactly the same
privileges our RetinaSM gets on the static NoopAgent: the same 64x64
patch, and an internal saccade (the reference eye gets its extent
from a MOVING sensor; on a frozen skull it would collapse to a point
object, which would be sabotage, not comparison).

The ONLY difference that remains is the coordinate choice:

    CamSM   location = the fixated point's true 3D position (pinhole,
            meters) -- Monty's native vision-as-touch frame.
    RetinaSM location = (log-r, theta) about the fovea -- scale and
            image rotation become translations.

Pre-registered prediction (2026-09-12): their eye SURVIVES image
rotation (the evidence LM searches rigid rotations natively) but
FAILS scale (a 1.5x mug's 3D graph is 1.5x bigger; displacements and
max_match_distance live in meters and there is no scale hypothesis
to save it). Ours converts both to translation and survives both.
Either outcome is a sentence worth having.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import rgb_to_hsv

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)

CAM_SENSOR_ID = "patch"
NEIGH = 2
FOOT = 1
MAX_RANGE_M = 1.0       # anything nearer than a meter is scene, not sky
MIN_PX = 12


class CamSM:
    """The reference eye's percept recipe on the shared saccade walk:
    3D metric locations, normal/curvature pose, HSV color feature."""

    def __init__(
        self,
        sensor_module_id: str = CAM_SENSOR_ID,
        lit: bool = True,
        save_raw_obs: bool = False,
    ) -> None:
        self.sensor_module_id = sensor_module_id
        self.features = ["hsv", "curvature"]
        self.lit = lit
        self.save_raw_obs = save_raw_obs
        self.state: SensorState | None = None
        self.is_exploring = False
        self.processed_obs: list[dict[str, Any]] = []
        self._visited: set[tuple[int, int]] = set()
        self._last_uv: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.processed_obs = []
        self.is_exploring = False
        self._visited = set()
        self._last_uv = None

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
                "confidence": percept.confidence,
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

    def _percept(self, observation) -> Message:
        if not self.lit:
            return self._no_percept()
        depth = np.asarray(observation["depth"], dtype=np.float64)
        rgba = np.asarray(observation["rgba"], dtype=np.float64)
        h, w = depth.shape
        on = depth < MAX_RANGE_M
        on[:NEIGH, :] = on[-NEIGH:, :] = False
        on[:, :NEIGH] = on[:, -NEIGH:] = False
        if on.sum() < MIN_PX:
            return self._no_percept()

        cand = {(int(v), int(u)) for v, u in zip(*np.nonzero(on))}
        cand -= self._visited
        if not cand:
            self._visited = set()
            cand = {(int(v), int(u)) for v, u in zip(*np.nonzero(on))}
        if self._last_uv is None:
            vc, uc = min(cand, key=lambda p: depth[p[0], p[1]])
        else:
            lv, lu = self._last_uv
            vc, uc = min(cand, key=lambda p: (p[0] - lv) ** 2
                         + (p[1] - lu) ** 2)
        self._last_uv = (vc, uc)
        for dv in range(-FOOT, FOOT + 1):
            for du in range(-FOOT, FOOT + 1):
                self._visited.add((vc + dv, uc + du))

        f = w / 2.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        vs, us = np.mgrid[vc - NEIGH:vc + NEIGH + 1,
                          uc - NEIGH:uc + NEIGH + 1]
        ds = depth[vc - NEIGH:vc + NEIGH + 1, uc - NEIGH:uc + NEIGH + 1]
        near = ds < MAX_RANGE_M
        if near.sum() < 6:
            return self._no_percept()
        px = (us[near] - cx) / f * ds[near]
        py = (vs[near] - cy) / f * ds[near]
        pts = np.column_stack([px, py, ds[near]])
        contact = pts.mean(axis=0)

        centered = pts - contact
        cov = centered.T @ centered / len(pts)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        if normal[2] > 0:
            normal = -normal
        curvature = float(np.log10(max(
            evals[0] / max(evals.sum(), 1e-18), 1e-8)))
        pose_vectors = np.vstack([normal, evecs[:, 2],
                                  np.cross(normal, evecs[:, 2])])

        hsv = rgb_to_hsv(rgba[vc, uc, :3] / 255.0)

        return Message(
            location=np.array(contact),
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": np.asarray(hsv, dtype=float),
                "curvature": curvature,
            },
            confidence=1.0,
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )
