"""TouchSM: the fingertip as a Monty sensor module.

Touch is the sense Monty was born with -- the reference CameraSM is
vision-as-touch, a fingertip that happens to use photons. This module
is the honest inverse: a fingertip that REFUSES photons. It reads
only the depth map (contact geometry), so it works in absolute
darkness and carries no color; and it only works INSIDE ARM'S REACH.

The reach limit is not decoration. The real-sonar takes measured the
ear's near-field confound (its own click owns everything under
~0.5 m), and the eye needs light. Touch is the modality that owns
the donut hole: REACH = 0.45 m, and beyond it the finger reports
nothing, honestly. Ear far, eye lit, touch near -- three failure
modes, three home turfs.

THE TOUCH LOCATION. Unlike the ear (log-lag octaves) and the eye
(log-polar about the fovea), touch needs no invented frame: contact
points live in the skull's own metric 3D space, via the pinhole
model (fovh 90 deg -> f = W/2 pixels). A mug's handle is 4 cm from
its body because the finger MEASURED 4 cm. This is Monty's native
YCB scale; no magnification constant required.

THE PALPATION WALK. Each step the finger touches the on-object,
in-reach point nearest its previous contact that it has not already
visited (inhibition of return per episode) -- a stroke, not a
teleport. The stroke's path is the object's surface; the graph is
the shape, in true meters.

FEATURES. One, physical: curvature (local surface variation from
the PCA of the 3D contact neighborhood, log-scaled). Flat cylinder
wall, tight handle bend, banana horn -- the fingertip's entire
vocabulary, and it is enough because the graph carries the shape.
Pose: the surface normal (PCA minor axis, flipped toward the eye)
with tangent directions, pose_fully_defined False.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)

TOUCH_SENSOR_ID = "touch"
REACH_M = 0.45          # arm's length: beyond this the finger finds air
NEIGH = 2               # contact neighborhood half-width (5x5 pixels)
FOOT = 1                # visited footprint half-width (3x3): a thin
                        # object seen edge-on is only a few pixels wide,
                        # and a 5x5 footprint exhausted the banana in
                        # three touches (measured: 3 distinct contacts
                        # from 60 steps). The PCA neighborhood stays 5x5.
MIN_CONTACT_PX = 12     # fewer on-object pixels than this = nothing there


class TouchSM:
    """SensorModule turning depth observations into contact percepts.
    Same duck-typed contract as AudioSM / RetinaSM."""

    def __init__(
        self,
        sensor_module_id: str = TOUCH_SENSOR_ID,
        numb: bool = False,
        save_raw_obs: bool = False,
    ) -> None:
        """numb=True is touch's dark-equivalent control: the finger
        reports nothing, and whatever recognition survives belongs
        to the other senses."""
        self.sensor_module_id = sensor_module_id
        self.features = ["curvature"]
        self.numb = numb
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
        percept = self._percept_from_depth(observation)
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

    def _percept_from_depth(self, observation) -> Message:
        if self.numb:
            return self._no_percept()
        depth = np.asarray(observation["depth"], dtype=np.float64)
        h, w = depth.shape
        on = depth < REACH_M                 # in reach AND on something
        # keep the border clear so every contact has a full neighborhood
        on[:NEIGH, :] = on[-NEIGH:, :] = False
        on[:, :NEIGH] = on[:, -NEIGH:] = False
        if on.sum() < MIN_CONTACT_PX:
            return self._no_percept()        # the finger finds air

        cand = {(int(v), int(u)) for v, u in zip(*np.nonzero(on))}
        cand -= self._visited
        if not cand:                          # surface exhausted: lift off,
            self._visited = set()             # start a fresh pass
            cand = {(int(v), int(u)) for v, u in zip(*np.nonzero(on))}
        if self._last_uv is None:
            vc, uc = min(cand,               # first contact: nearest point
                         key=lambda p: depth[p[0], p[1]])
        else:
            lv, lu = self._last_uv
            vc, uc = min(cand, key=lambda p: (p[0] - lv) ** 2
                         + (p[1] - lu) ** 2)
        self._last_uv = (vc, uc)
        # a fingertip is not a pixel: mark its footprint visited
        for dv in range(-FOOT, FOOT + 1):
            for du in range(-FOOT, FOOT + 1):
                self._visited.add((vc + dv, uc + du))

        # pinhole: fovh 90 deg -> f = w/2; depth is planar z
        f = w / 2.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        vs, us = np.mgrid[vc - NEIGH:vc + NEIGH + 1,
                          uc - NEIGH:uc + NEIGH + 1]
        ds = depth[vc - NEIGH:vc + NEIGH + 1, uc - NEIGH:uc + NEIGH + 1]
        near = ds < REACH_M
        if near.sum() < 6:
            return self._no_percept()        # an edge sliver, not a surface
        px = (us[near] - cx) / f * ds[near]
        py = (vs[near] - cy) / f * ds[near]
        pts = np.column_stack([px, py, ds[near]])
        contact = pts.mean(axis=0)

        # PCA of the contact neighborhood: normal = minor axis,
        # curvature = surface variation (Pauly's lambda_min / sum)
        centered = pts - contact
        cov = centered.T @ centered / len(pts)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        if normal[2] > 0:                    # flip toward the eye (-z ray)
            normal = -normal
        variation = float(evals[0] / max(evals.sum(), 1e-18))
        curvature = float(np.log10(max(variation, 1e-8)))

        dir1 = evecs[:, 2]
        dir2 = np.cross(normal, dir1)
        pose_vectors = np.vstack([normal, dir1, dir2])

        return Message(
            location=np.array(contact),
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "curvature": curvature,
            },
            confidence=1.0,
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )
