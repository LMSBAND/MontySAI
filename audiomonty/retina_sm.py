"""RetinaSM: the log-polar eye as a Monty sensor module.

The AudioSM's move, replayed for photons. The SAI gave sound a frame
(log-lag x place) where pitch ratios are translations, and "key is
pose" fell out. The retina's version is log-polar (Test 1 for eyes,
experiments/logpolar_scale.py): remap the image about the fixation
point into (theta, log r), and SCALE IS TRANSLATION along log r,
rotation is translation along theta. So the percept's location lives
in RETINAL log-polar space:

    x = log2(r / r0) * LR_SCALE      (octaves of eccentricity)
    y = theta * TH_SCALE             (angle about the fovea)
    z = 0

An object seen twice as large is the same graph slid along x; an
object rotated in the image plane is the same graph slid along y.
Monty matches on relative displacements, so both come out as pose --
"scale is pose", the visual twin of "key is pose". (Known caveat:
theta wraps at +-pi and the printed frame does not; a graph straddling
the seam tears. The SAI never had a seam. Lived with, for now.)

WHAT MOVEMENT MEANS here: the saccade is internal. The scene is
static and the agent sits still, so extent cannot come from the
world's evolution the way a siren draws its path. Instead the fovea
fixates the object (its pixel centroid -- pointing the eye, Bryan's
head-pointing gesture in miniature) and each step visits the most
salient log-polar cell not recently visited (inhibition of return,
reset per episode). The eye walks the object's contour; the walk is
the saccade sequence; the graph is the shape.

Features are deliberately few and physical, per the timbre lesson
(compress to numbers a receptor could report, don't ship raw slices):
opponent color (r-g, b-y, level-normalized), shade (value), and edge
orientation energy at the fixated cell.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)

RETINA_SENSOR_ID = "patch"

# THE MAGNIFICATION OF RETINAL SPACE. Same obligation as auditory
# space (audio_sm.py): Monty's constants think in YCB millimeters, so
# the retinal frame is printed at a scale where one object's contour
# walk spans centimeters. An on-object radius range of ~2 octaves at
# 5 cm/octave is a 10 cm graph; a full turn of theta at 2 cm/radian
# is ~12.6 cm around.
LR_SCALE = 0.05        # meters per octave of eccentricity
TH_SCALE = 0.02        # meters per radian about the fovea
R0_PX = 1.5            # eccentricity reference (px) -- the log's zero
LP_SIZE = 32           # log-polar raster: LP_SIZE thetas x LP_SIZE radii
MAX_RADIUS = 32.0      # FIXED across scales, or scale-as-pose is a lie
ON_OBJECT_DEPTH = 1.0  # MuJoCo far plane reads ~800; the desk is <1 m
IOR_DECAY = 0.92       # inhibition of return: visited cells recover slowly


class RetinaSM:
    """SensorModule turning camera-patch observations (rgba + depth)
    into log-polar percepts. Same duck-typed contract as AudioSM:
    step / update_state / reset / state_dict."""

    def __init__(
        self,
        sensor_module_id: str = RETINA_SENSOR_ID,
        lit: bool = True,
        save_raw_obs: bool = False,
    ) -> None:
        """lit=False is the DARK control: the retina reports nothing,
        honestly (a Message with pass_message=False), and whatever
        recognition survives belongs to the ear alone."""
        self.sensor_module_id = sensor_module_id
        self.features = ["opponent", "shade", "edge_energy"]
        self.lit = lit
        self.save_raw_obs = save_raw_obs
        self.state: SensorState | None = None
        self.is_exploring = False
        self.processed_obs: list[dict[str, Any]] = []
        self._visited = np.zeros((LP_SIZE, LP_SIZE))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.processed_obs = []
        self.is_exploring = False
        self._visited = np.zeros((LP_SIZE, LP_SIZE))

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
        percept = self._percept_from_patch(observation)
        if motor_only_step:
            percept.pass_message = False
        if not self.is_exploring:
            self.processed_obs.append({
                "location": None if percept.location is None
                else percept.location.tolist(),
                "features": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in percept.non_morphological_features.items()
                },
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

    def _percept_from_patch(self, observation) -> Message:
        if not self.lit:
            return self._no_percept()
        rgba = np.asarray(observation["rgba"], dtype=np.float64)
        depth = np.asarray(observation["depth"], dtype=np.float64)
        on = depth < ON_OBJECT_DEPTH
        if on.sum() < 4:
            return self._no_percept()      # nothing near enough to be a thing

        # FIXATE: fovea to the object's pixel centroid. This is the
        # pointing gesture -- the same act that aims the head aims the
        # log-polar origin, and everything downstream is relative to it.
        ys, xs = np.nonzero(on)
        fovea = (float(xs.mean()), float(ys.mean()))

        value = rgba[..., :3].mean(axis=2) / 255.0
        flags = cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR
        lp_val = cv2.warpPolar(value, (LP_SIZE, LP_SIZE), fovea,
                               MAX_RADIUS, flags)
        lp_on = cv2.warpPolar(on.astype(np.float64), (LP_SIZE, LP_SIZE),
                              fovea, MAX_RADIUS, flags)
        lp_rgb = cv2.warpPolar(rgba[..., :3] / 255.0, (LP_SIZE, LP_SIZE),
                               fovea, MAX_RADIUS, flags)
        # warpPolar output rows = theta, cols = log r

        # SALIENCE: edge energy, weighted toward cells that are at
        # least half on the object -- the contour, mostly.
        gy, gx = np.gradient(lp_val)
        edge = np.hypot(gx, gy) * (lp_on > 0.25)
        if edge.max() <= 1e-9:
            return self._no_percept()      # a featureless blob is not news

        # SACCADE: most salient cell we have not just visited.
        self._visited *= IOR_DECAY
        score = edge * (1.0 - np.clip(self._visited, 0.0, 0.95))
        ti, ri = np.unravel_index(int(np.argmax(score)), score.shape)
        self._visited[max(ti - 1, 0):ti + 2, max(ri - 1, 0):ri + 2] = 1.0

        # r=0 column is the fixation blob (logpolar_scale.py's bug #1);
        # the log needs a reference anyway, so eccentricity is measured
        # in octaves above R0_PX and the innermost cells rarely win
        # (their edge energy is fovea noise, masked by lp_on).
        r_px = MAX_RADIUS ** ((ri + 0.5) / LP_SIZE)
        theta = (ti + 0.5) / LP_SIZE * 2.0 * np.pi - np.pi
        x = np.log2(max(r_px, R0_PX) / R0_PX) * LR_SCALE
        y = theta * TH_SCALE
        location = np.array([x, y, 0.0])

        # pose: normal out of the retinal plane, edge tangent in it
        g = np.array([gx[ti, ri], gy[ti, ri], 0.0])
        n = np.linalg.norm(g)
        dir1 = g / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        dir2 = np.cross([0.0, 0.0, 1.0], dir1)
        pose_vectors = np.vstack([[0.0, 0.0, 1.0], dir1, dir2])

        r_, g_, b_ = lp_rgb[ti, ri]
        lum = max(r_ + g_ + b_, 1e-9)
        opponent = np.array([(r_ - g_) / lum,
                             (b_ - 0.5 * (r_ + g_)) / lum])
        shade = float(lp_val[ti, ri])
        edge_energy = float(edge[ti, ri] / edge.max())

        return Message(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "opponent": opponent,
                "shade": shade,
                "edge_energy": edge_energy,
            },
            confidence=1.0,
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )
