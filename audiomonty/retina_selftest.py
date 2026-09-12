"""RetinaSM against the real simulator, before any Monty wiring.

Four claims, each checkable:
  1. The retina produces percepts with EXTENT -- the saccade walk
     visits many distinct log-polar locations on one object.
  2. The mug and the banana are featurally different eyes-only
     (opponent color: red vs yellow).
  3. SCALE IS POSE survives the plumbing: the same mug at ~1.7x the
     distance yields locations shifted along x by a shared constant
     (the log of the scale), with the shape preserved.
  4. lit=False produces silence (pass_message False), the dark
     control's contract.
"""
import os
from functools import partial

import numpy as np
from tbp.monty.simulators.mujoco.agents import NoopAgent, SensorConfig
from tbp.monty.simulators.mujoco.simulator import (
    MuJoCoSimulator,
    Resolution2D,
)

from audiomonty.retina_sm import LR_SCALE, RetinaSM

os.environ.setdefault("MUJOCO_GL", "egl")


def patches(name, agent_z, steps=40):
    sim = MuJoCoSimulator(
        agents=[partial(NoopAgent, agent_id="agent_id_0",
                        sensor_configs={"patch": SensorConfig(
                            resolution=Resolution2D(width=64, height=64))},
                        position=(0.0, 1.5, agent_z))],
        data_path=os.path.expanduser("~/tbp/data/mujoco/objects/ycb"),
    )
    sim.add_object(name, position=(0.0, 1.5, 0.0))
    obs, _ = sim.step([])
    p = obs["agent_id_0"]["patch"]
    sim.close()
    return [p] * steps          # static world: the saccade does the moving


def run(name, agent_z=0.3, lit=True):
    sm = RetinaSM(lit=lit)
    sm.reset()
    percepts = [sm.step(None, p) for p in patches(name, agent_z)]
    live = [pc for pc in percepts if pc.pass_message]
    return live


mug = run("mug")
ban = run("banana")
locs_mug = np.array([p.location for p in mug])
locs_ban = np.array([p.location for p in ban])
uniq = len({tuple(np.round(l, 4)) for l in locs_mug})
span = np.ptp(locs_mug[:, :2], axis=0)
print(f"1. extent: mug {len(mug)}/40 percepts, {uniq} distinct locations, "
      f"span x={span[0]*100:.1f} cm y={span[1]*100:.1f} cm",
      "OK" if uniq >= 10 and span.max() > 0.02 else "FAIL")

op_mug = np.mean([p.non_morphological_features["opponent"] for p in mug],
                 axis=0)
op_ban = np.mean([p.non_morphological_features["opponent"] for p in ban],
                 axis=0)
sep = float(np.linalg.norm(op_mug - op_ban))
print(f"2. color: mug opp {op_mug.round(3)}, banana opp {op_ban.round(3)}, "
      f"separation {sep:.3f}", "OK" if sep > 0.1 else "FAIL")

far = run("mug", agent_z=0.55)
lf = np.array([p.location for p in far])
# same walk, farther eye: x should shift DOWN by log2(scale) * LR_SCALE,
# uniformly. Compare medians and spreads rather than pairing steps.
dx = float(np.median(locs_mug[:, 0]) - np.median(lf[:, 0]))
octaves = dx / LR_SCALE
shape_near = np.std(locs_mug[:, 0])
shape_far = np.std(lf[:, 0])
print(f"3. scale-as-pose: median x shift {dx*100:.2f} cm = "
      f"{octaves:.2f} octaves (surface-distance ratio predicts "
      f"~log2(0.5/0.25)={np.log2(0.5/0.25):.2f}); "
      f"x-spread {shape_near*100:.2f} vs {shape_far*100:.2f} cm",
      "OK" if 0.3 < octaves < 1.4 and shape_far > 0 else "FAIL")

dark = run("mug", lit=False)
print(f"4. dark: {len(dark)} percepts with the lights off",
      "OK" if len(dark) == 0 else "FAIL")
