"""TouchSM against the real simulator, before any Monty wiring.

Five claims:
  1. EXTENT + METRIC HONESTY: the palpation walk visits many distinct
     3D contacts, and the contact cloud is object-sized in TRUE
     meters (a YCB mug is ~8-10 cm; no magnification constant).
  2. SHAPE: mug and banana contact clouds differ in aspect
     (elongation) and curvature statistics -- the fingertip's
     vocabulary, no color anywhere.
  3. REACH: from 0.8 m the finger finds air. Nothing is reported.
  4. NUMB: the control contract -- numb=True reports nothing.
  5. NO PHOTONS: zeroing the rgba changes no percept (touch never
     reads it) -- checked by feeding an observation with rgba absent.
"""
import os
from functools import partial

import numpy as np
from tbp.monty.simulators.mujoco.agents import NoopAgent, SensorConfig
from tbp.monty.simulators.mujoco.simulator import (
    MuJoCoSimulator,
    Resolution2D,
)

from audiomonty.touch_sm import TouchSM

os.environ.setdefault("MUJOCO_GL", "egl")


def get_patch(name, agent_z=0.3):
    sim = MuJoCoSimulator(
        agents=[partial(NoopAgent, agent_id="a",
                        sensor_configs={"touch": SensorConfig(
                            resolution=Resolution2D(width=128, height=128))},
                        position=(0.0, 1.5, agent_z))],
        data_path=os.path.expanduser("~/tbp/data/mujoco/objects/ycb"),
    )
    sim.add_object(name, position=(0.0, 1.5, 0.0))
    obs, _ = sim.step([])
    p = obs["a"]["touch"]
    sim.close()
    return p


def run(name, agent_z=0.3, steps=60, numb=False, no_rgba=False):
    p = get_patch(name, agent_z)
    if no_rgba:
        p = {"depth": p["depth"]}
    sm = TouchSM(numb=numb)
    sm.reset()
    out = [sm.step(None, p) for _ in range(steps)]
    return [pc for pc in out if pc.pass_message]


mug = run("mug")
ban = run("banana")
lm = np.array([p.location for p in mug])
lb = np.array([p.location for p in ban])
span_m = np.ptp(lm, axis=0) * 100
span_b = np.ptp(lb, axis=0) * 100
uniq = len({tuple(np.round(l, 4)) for l in lm})
print(f"1. extent: mug {len(mug)}/60 contacts, {uniq} distinct, span "
      f"[{span_m[0]:.1f} {span_m[1]:.1f} {span_m[2]:.1f}] cm; contact "
      f"depth ~{np.median(lm[:,2])*100:.1f} cm",
      "OK" if uniq >= 20 and 4 < max(span_m) < 20 else "FAIL")

def aspect(l):
    ev = np.linalg.eigvalsh(np.cov((l - l.mean(0)).T))
    return float(np.sqrt(ev[-1] / max(ev[1], 1e-12)))
cm = np.median([p.non_morphological_features["curvature"] for p in mug])
cb = np.median([p.non_morphological_features["curvature"] for p in ban])
print(f"2. shape: aspect mug {aspect(lm):.2f} vs banana {aspect(lb):.2f}; "
      f"median curvature mug {cm:.2f} vs banana {cb:.2f}",
      "OK" if abs(aspect(lb) - aspect(lm)) > 0.3 or abs(cm - cb) > 0.15
      else "FAIL")

far = run("mug", agent_z=0.8)
print(f"3. reach: {len(far)} contacts from 0.8 m",
      "OK" if len(far) == 0 else "FAIL")

numb = run("mug", numb=True)
print(f"4. numb: {len(numb)} contacts while numb",
      "OK" if len(numb) == 0 else "FAIL")

blind = run("mug", no_rgba=True)
same = (len(blind) == len(mug)
        and np.allclose(np.array([p.location for p in blind]), lm))
print(f"5. no photons: rgba removed entirely -> {len(blind)} contacts, "
      f"identical locations: {same}", "OK" if same else "FAIL")
