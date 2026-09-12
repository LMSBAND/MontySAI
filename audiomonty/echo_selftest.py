"""EchoSM + BatRoomEnvironment against real acoustics, pre-Monty.

Claims:
  1. EXTENT + ACCURACY: a 40-chirp patrol of the_yard yields many
     percepts, and unscaling them back to world meters puts the
     median echo within ~0.35 m of a true post (sonar's measured
     error is 3.3 cm; the bearing arm and post spacing dominate).
  2. ROOMS DIFFER: the_yard and the_grove clouds differ.
  3. POSE IS ENTRY: the same yard entered at start_phase 120 deg
     yields the same cloud under one rigid transform -- verified by
     aligning with the KNOWN transform and measuring residual.
  4. DEAF: no percepts.
"""
import numpy as np

from audiomonty.bat_env import BatRoomEnvironment, ROOMS, SONAR_SENSOR_ID
from audiomonty.echo_sm import ROOM_SCALE, EchoSM


def patrol(room, phase=0.0, steps=40, deaf=False):
    env = BatRoomEnvironment(start_phase_deg=phase)
    env.add_object(room)
    sm = EchoSM(deaf=deaf)
    sm.reset()
    obs, _ = env.reset()
    locs = []
    for _ in range(steps):
        pc = sm.step(None, obs["agent_id_0"][SONAR_SENSOR_ID])
        if pc.pass_message:
            locs.append(pc.location.copy())
        obs, _ = env.step([])
    return np.array(locs) if locs else np.zeros((0, 3))


yard = patrol("the_yard")
grove = patrol("the_grove")

# 1: unscale to world, nearest true post distance
posts = np.array(ROOMS["the_yard"])
world = yard[:, :2] / ROOM_SCALE            # ego frame == world frame at
                                            # phase 0 up to origin shift
from audiomonty.bat_env import PATROL_CENTER, PATROL_RADIUS
p0 = PATROL_CENTER + PATROL_RADIUS * np.array([1.0, 0.0])
h0 = np.pi / 2
c, s = np.cos(h0), np.sin(h0)
world = np.stack([c * world[:, 0] - s * world[:, 1],
                  s * world[:, 0] + c * world[:, 1]], axis=1) + p0
dists = [np.min(np.linalg.norm(posts - w, axis=1)) for w in world]
print(f"1. extent: {len(yard)}/40 percepts; median nearest-post "
      f"error {np.median(dists):.2f} m",
      "OK" if len(yard) >= 20 and np.median(dists) < 0.35 else "FAIL")

# 2: room separability (cloud centroid + spread differ)
def sig(a):
    return np.concatenate([a[:, :2].mean(0), a[:, :2].std(0)])
sep = float(np.linalg.norm(sig(yard) - sig(grove)))
print(f"2. rooms differ: signature distance {sep:.3f}",
      "OK" if sep > 0.01 else "FAIL")

# 3: entry-at-120deg = rigid transform of the phase-0 cloud
yard2 = patrol("the_yard", phase=120.0)
ang = np.deg2rad(120.0)
p0b = PATROL_CENTER + PATROL_RADIUS * np.array([np.cos(ang), np.sin(ang)])
h0b = ang + np.pi / 2
def to_world(a, p0_, h0_):
    w = a[:, :2] / ROOM_SCALE
    c_, s_ = np.cos(h0_), np.sin(h0_)
    return np.stack([c_ * w[:, 0] - s_ * w[:, 1],
                     s_ * w[:, 0] + c_ * w[:, 1]], axis=1) + p0_
wa, wb = to_world(yard, p0, h0), to_world(yard2, p0b, h0b)
res = [np.min(np.linalg.norm(wa - b, axis=1)) for b in wb]
print(f"3. pose is entry: {len(yard2)} percepts from 120°, residual "
      f"to phase-0 cloud after the KNOWN transform: median "
      f"{np.median(res):.2f} m",
      "OK" if len(yard2) >= 20 and np.median(res) < 0.4 else "FAIL")

deaf = patrol("the_yard", deaf=True)
print(f"4. deaf: {len(deaf)} percepts", "OK" if len(deaf) == 0 else "FAIL")
