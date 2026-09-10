"""The percept streams of two auditory objects, side by side.

A humming mug (steady 147 Hz drone) and a chirping mug (2->4 kHz bird
cartoon), each listened to for 3 seconds through the full stack:
MuJoCo world -> acoustic scene -> streaming CARFAC+SAI -> AudioSM ->
CMP Messages. If Monty is to learn these as objects, their trajectories
through auditory space must differ in exactly the way the sounds do:
the drone parks at one location; the chirp sweeps.
"""
import os
from functools import partial

import numpy as np
from tbp.monty.simulators.mujoco.agents import NoopAgent, SensorConfig
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator, Resolution2D

from audiomonty.acoustics import tone, chirp_call
from audiomonty.audio_env import AudioEnvironment, AUDIO_SENSOR_ID
from audiomonty.audio_sm import AudioSM

os.environ.setdefault("MUJOCO_GL", "egl")

def siren(f0, f1, period_s):
    """Slow harmonic glide INSIDE the tracker's range -- an auditory
    object whose identity is a trajectory, not a point."""
    import numpy as np
    def gen(t):
        u = 0.5 - 0.5 * np.cos(2 * np.pi * t / period_s)   # 0..1..0
        k = np.log(f1 / f0)
        # phase = 2*pi * integral of f0*exp(k*u(t)) dt, done numerically
        # per call is fine: t arrives as a contiguous ramp
        inst = f0 * np.exp(k * u)
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 0))
        phase = 2 * np.pi * np.cumsum(inst * dt)
        return (np.sin(phase) + 0.5 * np.sin(2 * phase)
                + 0.33 * np.sin(3 * phase)) / 1.83
    return gen

VOICES = {
    "drone": tone(147.0, harmonics=(1, .5, .33, .25, .2)),
    "bird": chirp_call(2000, 4000, period_s=0.5),
    "siren": siren(110.0, 220.0, period_s=3.0),
}

def listen(voice_name, seconds=3.0):
    sim = MuJoCoSimulator(
        agents=[partial(NoopAgent, agent_id="agent_id_0",
                        sensor_configs={"patch": SensorConfig(
                            resolution=Resolution2D(width=16, height=16))},
                        position=(0.0, 1.5, 0.5))],
        data_path=os.path.expanduser("~/tbp/data/mujoco/objects/ycb"),
    )
    env = AudioEnvironment(sim, {"mug": VOICES[voice_name]}, "agent_id_0")
    env.add_object("mug", position=(0.0, 1.5, 0.0))
    sm = AudioSM()
    percepts = []
    for _ in range(int(seconds / 0.25)):
        obs, states = env.step([])
        sm.update_state(states["agent_id_0"])
        msg = sm.step(None, obs["agent_id_0"][AUDIO_SENSOR_ID])
        if msg.pass_message:
            percepts.append(msg)
    env.close()
    return percepts

for name in ("drone", "bird", "siren"):
    ps = listen(name)
    locs = np.array([p.location for p in ps])
    pitches = [p.non_morphological_features["pitch_hz"] for p in ps]
    span = locs[:, 0].max() - locs[:, 0].min() if len(locs) else 0
    path = float(np.linalg.norm(np.diff(locs, axis=0), axis=1).sum()) \
        if len(locs) > 1 else 0
    print(f"{name:6}: {len(ps):2d} percepts  "
          f"pitch {min(pitches):7.1f}-{max(pitches):7.1f} Hz  "
          f"x-span {span*1000:5.1f} mm  path {path*1000:6.1f} mm")

print("\nOBJECTS ARE TRAJECTORIES" )
