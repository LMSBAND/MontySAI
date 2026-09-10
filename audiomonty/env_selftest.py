"""The wrapped MuJoCo world, heard end to end.

A real MuJoCoSimulator, a real YCB mug that hums at 147 Hz, a NoopAgent
listener. Steps produce camera AND audio observations; the ear turns
the audio into SAI frames; the tracker must read the mug's hum. Then
the same mug listened to from 4x the distance must arrive quieter.
"""
import os
from functools import partial

import numpy as np
from tbp.monty.simulators.mujoco.agents import NoopAgent, SensorConfig
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator, Resolution2D

from audiomonty.acoustics import tone
from audiomonty.audio_env import AudioEnvironment, AUDIO_SENSOR_ID
from audiomonty.ear import Ear
from audiomonty import sai_ref as R

VOICES = {"mug": tone(147.0, harmonics=(1, .5, .33, .25, .2))}

def make_env(agent_pos):
    sim = MuJoCoSimulator(
        agents=[partial(NoopAgent, agent_id="agent_id_0",
                        sensor_configs={"patch": SensorConfig(
                            resolution=Resolution2D(width=32, height=32))},
                        position=agent_pos)],
        data_path=os.path.expanduser("~/tbp/data/mujoco/objects/ycb"),
    )
    env = AudioEnvironment(sim, VOICES, "agent_id_0")
    env.add_object("mug", position=(0.0, 1.5, -0.1))
    return env

os.environ.setdefault("MUJOCO_GL", "egl")

env = make_env((0.0, 1.5, 0.4))         # half a metre from the mug
obs, states = env.step([])
assert AUDIO_SENSOR_ID in obs["agent_id_0"], "no audio observation"
assert "rgba" in obs["agent_id_0"]["patch"], "camera observation lost"
w_near = obs["agent_id_0"][AUDIO_SENSOR_ID]["waveform"]
print(f"observation keys: {list(obs['agent_id_0'].keys())}")

ear = Ear(44100.0)
frames = []
for _ in range(6):                       # 1.5 s of listening
    obs, _ = env.step([])
    frames += ear.hear(obs["agent_id_0"][AUDIO_SENSOR_ID]["waveform"])
mg = frames[-1].sum(axis=0)
f0 = 44100.0 / R.first_peak(mg)
print(f"ear hears the mug at {f0:.1f} Hz (planted 147.0)",
      "OK" if abs(f0 - 147.0) < 3 else "FAIL")
env.close()

far = make_env((0.0, 1.5, 1.9))          # 2 m away
obs, _ = far.step([])
w_far = obs["agent_id_0"][AUDIO_SENSOR_ID]["waveform"]
rn, rf = np.sqrt((w_near**2).mean()), np.sqrt((w_far**2).mean())
print(f"rms at 0.5 m {rn:.4f}, at 2 m {rf:.4f}",
      "OK" if rf < rn * 0.5 else "FAIL")
far.close()
print("WORLD SINGS")
