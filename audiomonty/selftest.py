"""Does the standalone pipeline hear what we planted?

A 147 Hz drone at one place, a chirping cartoon bird at another.
Walk the listener from the drone to the bird and check three things:
  1. streaming == one-shot (chunking must not change the arithmetic
     for the acoustic layer feeding one continuous ear)
  2. near the drone, the SAI marginal peaks at the drone's period
  3. the level trades places as the listener walks
"""
import numpy as np
from audiomonty.acoustics import AcousticScene, SoundSource, tone, chirp_call
from audiomonty.ear import Ear
from audiomonty import sai_ref as R

SR = 44100
rng = np.random.default_rng(3)

drone = SoundSource("drone", np.array([0.0, 0.0, 0.0]),
                    tone(147.0, harmonics=(1, .5, .33, .25, .2)))
bird = SoundSource("bird", np.array([8.0, 0.0, 0.0]),
                   chirp_call(2000, 4000, period_s=0.5))

# 1 -- determinism / stream-equals-batch at the scene level
sceneA = AcousticScene(SR, [drone, bird])
sceneB = AcousticScene(SR, [drone, bird])
pos = np.array([1.0, 0.0, 0.0])
wholeshot = sceneA.render(pos, SR)
chunks = np.concatenate([sceneB.render(pos, 4410) for _ in range(10)])
err = np.abs(wholeshot - chunks).max()
print(f"scene stream==batch: max diff {err:.2e}",
      "OK" if err == 0 else "FAIL")

# 2 -- stand near the drone, listen for 1.5 s, read the marginal
ear = Ear(SR)
frames = []
for _ in range(15):
    frames += ear.hear(sceneA.render(pos, 4410))
mg = frames[-1].sum(axis=0)
lag = R.first_peak(mg)
f0 = SR / lag
print(f"near drone: tracker reads {f0:.1f} Hz (planted 147.0)",
      "OK" if abs(f0 - 147.0) < 3 else "FAIL")

# 3 -- walk to the bird; drone level in the waveform must fall
near_drone = sceneA.render(np.array([0.5, 0, 0]), 4410)
near_bird = sceneA.render(np.array([7.5, 0, 0]), 4410)
print(f"rms near drone {np.sqrt((near_drone**2).mean()):.4f}  "
      f"near bird {np.sqrt((near_bird**2).mean()):.4f}")
