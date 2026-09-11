"""One blind escape: reflex policy, logging (pose, echo ranges) at
every step. Sim pose is exact -- the map error we measure is the
floor; a robot adds IMU drift on top."""
import numpy as np
from audiomonty.acoustics import AcousticScene, SoundSource, Reflector
from audiomonty.voices import BatCall
from audiomonty.ear import Ear
from audiomonty.sonar import echo_ranges, gated_marginal

SR = 44100
STEP, TURN, NEAR = 0.22, np.deg2rad(38), 0.55

def build_scene():
    scene = AcousticScene(SR)
    for y in np.arange(-2.0, 2.01, 0.25):
        if 0.2 <= y <= 1.4:
            continue
        scene.add_reflector(Reflector(f"p{y:.2f}", np.array([3.0, y, 0.0])))
    src = SoundSource("call", np.zeros(3), BatCall(), level=0.6,
                      self_atten=0.1)
    scene.add(src)
    return scene, src

def escape(start, seed, max_steps=70):
    scene, src = build_scene()
    ear = Ear(SR)
    pos = np.array(start, dtype=float)
    heading = 0.0
    log = []            # (x, y, [ranges])
    path = [pos.copy()]
    for step in range(max_steps):
        src.position = np.array([pos[0], pos[1], 0.0])
        fr = ear.hear(scene.render(np.array([pos[0], pos[1], 0.0]),
                                   int(0.25*SR)))
        e = echo_ranges(gated_marginal(fr))
        ranges = [t["range_m"] for t in e if 0.5 <= t["range_m"] <= 5.0]
        log.append((pos[0], pos[1], ranges))
        nearest = min((t["range_m"] for t in e), default=99.0)
        if nearest < NEAR:
            heading += TURN
        else:
            heading *= 0.8
        pos = pos + STEP*np.array([np.cos(heading), np.sin(heading)])
        path.append(pos.copy())
        if pos[0] > 3.6 or abs(pos[1]) > 3.4:
            break
    return log, np.array(path)
