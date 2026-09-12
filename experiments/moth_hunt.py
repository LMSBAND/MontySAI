import numpy as np
from audiomonty.acoustics import AcousticScene, SoundSource, Reflector
from audiomonty.voices import BatCall
from audiomonty.ear import Ear
from audiomonty.sonar import echo_ranges, gated_marginal

SR = 44100
ARTIFACT = 0.55        # the call's own near-field family lives below this
CATCH = 0.70           # a wingbeat away, and above the artifact zone

def hunt(moth_speed, flee=False, max_steps=140, seed=0):
    r = np.random.default_rng(seed)
    scene = AcousticScene(SR)
    src = SoundSource("call", np.zeros(3), BatCall(), level=0.6,
                      self_atten=0.1)
    scene.add(src)
    moth = Reflector("moth", np.array([2.5, 1.5, 0.0]), reflectivity=0.9)
    scene.add_reflector(moth)
    ear = Ear(SR)
    bat = np.array([0.0, 0.0]); heading = 0.0
    last = None
    bpath, mpath = [bat.copy()], [moth.position[:2].copy()]
    STEP = 0.22
    for step in range(max_steps):
        src.position = np.array([bat[0], bat[1], 0.0])
        fr = ear.hear(scene.render(np.array([bat[0], bat[1], 0.0]),
                                   int(0.25*SR)))
        e = echo_ranges(gated_marginal(fr))
        targets = [t["range_m"] for t in e if t["range_m"] > ARTIFACT]
        rr = min(targets, default=None)
        true_d = float(np.linalg.norm(moth.position[:2] - bat))
        if true_d < CATCH:
            return np.array(bpath), np.array(mpath), step, True
        if rr is None or (last is not None and rr > last + 0.02):
            heading += r.uniform(0.6, 2.4) * r.choice([-1, 1])   # tumble
        last = rr
        bat = bat + STEP*np.array([np.cos(heading), np.sin(heading)])
        m = moth.position[:2]
        if flee:
            away = (m - bat); away = away/np.linalg.norm(away)
            ang = np.arctan2(away[1], away[0]) + r.normal(0, 0.6)
        else:
            ang = r.uniform(0, 2*np.pi)
        m = m + moth_speed*np.array([np.cos(ang), np.sin(ang)])
        moth.position = np.array([m[0], m[1], 0.0])
        bpath.append(bat.copy()); mpath.append(m.copy())
    return np.array(bpath), np.array(mpath), max_steps, False

def hunt_buzz(moth_speed, max_steps=140, seed=0):
    """Terminal-buzz pursuit: gentle tumbles (don't throw away your
    heading over one bad reading), sprint when the echo is close."""
    r = np.random.default_rng(seed)
    scene = AcousticScene(SR)
    src = SoundSource("call", np.zeros(3), BatCall(), level=0.6,
                      self_atten=0.1)
    scene.add(src)
    moth = Reflector("moth", np.array([2.5, 1.5, 0.0]), reflectivity=0.9)
    scene.add_reflector(moth)
    ear = Ear(SR)
    bat = np.array([0.0, 0.0]); heading = 0.0
    last = None
    bpath, mpath = [bat.copy()], [moth.position[:2].copy()]
    for step in range(max_steps):
        src.position = np.array([bat[0], bat[1], 0.0])
        fr = ear.hear(scene.render(np.array([bat[0], bat[1], 0.0]),
                                   int(0.25*SR)))
        e = echo_ranges(gated_marginal(fr))
        targets = [t["range_m"] for t in e if t["range_m"] > ARTIFACT]
        rr = min(targets, default=None)
        true_d = float(np.linalg.norm(moth.position[:2] - bat))
        if true_d < CATCH:
            return np.array(bpath), np.array(mpath), step, True
        closing = (rr is not None and last is not None and rr < last - 0.01)
        if rr is None:
            heading += r.uniform(0.6, 2.0)*r.choice([-1, 1])
        elif not closing:
            heading += r.uniform(0.35, 0.9)*r.choice([-1, 1])  # gentle
        last = rr
        step_len = 0.34 if (rr is not None and rr < 1.6) else 0.22
        bat = bat + step_len*np.array([np.cos(heading), np.sin(heading)])
        m = moth.position[:2]
        away = (m - bat); away = away/np.linalg.norm(away)
        ang = np.arctan2(away[1], away[0]) + r.normal(0, 0.6)
        m = m + moth_speed*np.array([np.cos(ang), np.sin(ang)])
        moth.position = np.array([m[0], m[1], 0.0])
        bpath.append(bat.copy()); mpath.append(m.copy())
    return np.array(bpath), np.array(mpath), max_steps, False
