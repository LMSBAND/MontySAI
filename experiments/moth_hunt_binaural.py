import numpy as np
from audiomonty.acoustics import AcousticScene, SoundSource, Reflector
from audiomonty.voices import BatCall
from audiomonty.binaural import BinauralBat

SR = 44100; CATCH = 0.70

def hunt_binaural(moth_speed, flee=True, max_steps=140, seed=0,
                  save=None):
    r = np.random.default_rng(seed)
    scene = AcousticScene(SR)
    src = SoundSource("call", np.zeros(3), BatCall(), level=0.6,
                      self_atten=0.1)
    scene.add(src)
    moth = Reflector("moth", np.array([2.5, 1.5, 0.0]), reflectivity=0.9)
    scene.add_reflector(moth)
    bat_head = BinauralBat()
    bat = np.array([0.0, 0.0]); heading = 0.0
    bpath, mpath = [bat.copy()], [moth.position[:2].copy()]
    for step in range(max_steps):
        # the mouth sits at the bat; render from there for the moth's echo
        src.position = np.array([bat[0], bat[1], 0.0])
        rng, brg = bat_head.listen(scene, bat, heading)
        if float(np.linalg.norm(moth.position[:2]-bat)) < CATCH:
            if save:
                np.save(f"{save}_bat.npy", np.array(bpath))
                np.save(f"{save}_moth.npy", np.array(mpath))
            return np.array(bpath), np.array(mpath), step, True
        if brg is None:
            heading += r.uniform(0.6, 2.0)*r.choice([-1, 1])  # lost: search
        else:
            heading += np.clip(brg, -0.6, 0.6)               # STEER to bearing
        step_len = 0.34 if (rng is not None and rng < 1.6) else 0.22
        bat = bat + step_len*np.array([np.cos(heading), np.sin(heading)])
        m = moth.position[:2]
        away = (m - bat); away = away/np.linalg.norm(away)
        ang = np.arctan2(away[1], away[0]) + r.normal(0, 0.6)
        m = m + moth_speed*np.array([np.cos(ang), np.sin(ang)])
        moth.position = np.array([m[0], m[1], 0.0])
        bpath.append(bat.copy()); mpath.append(m.copy())
    if save:
        np.save(f"{save}_bat.npy", np.array(bpath))
        np.save(f"{save}_moth.npy", np.array(mpath))
    return np.array(bpath), np.array(mpath), max_steps, False
