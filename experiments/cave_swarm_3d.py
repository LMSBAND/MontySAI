"""A group threads a 3D cave: dodging stone, and each other.

Everything from the campaign at once. A cave hung with stalactites
from the ceiling and spiked with stalagmites from the floor, and a
squad of bats flying through it, each echolocating: ranging the stone
by echo, ranging its neighbours the same way. Two reflexes, both
built from distances the ears already return -- push away from stone,
push away from a neighbour, lean toward the far end.

The experiment is the second reflex. With it, the group flows through
the gaps as a loose flock and nobody touches anybody. Take it away --
each bat still dodges the rock but goes deaf to the others -- and the
bats fly through each other, because avoiding a wall and avoiding a
flockmate are the same computation done on the same echoes, and a
bat that can hear only the wall is only half a navigator.

No self/other confusion here beyond the earlier jamming result: each
bat simply hears where the others are and treats them as soft
obstacles that move. The flock is not coordinated by any signal. It
coheres because every body is, to every other body, one more thing in
the room worth not hitting.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CAVE_LEN = 12.0
Y_HALF, Z_TOP = 1.9, 3.0
N_BATS = 5
SPEED = 0.33
R_BAT = 0.16
R_COL = 0.35
SENSE_OBST = 3.0
SENSE_BAT = 2.0
GOAL_W = 1.2
OBST_W = 4.0
BAT_W = 1.9
MAX_STEPS = 300


def make_cave(seed):
    """Stalactites hang from the ceiling, stalagmites rise from the
    floor; each is a vertical segment (top, bottom) with a radius."""
    rs = np.random.default_rng(seed)
    cols = []
    x = 1.6
    while x < CAVE_LEN - 0.5:
        y = rs.uniform(-Y_HALF + 0.3, Y_HALF - 0.3)
        if rs.random() < 0.5:                          # stalactite
            bottom = rs.uniform(1.0, 1.8)
            cols.append((np.array([x, y, Z_TOP]),
                         np.array([x, y, bottom])))
        else:                                          # stalagmite
            top = rs.uniform(1.2, 2.0)
            cols.append((np.array([x, y, 0.0]),
                         np.array([x, y, top])))
        x += rs.uniform(1.25, 1.8)
    return cols


def dist_to_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest), closest


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def run(social=True, seed=0):
    rs = np.random.default_rng(seed)
    cols = make_cave(seed + 40)
    bats = np.array([[0.2, y, z] for y, z in zip(
        np.linspace(-0.8, 0.8, N_BATS),
        rs.uniform(1.0, 2.0, N_BATS))])
    hand = np.array([1.0 if i % 2 == 0 else -1.0
                     for i in range(N_BATS)])
    paths = [[b.copy()] for b in bats]
    done = np.zeros(N_BATS, bool)
    hits_stone, hits_bat = 0, 0
    counted = set()
    for step in range(MAX_STEPS):
        new = bats.copy()
        for i in range(N_BATS):
            if done[i]:
                new[i] = bats[i]
                continue
            p = bats[i]
            desired = GOAL_W * np.array([1.0, 0, 0])
            # dodge stone
            for a, b in cols:
                d, closest = dist_to_segment(p, a, b)
                if d < SENSE_OBST:
                    push = min(OBST_W / max(d - R_COL, 0.15) ** 2 * 0.1,
                               6.0)
                    desired = desired + push * unit(p - closest)
                if d < R_BAT + R_COL:
                    counted_key = ("s", i, id(a))
                    if counted_key not in counted:
                        hits_stone += 1
                        counted.add(counted_key)
            # dodge neighbours (only if social)
            if social:
                for j in range(N_BATS):
                    if j == i or done[j]:
                        continue
                    v = p - bats[j]
                    d = np.linalg.norm(v)
                    if d < SENSE_BAT:
                        away = unit(v)
                        veer = hand[i] * unit(np.cross(away, [0, 0, 1.0]))
                        desired = desired + (BAT_W * away + 1.1 * veer) / \
                            max(d, 0.2) ** 2 * 0.1
            new[i] = p + SPEED * unit(desired)
            new[i][1] = np.clip(new[i][1], -Y_HALF, Y_HALF)
            new[i][2] = np.clip(new[i][2], 0.1, Z_TOP - 0.1)
            if new[i][0] >= CAVE_LEN:
                done[i] = True
        # bat-bat collisions on the updated positions
        for i in range(N_BATS):
            for j in range(i + 1, N_BATS):
                if done[i] or done[j]:
                    continue
                if np.linalg.norm(new[i] - new[j]) < 2 * R_BAT:
                    key = ("b", i, j, step // 4)
                    if key not in counted:
                        hits_bat += 1
                        counted.add(key)
        bats = new
        for i in range(N_BATS):
            paths[i].append(bats[i].copy())
        if done.all():
            break
    return dict(cols=cols, paths=[np.array(p) for p in paths],
                hits_stone=hits_stone, hits_bat=hits_bat,
                through=int(done.sum()), steps=step + 1)


if __name__ == "__main__":
    SEEDS = range(8)
    for social, label in [(True, "hears its flock"),
                          (False, "deaf to the flock")]:
        rs = [run(social=social, seed=s) for s in SEEDS]
        thru = sum(r["through"] for r in rs)
        stone = sum(r["hits_stone"] for r in rs)
        batb = sum(r["hits_bat"] for r in rs)
        print(f"{label:20s}: {thru}/{len(SEEDS)*N_BATS} through · "
              f"stone hits {stone} · bat-bat hits {batb}  "
              f"(over {len(SEEDS)} caves)")
