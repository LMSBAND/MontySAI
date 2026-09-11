# Blind cartography (2026-09-11, evening)

Design (Bryan + the other Claude): ten reflex escapes from random
starts, every (pose, echo-range) logged; a mono ear gives range
without bearing, but MOTION supplies the bearing -- each observation
constrains a post to a circle, and posts appear where circles agree.
Then escape eleven runs the same reflex with the map steering.

Results:
- 872 range observations across ten escapes; all ten exited OVER THE
  TOP of the fence (the Roomba result, ten for ten -- the control).
- Map: 12/12 posts detected (>=6 circle crossings, within 20 cm),
  median error 4.5 cm, worst 11.7 cm -- from oblique glances during
  flinches, not head-on ranging.
- Prediction confirmed visually: the occupancy is BRIGHTEST at the
  door-adjacent posts (heard from both sides by every wanderer);
  the far fence is faint. The map is sharpest where a gap-finder
  needs it.
- Honest wart: 557 ghost peaks from stray circle intersections; they
  wrecked the map's door-WIDTH estimate (0.16 m claimed vs 1.20
  true) while its CENTER landed at y=0.84 vs truth 0.80. Ghost
  suppression (crossing-count weighting) is the map's next chore.
- ESCAPE ELEVEN: crossed the fence at y=0.83 -- THROUGH THE DOOR,
  21 steps, first crossing in eleven attempts. Same ear, same
  reflex underneath; the only addition was memory.

Sim pose is exact, so these errors are the FLOOR; a robot adds IMU
drift on top. The sim number is the clean half of that comparison,
kept here for the day someone hands this thing a body.


## Reviewer audit (same evening, via the other Claude)

The mask question: ghosts are removed by AGREEMENT ACROSS ESCAPES
(>=8 range-circles), never by reference to truth. Blue rings in the
figure are overlay only. BUT the audit caught a real leak elsewhere:
the first door-finder searched the band x in [2.6, 3.4] -- chosen
knowing the fence is at x=3. Fixed truth-free: fence x inferred from
the map itself (3.06 vs truth 3.00), and openings found as regions
that are BOTH weak in agreement AND certified empty -- each chirp's
nearest echo at range r proves the whole disc inside r is empty
(free-space evidence, previously discarded).

What the honest finder produced: three openings -- beyond the
fence's end (1.76 m), THE DOOR (1.04 m at y=0.80; truth 1.20 at
0.80), and a false southern opening traced to the range-log filter
clipping sub-0.5 m echoes (discs overestimated; fix logged).
Goalless, the animal takes the biggest opening: over the top --
correctly. With a destination beyond the door, travel cost picks the
door (4.83 m vs 8.13) and it crosses at y=0.72. Same map, different
goal, different route: the choice is real now, which sets up the
proper Tolman experiment (train a detour habit in a Monty LM, open
a shortcut, see which it takes).

Scope, on the record: sim pose is exact, so all map errors are the
FLOOR. The robot number is this plus IMU drift -- the ceiling is the
next experiment.
