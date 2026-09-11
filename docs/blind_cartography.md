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
