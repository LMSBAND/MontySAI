# The 3D moth hunt: federation, elevation, acquisition (2026-09-11/12)

Bat = binaural ears (range + azimuth, calibrated to the verified
CARFAC pipeline) + two eyes (geometric stereo, range + full 3D
bearing, lit only). Ear = audiomonty/binaural.py, eye =
audiomonty/eyes.py. No Monty code touched.

## Elevation: two ears are deaf to UP
A horizontal ear-baseline resolves azimuth but not elevation -- the
cone of confusion. A moth that CLIMBS as it flees:
  two ears only .......... 0/6  (bat wanders the floor)
  ears + eyes (lit) ...... 6/6  (eyes supply the vertical)
  ears + eyes (DARK) ..... 0/6  (control: vision did the climb)

## Speed: the eye's edge is acquisition, not the stern-chase
Bryan predicted eyes win over speed. The sweep refined it:
- LEVEL moth, starts near (inside the 4.3 m sonar bubble): ears AND
  eyes both hold 100% until the moth outruns the bat (~100% of bat
  speed), then physics wins. Sonar suffices in its own range --
  speed alone does not separate the senses here.
- LEVEL moth, starts at 8 m (BEYOND sonar): ears blind (~0-17%, the
  17% only when the moth wanders back into the bubble); eyes acquire
  and hold 100% to the same physical speed limit. "Once the eye is
  on, it's on."

The eye is the pursuit sensor for three reasons that converge: it
sees past the sonar horizon (acquisition), it gives elevation
(3D), and -- from the log-polar result -- it holds the target's
identity as the target dilates on approach (scale invariance).
Sonar owns the dark and the terminal close range. Each sense covers
the other's measured blind spot; that is why an animal keeps both.
