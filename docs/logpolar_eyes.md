# Test 1, for eyes: scale is translation in log-polar (2026-09-11)

The bet (Bryan's, with the other Claude): the cochlea gave
transposition invariance for free because its axis is log-lag, so a
key change is a translation. The retina's cortical map is log-polar
(Schwartz; Weiman & Chaikin), so a SIZE change should be a
translation too -- and a Monty on that front end should recognize an
object at a never-seen scale with no scale code.

Evaluated OpenCV's warpPolar(WARP_POLAR_LOG) as the front end.
(The faithful biological retina -- bioinspired/convis, photoreceptor
adaptation + parvo/magno center-surround -- is the next layer;
needs opencv-contrib, not installed here. warpPolar alone carries
the geometry, which is the claim under test.)

Geometry proof: predicted radial shift M*log(scale) matched measured
to sub-pixel, 4/5 within 4 px (the miss at 2.0x is the object
clipping the field of view -- a real optical limit, named).

Recognition: same object 0.97-1.00 from 0.5x to 1.75x (flat over 1.8
octaves); different object 0.21; mirror-flip 0.00 (log-polar
preserves angle -- a reflection is not a radial shift, correctly
unrecoverable). Raw correlation ties only on near-twins; sharp
discrimination is the feature-graph's job (same lesson as timbre).

Three bugs caught by adversarial self-audit before trusting a number
(all in experiments/logpolar_scale.py docstring): fixation-point blob
at r=0, correlate() edge artifact, scaling about object vs fixation.

Next: a RetinaSM emitting feature-at-(angle, log-radius) percepts on
the SensorModule contract -- then a Monty that learns an object at
one size and recognizes it at another, and finally ear + eye voting
on one object. Q3 in the plan; the belt is loose but the buckle holds.
