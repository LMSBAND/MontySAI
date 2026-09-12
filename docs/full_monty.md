# The Full Monty: ear + eye vote on one creature (2026-09-11)

Percept-level sensor fusion, NO Monty code changed (the framework's
_combine_votes + a 2-SM connectivity config is the real-integration
follow-up; this proves the science first). Two creatures, each a
voice AND a face:
  A = 147 Hz drone + MOTH visual signature
  B = 294 Hz drone + RING visual signature
Ear posterior = fit_fraction of the SAI percept stream to each
creature's pitch-location memory. Eye posterior = log-polar match to
each creature's template. Fuse by product (independent evidence).

Results, four conditions (truth = A):
  Full Monty (ear+eye):  ear 1.00, eye 0.82 -> FUSED 1.00  = A
  dark room (ear only):  ear 1.00, eye 0.50 -> FUSED 1.00  = A
  silent (eye only):     ear 0.50, eye 0.82 -> FUSED 0.82  = A
  CONFLICT (looks A, sounds B): ear 0.00, eye 0.82 -> FUSED 0.00 = B

Findings:
- NO SINGLE POINT OF FAILURE: lose either sense and the other still
  carries the vote to A. This is the thousand-brains claim -- a
  federation degrades gracefully -- shown with two real front ends
  speaking one percept language.
- CONFLICT IS RELIABILITY-WEIGHTED, not split-the-difference: the
  fusion followed the more certain sense (the ear, categorical for
  cleanly-separable pitches) and reported B. That is optimal cue
  integration (Ernst & Banks 2002, visual-haptic), not a bug. The
  McGurk-style "conflict -> genuine uncertainty" outcome requires
  the two cues to be COMPARABLY reliable; here they were not.
- Honest scope: two creatures whose EARS overlap (drone vs a siren
  that sweeps through the drone's pitch) gave a dead ear (P=0.50
  everywhere) on the first run -- caught and recast to separable
  voices before trusting anything. A dead sensor makes a fake
  federation.

NEXT: (1) equal-reliability conflict (degrade the ear to match the
eye's confidence) to elicit the true McGurk uncertainty; (2) the
real integration -- RetinaSM on the SensorModule contract + a
1audio_1retina connectivity config, so Monty's own _combine_votes
runs the fusion instead of this percept-level stand-in.
