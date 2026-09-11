# The damaged ear: predictions before the lesions

Human hearing-loss research gives us measured outcomes to aim at.
Each row: a lesion we can apply in config, the human result it
models, and a pre-registered prediction with its failure reading.
Sources at bottom. Nothing below has been run yet.

## The knobs (all in the CARFAC design, nothing ad hoc)

| lesion | CARFAC knob | models |
|---|---|---|
| L1 dead region | zero IHC conductance over a channel band | Moore's dead regions |
| L2 OHC loss | scale down undamping (zb / cochlear amplifier gain) | sensorineural loss, impaired frequency selectivity |
| L3 TFS loss | raise hair-cell tau_lpf / tau1_in (polarity corner drops) | degraded phase locking, temporal-fine-structure loss |
| L4 channel starvation | fewer channels (42 -> 16 -> 8 -> 4) | CI-like spectral resolution limits |
| L5 recruitment | AGC open loop | loss of cochlear compression |

## Predictions (falsifiable, with the human anchor)

**P1 — melody survives a mid-band dead region better than human
pitch quality does.** Humans hear off-place, distorted, "not tonal"
tones in dead zones (Moore). But our pitch comes from the SAI lag
axis, which POOLS every surviving channel: harmonics outside the
dead band still strobe at the fundamental's period. Predict:
recognition holds for dead bands up to ~1 octave; the place-centroid
feature drifts toward the band edge (the off-frequency-listening
signature, measurable per node); timbre bins covering the band go to
floor. Falsified if recognition dies with a narrow dead band -- that
would mean identity depends on place more than we claim.

**P2 — channel starvation has a cliff, and it sits near the CI
result.** CI users without rhythm cues recognize familiar melodies
at ~29% (vs 58% with rhythm), performing like normal hearers given
1-2 spectral channels; contour identification improves as intervals
widen. Predict: our recognition holds at 42 and 16 channels,
degrades at 8, near-chance at 4; and the wide-interval siren
outlives the narrow-interval drone cluster at every starvation
level (the interval-widening effect). NOTE our system has no rhythm
pathway (tempo is invisible by construction), so our starved ear is
the "rhythm removed" condition -- the honest comparison is against
the 29%, not the 58%.

**P3 — TFS loss fattens the nodes before it breaks the tune.**
Hearing-impaired listeners lose F0 discrimination for complex tones
while envelope processing survives. Our strobe rides fine structure
below the polarity corner; slowing the hair cell drops that corner.
Predict: tracker scatter (cents) grows smoothly as tau_lpf rises;
learned clusters fatten; recognition survives until cluster width
crosses the 10 mm match radius, then fails abruptly. The measurable:
scatter-vs-tau curve, and the tau at which the melody dies.

**P4 — OHC loss is a threshold shift first, a selectivity loss
second.** Scaling down undamping should (a) raise the level needed
for the salience gate to pass (audiometric threshold analog), and
(b) broaden effective filters -> place-centroid noise grows.
Predict: at moderate loss, quiet takes stop producing percepts
(sounds below "threshold") while loud takes still recognize; at
severe loss, recognition degrades even loud. Recruitment check (L5):
with the AGC open, the usable dynamic range between "inaudible" and
"everything saturates" narrows -- the classic recruitment picture.

## Method notes

- One lesion per run; healthy-ear baseline frozen first (the unit-
  surgery protocol, reused).
- Score with the existing anatomy toolkit: fit%, breaker
  decomposition, tracker cents, plus threshold sweeps for P4/P5.
- Human anchors are qualitative patterns, not numbers to hit --
  except P2's cliff, where the CI literature gives an actual
  performance-vs-channels curve to compare shapes against.

## Sources

- Melodic contour identification by CI listeners; FMI 58% with
  rhythm vs 29% without; 1-2 channel equivalence:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3627492/ and
  https://pubmed.ncbi.nlm.nih.gov/17485980/
- Expanded pitch contours aid CI melody identification:
  https://pubmed.ncbi.nlm.nih.gov/22193870/
- Moore, dead regions: diagnosis and perceptual consequences
  (off-frequency listening, non-tonal percepts):
  https://doi.org/10.1177/108471380100500102 and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4168936/
- TFS and pitch/masking/speech in normal and impaired hearing:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2580810/
- F0 discrimination, musical training, and TFS processing
  (psychophysics + modeling):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6513935/
- Impaired frequency selectivity and TFS sensitivity, envelope
  spared, in mild-moderate SNHL:
  https://pubs.aip.org/asa/jasa/article/146/6/4299/951580/

---

## OUTCOMES (run 2026-09-11, same afternoon as registration)

Method as registered: healthy-trained memory (Minor Swing +
Sentimental Reasons, real takes), lesioned listener at eval, plus a
TIMBRE-BLIND learning module as the pathway control (the caveat
above, made into a measurement). 24 full-percept runs + 16
timbre-blind runs. Figure: figures/lesion_study.png.

| prediction | outcome |
|---|---|
| P1 dead region: melody survives where human pitch quality dies | **CONFIRMED** — 400–800 Hz dead band, correct @21 even timbre-blind |
| P2 channel-starvation cliff near the CI result | **REFUTED AS STATED, replaced with something sharper** — place starvation ALONE is benign down to 4 of 83 channels (the SAI lag axis pools survivors). The cliff needs the temporal code lesioned too: 8ch + tau×8 (the true CI condition, which starves BOTH codes) = no_match on the riff, hedge-to-timeout on the drone. The interval effect confirmed: the octave-wide siren survives the CI condition at 21 steps while the cluster-object hedges. |
| P3 TFS loss fattens then breaks | **DIRECTION CONFIRMED** — tau×8 timbre-blind = confused; the smooth scatter-vs-tau curve and the tau×4/×8 non-monotonicity (full-percept runs) remain to be measured at percept level |
| P4 OHC loss = threshold first | **OPEN** — benign at performance level down to ×0.1 undamping; the quiet-level threshold sweep is still owed |
| caveat: raw-observation timbre masks ear damage | **VALIDATED EMPIRICALLY** — tau×8 reads correct with timbre, confused without. Lesion studies in this system must run timbre-blind. |

Headline: **pitch identity rides two redundant codes — place and
time. Either alone suffices; the temporal code is load-bearing for
melody; both gone (the implant condition) is melody-deafness,
matching the human FMI collapse.**
