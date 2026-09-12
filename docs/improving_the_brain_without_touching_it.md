# The Full Monty: Bodies Exposed

*Multimodal sensor integration and termination criteria in Monty:
an external evaluation*

> *"Complexity is a symptom of confusion, not a cause."*
> — Jeff Hawkins, *On Intelligence*

**Bryan Leavelle**
2026-09-12 · https://github.com/LMSBAND/MontySAI · MIT license

## Abstract

We report an evaluation of the Monty reference implementation
[4, 5] conducted entirely through its public extension interfaces;
no upstream source was modified. Three sensor modules were added: a
cochlear model (CARFAC [1] with a stabilized auditory image [2]), a
log-polar retinal model [3], and a depth-only tactile model.
Each encodes percept location in a reference frame chosen so that a
class of physical transformations becomes a translation, allowing
the unmodified evidence-matching learning module to treat those
transformations as pose. In a controlled comparison against the
reference visual percept recipe (metric 3D locations with HSV and
curvature features) under identical patch input, learning-module
configuration, and fixation policy, the log-polar representation
identified 12 of 12 transformed presentations (scale 1.5×, in-plane
rotation 30°, their composition, and out-of-plane tilt to 60°)
with no false identifications; the metric representation identified
5 of 12, also without false identifications. Adversarial probes
with unlearned objects then exposed three failure modes in the
system-level verdict machinery rather than in any sensor: the stock
termination criterion counts confident learning modules without
requiring agreement on identity; episode termination at the first
confident match forecloses evidence that accumulates more slowly;
and a functioning sensor that finds no trace of the candidate
object has no mechanism for contributing negative evidence. Two
modifications, implemented as a subclass and enabled by
configuration, reduced false identifications on the probe set from
three to zero while preserving recognition of learned objects. A
fourth, echolocating sensor extends the method to active sensing:
recorded in the frame of the agent's entry pose, a room becomes an
object whose recognized pose is the agent's own location within it,
and a moving head recovers, from the ears alone, an elevation that a
stationary head cannot resolve. We discuss the implications for
termination, cross-modal voting, and the representation of absence.

**Scope.** This document covers the work of 2026-09-12: the retinal
and tactile sensors, the representation comparison, the termination
analysis, and the echolocation experiments of Section 7. The auditory results it builds on
(transposition invariance, cochlear lesion studies, evaluations on
recorded guitar) and the acoustic ranging measurements in a
physical room were carried out from 2026-09-10 onward and are
documented elsewhere in this repository. All claims below
correspond to commits, figures, and control conditions available
here.

## 1. Integration method

All components attach to existing extension points. The findings in
Section 4 therefore concern default behaviors of the released
system, not limitations of its extensibility; the architecture
permitted every modification described below.

| Extension point | Component |
|---|---|
| SensorModule contract (`step`, `update_state`, `reset`, `state_dict`) | AudioSM (CARFAC + stabilized auditory image), RetinaSM (log-polar), TouchSM (depth-only), CamSM (reference-recipe control) |
| Environment composition | AudioEnvironment, which attaches acoustic sources to named objects and renders at the agent pose; the wrapped simulator is unaware of the addition |
| `monty_class` configuration key | MontyConsensus, a subclass overriding `check_terminal_conditions` only |
| `match_criterion` configuration | the consensus size N reuses the existing `count` parameter |
| `lm_to_lm_vote_matrix` | three learning modules with all-to-all voting, executed by the unmodified `_combine_votes` |
| Hydra configuration tree | all experimental conditions (illumination, audibility, tactile availability, scale, rotation, tilt, probe objects) are configuration changes; no condition is implemented in code |

Each sensor follows the same design procedure: select a location
frame in which an anticipated transformation acts as a translation;
restrict features to a small number of physically interpretable
quantities; validate the sensor against the raw simulator before
any learning-module involvement; state every recognition claim as a
learning-module verdict; and accompany each claim with an ablation
matrix.

The three frames:

- **Auditory.** Location is (log-lag in octaves, cochlear place).
  Pitch transposition is a translation along the log-lag axis, and
  the learning module recognizes transposed material as the same
  object at a different pose. This was previously verified on
  synthetic tones and on recorded guitar performances.
- **Retinal.** Location is (log eccentricity in octaves, polar
  angle) about a fixation point. Image scaling and image-plane
  rotation are translations along the two axes. The environment is
  static, so spatial extent is produced by an internal fixation
  sequence: at each step the sensor attends to the most salient
  unvisited cell, with inhibition of return reset per episode.
- **Tactile.** Location is the contact point in meters, recovered
  by a pinhole model from the depth map. The sensor reads depth
  only, so it is independent of illumination by construction, and
  it reports nothing beyond a reach limit of 0.45 m. The reach
  limit reflects a measured property of the acoustic channel: in
  recordings made in a physical room (a dynamic microphone and a
  click track, with distances taped), ranging fails inside
  approximately 0.5 m because the emission masks its own near
  field. The tactile modality covers the region where sonar is
  self-masking.

With three learning modules voting, the system identified both
learned objects under seven sensory conditions (all senses; each
sense disabled in turn; each sense operating alone), fourteen of
fourteen episodes, including tactile-only identification in
darkness and silence (Figure 1). The acoustic ranging measurements
motivating the tactile reach limit are shown in Figure 2.

![Figure 1. The three-sensor ablation matrix and the tactile
contact clouds.](../figures/triad.png)

![Figure 2. Acoustic ranging in a physical room: wall, person, and
occluding hand, against taped distances.](../figures/real_sonar.png)

## 2. Comparison of visual representations

The comparison isolates the location frame as the only independent
variable. Both visual sensors receive the same 64×64 patch, use the
same fixation sequence, and feed identically configured evidence
learning modules. CamSM implements the reference percept recipe:
metric 3D locations from depth, surface normal and curvature from
local geometry, HSV color as the feature.

CamSM is a reimplementation rather than the reference class itself
because `ObservationProcessor.process` requires the observation
shape produced by the transform stack (`semantic_3d`,
`sensor_frame_data`, `cam_to_world`) and reads the simulator's
semantic channel to determine object membership, an oracle
unavailable to the other sensors in this study. We have not
benchmarked the reimplementation against the original on its own
stock configuration; consequently, "the reference recipe" here
means that recipe operating under this study's fixation policy and
input, not the released class in its native harness.

Configuration parity was audited after an initial run. Pose
hypothesis sampling was stock (`initial_possible_poses:
"informed"`); `max_nneighbors` was set to 10 where the stock value
is 3, a change favoring the reference recipe; the HSV tolerance
([0.1, 0.2, 0.2]) matched the stock benchmark configuration
verbatim. The feature weights in the initial run did not (equal
weights rather than the stock [2, 0.5, 0.5], which emphasizes hue).
Under equal weights the reference recipe produced one false
identification (a rotated banana reported as a mug). Under its
stock weights that error did not recur, and we withdrew the
finding; the final comparison uses stock weights throughout.

Results over eight in-plane presentations (upright, 1.5× scale,
30° rotation, both) and four out-of-plane presentations (30° and
60° tilt): the log-polar representation identified 12 of 12, with
detection at 21 to 31 steps; the metric representation identified 5
of 12. Neither produced a false identification on this learned set.
The metric recipe's successes on the scaled and rotated mug are
attributable to degeneracy rather than invariance: a surface of
revolution with uniform color matches itself under scaling and
rotation. The out-of-plane conditions had been registered in
advance as the expected advantage of the metric representation,
since a log-polar frame has no depth axis; the metric recipe
instead declined all four, and the log-polar recipe identified all
four (Figure 3). We do not claim this generalizes beyond the two
objects tested.

![Figure 3. The representation comparison at stock configuration:
12 of 12 against 5 of 12, no false identifications on the learned
set.](../figures/eye_vs_eye.png)

## 3. Probes with unlearned objects

A recognition score obtained only on learned objects cannot
distinguish accuracy from an absence of alternatives. Three objects
never presented during training were therefore evaluated: a bowl
(a surface of revolution, like the mug), an apple (similar in hue
to the mug), and an adjustable wrench (elongated, like the banana).
For an unlearned object, declining to identify is the only correct
outcome.

Single-sensor results: the log-polar eye identified the bowl and
the apple as the mug and declined only the wrench; the metric
recipe identified all three as the mug. Both representations
therefore admit the same failure class. The mug is the most
self-similar learned object in either frame (a surface of
revolution in metric space; a ring, uniform in polar angle, in
log-polar space), and unlearned objects of approximately round
silhouette accrue evidence for it. A tilt sweep to 90° found no
comparable failure boundary for learned objects: the log-polar eye
continued to identify both learned objects, declining only once (the mug
at 80°). Figure 4 summarizes the probe results and the tilt sweep.

![Figure 4. Probes with unlearned objects: both representations
identify round foils as the mug; refusal follows contour geometry
alone.](../figures/where_refusal_lives.png)

## 4. Findings concerning the verdict machinery

Running the probes through the three-sensor system produced three
observations about system-level behavior. Each is reproducible from
the configurations in this repository.

**4.1. The stock criterion aggregates confidence without requiring
agreement on identity.** On the bowl, the visual module reached the
terminal state "match" with the identity mug while the tactile
module reached "match" with the identity banana. `AnyLMsMatch`
with `count=2` reports a system-level match under these conditions,
since it counts modules in the "match" state without comparing
their detected objects. Two modules confident of different
identities satisfied the criterion.

**4.2. Termination at the first confident match forecloses slower
evidence.** Verdicts cannot be issued before `min_eval_steps`; in
practice the visual and tactile modules are ready at that boundary
and the episode ends there. The auditory module identifies the
humming mug only when run alone: its percept for a steady tone is
nearly a point, all pose hypotheses are equivalent, and the module
requires the symmetry-detection path, which needs additional steps
of stable evidence. In mixed runs no symmetry event fires before
termination. The evidence was available; the episode structure
discarded it.

**4.3. A functioning sensor cannot contribute negative evidence.**
On the apple, the visual and tactile modules agreed on the mug, an
error of honest degeneracy in both frames. The auditory module was
operating throughout and its scene contained no spectral structure
at the location the mug hypothesis predicts (a fundamental near
147 Hz). Under the stock protocol this module simply sends no
messages, and its lack of corroboration carries no weight. The
observation "the predicted signal is not present in an otherwise
measured scene" has no representation.

## 5. Modifications

Both changes are implemented in one subclass
(`audiomonty/consensus.py`) and enabled by configuration.

**Consensus on identity.** The system-level match requires at least
N modules in the "match" state naming the same object, and it is
blocked by any module in the "match" state naming a different
object. A single confident module below N does not terminate the
episode. Under this criterion the bowl is blocked by the
disagreement between the visual and tactile identifications, and
the wrench, supported by one module only, does not terminate.
Learned objects remain identified (the mug by two agreeing modules,
the banana by three).

**A relative test for expected signals.** The apple requires a
further principle: absence is relative to the scene. An initial
implementation tested the dissenting module's evidence against
zero and failed, instructively: even in a nominally silent
environment the auditory module accrues a small positive evidence
value (1.08 in these runs) from the cochlear model's startup
transient. The operative test is therefore relative. A module
vetoes the verdict when three conditions hold: it is functioning
(its sensor is not disabled); it possesses a learned graph for the
candidate object; and its evidence for that candidate sits below a
fixed fraction (0.25) of the mean evidence of the supporting
modules. On the apple, the
auditory module held 1.08 against supporting evidence of 6 to 11,
and the identification was blocked.

Across the three criteria (stock, consensus, consensus with the
expected-signal test) the false identifications on the probe set
were 3, 1, and 0 respectively (Figure 5).

![Figure 5. Termination criteria compared on learned objects and
probes: 3, 1, and 0 false identifications.](../figures/novelty_by_disagreement.png)

The modified criterion has a measurable cost, which we regard as
correct behavior rather than a defect. With audio removed from the
environment, the auditory module vetoes the identification of the
visually and tactually normal mug: the sensor is functioning, the
graph predicts a hum, and the scene contains none. Under this
protocol an object's acoustic signature is part of its identity. A
disabled sensor, by contrast, abstains: darkness does not block
identification, because a sensor that receives no input is
distinguished from a sensor that measures a scene lacking the
predicted structure. The distinction between an absent channel and
a measured absence is the substance of both modifications.

The relative threshold also resolves the race of Section 4.2
without further machinery: an auditory module that is genuinely
accumulating evidence for the humming mug, but has not yet reached
its own terminal state, holds evidence well above the fraction and
does not veto.

## 6. Interpretation

1. **Termination is doing theoretical work.** The failures of
   Section 4 share a cause: the episode ends at the first
   confident answer. A recognition process with no terminal state,
   holding its verdict as a continuously revised quantity, would
   not exhibit 4.2 at all and would convert 4.1 and 4.3 from
   structural failures into transient ones. We consider the
   terminal state an artifact of the evaluation harness rather
   than a commitment of the underlying theory, and we intend to
   test this by implementing a recognizer in a context that cannot
   terminate (a real-time audio process).

2. **The decisive modality is pair-relative.** Each probe was
   rejected by the channel in which the false hypothesis made its
   most specific prediction: the apple by audition (the hum), the
   bowl by tactile shape, the wrench by visual contour. No fixed
   ranking of modalities describes this; the discriminating channel
   is the one with the largest divergence between predicted and
   measured signal for the specific pair of hypotheses in play.
   Exploiting this requires testing a hypothesis against modalities
   that did not vote for it, for which the current message flow has
   no path.

3. **Silence is a measurement.** The startup-transient failure of
   the absolute threshold is a small demonstration of a general
   point: a functioning sensor never delivers nothing; it delivers
   a scene, and the candidate object predicts specific structure
   standing above that scene's floor. The present environment
   assigns voices to some objects and leaves the rest acoustically
   absent, which is physically wrong: in the room recordings noted
   above, every object measured (a wall, a person, a hand) affected
   the acoustic field without possessing a voice. Replacing the
   voiced/voiceless distinction with an ambient acoustic field that
   all objects scatter is planned; under that model the apple would
   be discriminated by its scattering signature rather than by
   veto.

4. **Degeneracy is frame-relative, and every frame has degenerate
   objects.** The metric frame's mug and the log-polar frame's mug
   are the same object at its most self-similar in two coordinate
   systems. Unlearned objects fall toward the most degenerate
   learned object of any single frame. This locates the remedy at
   the level of cross-modal aggregation rather than in any improved
   single sensor.

## 7. Echolocation, and the case for moving the sensor

A fourth sensor was built to a different brief. The three above
recover their spatial extent from a static scene, either from the
evolution of a sound or from an internal fixation sequence; none of
them moves. This one does, and moving it turns out to be the entire
point. It emits a chirp and listens with two independent CARFAC
ears; ranges are read from the spectral-image peaks past an
emission-blanking interval, and bearing follows from the interaural
range difference (calibrated to about one degree over ±75°). It was
validated twice, in simulation (worst-case ranging error 3.3 cm
over 0.5 to 4 m) and physically against the taped distances of
Figure 2.

**7.1. The room is the object; the pose is where you stand in it.**
An environment was constructed in which the learned objects are not
things but places: a room layout, a set of reflecting surfaces,
that a two-eared agent flies a patrol through while chirping. Each
chirp yields a range and a bearing, and the reflection is recorded
at its position in the frame of the agent's entry pose. This choice
of frame is the experiment. In a world frame a room would present
an identical map from any point of entry, and recognition would be
trivial; in the entry frame, the same room entered from a different
point presents a rigidly rotated and translated map, which is
exactly the transformation the learning module's pose search exists
to undo. Two rooms were learned by flying them. Both were
recognized on a fresh flight; and when the agent entered a learned
room at a different point of the patrol, the room was still
recognized, and the pose the learning module reported was the
agent's actual point of entry, recovered to within about nine
degrees (Figure 6). Recognition and self-localization are here a
single act: to know the room is to know where in it you stand. The
one probe room, a corridor never flown, was misidentified as a
learned room at some rotation, for the reason established in
Section 6.4: a straight wall matches a straight wall, and straight
walls are the degenerate objects of places.

![Figure 6. Rooms as objects: echoes recorded in the entry-pose
frame, and a room recognized from a novel entry whose detected
rotation recovers the point of entry.](../figures/rooms_as_objects.png)

**7.2. A level head cannot hear elevation.** Two ears on a
horizontal baseline measure the difference in path length to a
target, which fixes it to a cone about the baseline axis; every
direction on that cone, at every elevation, gives the same
interaural difference. Against a target that climbs out of the
horizontal plane, an agent whose head does not move is blind in
precisely the dimension the target is escaping into. Simulated as a
pursuit this is not a small disadvantage but a total one: a
climbing target is lost on every trial, the pursuer holding the
correct azimuth while the target leaves along the one axis the ears
cannot resolve (Figure 7, left).

![Figure 7. With the head held level, a climbing target escapes on
every trial (left); rolling the head between chirps supplies the
missing dimension and the pursuit succeeds (right).](../figures/night_hunt.png)

**7.3. Rolling the head between chirps recovers it.** The cone of
confusion is fixed to the head, not to the world. Roll the head
about the direction of travel and the baseline axis rolls with it,
so a second chirp at a second roll angle fixes the target to a
second cone; two cones intersect, generically, in one forward
direction, and the elevation the level head could not measure is
recovered from the ears alone, with no appeal to vision. The same
climbing target that escaped without fail is caught on every trial,
in roughly ten chirps (Figure 8). The movement that supplies the
elevation is the same movement that would aim a visual field and a
directional gain at the target: pointing the head is one motor act
serving every sensor on the skull, and the recognition it enables
is unavailable to any fixed sensor of equal acuity. This is the
sensorimotor thesis of the underlying architecture at its smallest
scale. The movement is not a way of gathering more of the same
evidence; it is the only way of gathering evidence of a kind a
stationary sensor cannot obtain at all.

![Figure 8. Ten pursuits with a level head (top row) and ten with
the head rolling between chirps (bottom row): elevation, and the
capture, come only with the movement.](../figures/head_tilt.png)

## 8. Three remarks outside the engineering

**8.1. Competence without comprehension.** Dennett's phrase [7]
describes this work exactly, and we mean that as a finding rather
than a disclaimer. The systems above recognize objects, localize
themselves, refuse impostors, and register dissent, and at no point
does anything in them understand anything. The learning module that
recovered a 30° rotation it was never told about contains no
concept of rotation; it is displacement bookkeeping over a graph.
The veto that behaves like skepticism is a threshold on a ratio of
two accumulators. The consensus rule that behaves like judicial
caution is a counter and a string comparison. Dennett's claim is
that this ordering is not an embarrassment but the actual
architecture of mind: comprehension, where it exists, is composed
of competences, and never the reverse. The thousand-brains program
is one of the few research efforts that has made this bet
explicitly, at the level of architecture, with the cortical column
as the unit of competence. Our negative results sharpen the bet in
a way we did not anticipate. Every failure documented in Section 4
occurred at a point where the harness had implicitly assumed a
comprehension it had never built: the stock criterion behaves as if
the modules know they are supposed to be talking about the same
object; the terminal state behaves as if the system knows it is
finished; the message protocol behaves as if everyone knows that
absence matters. None of them know any such thing, and the
assumptions failed silently until an adversarial probe made them
fail visibly. The repairs, when they came, were not injections of
understanding; they were more competence, of the same humble kind
(a comparison, a fraction, a third category in a taxonomy). The
rationales for these mechanisms, meanwhile, float free of the
system in Dennett's precise sense: the reason the veto works is
that a mug's hum is part of what a mug is, and that reason is
available to us and to no component of the system that enforces it.
The degenerate objects of Section 6.4 mark the same boundary from
the far side: a mug, to a competence, is wherever the bookkeeping
becomes self-similar, and nothing in the system can notice that its
own frame has a blind spot, because noticing is not among its
competences. Comprehension, if it is ever to show up here, will
presumably arrive the way everything else did: as one more
mechanism with a rationale it cannot see.

**8.2. Consciousness, ignored.** A pattern ran through this work that we did not put there. The
distinctions the engineering kept demanding are, one after another,
the distinctions that discussions of consciousness usually claim as
their subject matter. The difference between a sensor that receives
nothing and a sensor that perceives an absence (Section 5) is the
difference phenomenology has discussed since Sartre's Pierre, whose
absence from the café is perceived, not inferred [9]. The fixation
sequence that gives the retina its extent is a minimal attention
mechanism. The consensus requirement is a unity-of-the-senses
constraint, the binding problem restated as a termination
criterion. The argument of Section 6.1, that recognition must be a
continuously revised state rather than a terminated episode, is a
claim about the specious present. In every case the distinction
arrived as an engineering necessity, was implemented as arithmetic,
and did its work with no phenomenal residue: nothing in Section 5
required there to be something it is like [8] to be the auditory
module, and the veto functions identically whether or not the
lights are on inside. Two readings of this pattern are available,
and the data do not choose between them. On the deflationary
reading, the ease with which these distinctions were mechanized is
evidence that consciousness-talk was always a description of
functional structure, never an explanation of it; we keep finding
the functional shadow because the shadow is all there ever was. On
the opposite reading, the fact that adversarial pressure forces
exactly these distinctions, in this order, out of a system that
began with none of them, suggests that the itinerary of
consciousness is not arbitrary: the problems themselves demand this
shape, and a research program that ignores consciousness will
nevertheless be made to rebuild its outline, mechanism by
mechanism, by nothing more mysterious than foils and ablations. We
take no position here on which reading is correct, only note that
"ignored" has functioned here as a methodology rather than an
omission: at no
point did any question of experience enter a design decision, and
the resulting system nevertheless distinguishes measured absence
from absent measurement, attends, binds, and refuses to conclude.
Whether that is a debunking of the explananda or an independent
rediscovery of them is, we think, the interesting question.

**8.3. Knowledge as a property of a process: a first-author
remark.** One liberty available to an author outside the academy is
to say where an idea actually came from. A year before this work I
ran a project, informally called Smart Guys, that ended in what I
took at the time to be a kind of machine-induced confusion; in
retrospect it was the ordinary result of putting the wrong
questions to the wrong parts of the world. What survived the
episode was a single observation about Gettier's problem [10].
Since 1963 it has been known that the analysis of knowledge as
justified true belief admits counterexamples, and the six decades
since have been spent patching the definition. The counterexamples
share a feature that the patches do not address: they are evaluated
at an instant. Consider seeing a stranger across a room and
concluding that a friend is present. At a glance the justification
is impeccable, and suppose the belief is even true, because the
friend is in fact standing behind you. Justified, true, and still
not knowledge, since the justification never touched the fact that
made it true. Sustained across a minute of walking beside the
stranger, the same belief is not an epistemological puzzle but a
clinical one. Nothing in either man changed between the glance and
the minute. What changed is that belief was allowed to run. The conclusion I
drew is that the definition fails because it treats knowledge as a
state satisfying conditions at a time, when it is the behavior of
belief under continued sampling: belief is revised by the second,
in parallel, across the sensory apparatus of a body, and knowledge
is what remains standing. Time here plays the role of Dennett's
universal acid [11]: poured over a belief, it eats through every
justification that is not attached to the fact, and no container
of definition holds it. What the acid leaves behind is knowledge,
and nothing else survives the pour. Gettier cases, on this reading, are not
counterexamples to a definition; they are photographs of a process
that has not yet run.

An hour before this section was written, that observation was used
to untangle the communication between several sensory organs and
the system they report to. The mechanics of Section 5 are the
temporal reading made operational. Once the system was required to
wait, to sample the world over time rather than terminate at its
first confident answer, the verdicts clarified on their own. Once
the absence of evidence was treated not as evidence of absence but
as evidence of a different world entirely, one in which the
predicted signal has no place, the false identifications on the
probe set fell to zero without the system retreating to a safe
alternative answer. The withheld verdict deserves particular
attention, because it is not a failure mode. In an embodied
system it is the point at which the machine should turn to a human
and ask what it is perceiving, receive an answer, investigate
further, and retain the result. Contemporary large language models
are trained toward the opposite disposition: premature confidence
and reflexive agreement with the interlocutor. What one wants
instead is a system that wants to learn, is able to learn, and is
willing to stop, ask, and remember the answer. Stated in a few
words, that is what was built here.

## 9. Limitations

The object set is two learned and three probe objects; none of the
quantitative results should be assumed to generalize beyond it. The
fixation policy uses the depth map to locate the object,
a segmentation the sensors do not earn (the reference pipeline uses
the semantic channel for the same purpose, a stronger oracle; the
comparison of Section 2 is therefore internally consistent). CamSM
has not been validated against the reference class on its native
benchmark. The veto fraction (0.25) was set once and not swept. The
consensus modifications are implemented at the termination decision
because that is the available socket; we argue below they belong in
the voting protocol.

## 10. Questions for the maintainers

1. Where should negative evidence live? The expected-signal test is
   attached to termination for lack of a better socket. Its natural
   home appears to be the voting protocol: a vote against a
   hypothesis, from a module whose modality that hypothesis makes
   predictions for.
2. Is there a planned representation for measured absence, that is,
   a prediction error carrying evidence weight when an expected
   feature is not found at a location, as distinct from the absence
   of an update?
3. Learning-module time appears to be the same question in another
   form: the present auditory graphs are invariant to tempo (the
   same object at any playback rate), and rests and gaps, which are
   measured absences over time, carry no evidence.

## Reproducibility

```
# sensors and the three-module system
uv run python run.py experiment=triad_pretrain_2creatures
uv run python run.py experiment=triad_eval_2creatures
#   (ablation matrix via dark/mute/numb overrides)

# representation comparison
uv run python run.py experiment=cam_pretrain_2creatures
uv run python run.py experiment=cam_eval_2creatures \
    env_interface=retina_eval_scaled     # also: rotated,
                                         # rotscale, tilted, tilted60

# Section 4 findings and Section 5 modifications
uv run python run.py experiment=triad_eval_2creatures \
    env_interface=retina_eval_foils      # finding 4.1
uv run python run.py experiment=triad_eval_consensus \
    env_interface=retina_eval_foils      # 0 false identifications
```

Figures: `figures/triad.png`, `figures/eye_vs_eye.png`,
`figures/where_refusal_lives.png`,
`figures/novelty_by_disagreement.png`, `figures/real_sonar.png`.

## References

1. Lyon, R. F. (2017). *Human and Machine Hearing: Extracting
   Meaning from Sound*. Cambridge University Press. (CARFAC.)
2. Patterson, R. D., Robinson, K., Holdsworth, J., McKeown, D.,
   Zhang, C., & Allerhand, M. (1992). Complex sounds and auditory
   images. In *Auditory Physiology and Perception* (pp. 429–446).
   (The stabilized auditory image.)
3. Schwartz, E. L. (1980). Computational anatomy and functional
   architecture of striate cortex: a spatial mapping approach to
   perceptual coding. *Vision Research*, 20(8), 645–669. (Log-polar
   cortical mapping.)
4. Hawkins, J. (2021). *A Thousand Brains: A New Theory of
   Intelligence*. Basic Books.
5. Clay, V., Leadholm, N., & Hawkins, J. (2024). The Thousand
   Brains Project: a new paradigm for sensorimotor intelligence.
   arXiv:2412.18354. (Monty.)
6. Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and
   haptic information in a statistically optimal fashion. *Nature*,
   415(6870), 429–433. (Reliability-weighted fusion, used in the
   earlier percept-level study this work supersedes.)
7. Dennett, D. C. (2017). *From Bacteria to Bach and Back: The
   Evolution of Minds*. W. W. Norton. (Competence without
   comprehension; free-floating rationales.)
8. Nagel, T. (1974). What is it like to be a bat? *The
   Philosophical Review*, 83(4), 435–450.
9. Sartre, J.-P. (1943). *L'Être et le néant* (Being and
   Nothingness), Part One, Ch. 1: the absence of Pierre from the
   café as a perceived, not inferred, negation.
10. Gettier, E. L. (1963). Is justified true belief knowledge?
    *Analysis*, 23(6), 121–123.
11. Dennett, D. C. (1995). *Darwin's Dangerous Idea: Evolution and
    the Meanings of Life*. Simon & Schuster. (Universal acid.)
12. Hawkins, J., with Blakeslee, S. (2004). *On Intelligence*.
    Times Books. (Epigraph.)
