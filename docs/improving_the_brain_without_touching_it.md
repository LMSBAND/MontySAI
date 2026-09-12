# Improving the brain without touching it

**TL;DR** — We gave Monty three senses through its stock extension
sockets (zero edits to tbp.monty), then pushed the federation with
adversarial foils and found the false IDs come from the verdict
machinery, not the sensors: (1) the stock quorum counts *confident*
LMs, not *agreeing* ones — it convicted on two senses naming
different objects; (2) first-winner termination forecloses evidence
that accumulates slower than confidence; (3) a working sense that
finds *no trace* of the candidate has no way to testify against it.
Two subclass-sized fixes — consensus-on-identity and a relative
expected-signal veto — took foil false-identifications **3 → 1 → 0**
while learned objects stayed recognized.

**Scope**: this document covers 2026-09-12 — the eye, the fingertip,
the eye-vs-eye duel, and the verdict-machinery work. The ear it
builds on (key-is-pose, the lesion studies, the riff evals) and the
real-room sonar takes are earlier work in this repo, 2026-09-10
onward; see `figures/` and `docs/`. Every claim below has a commit,
a figure, and its controls here.

We built Monty two new senses, ran its own eye against ours in a
controlled duel, then pushed the whole federation until it lied to
us — and found that the lies came from the *verdict machinery*, not
the sensors. We fixed that machinery through Monty's own extension
sockets, took the false-identification count from 3 to 0 on
adversarial foils, and paid the costs openly. This document is the
why, in all the details, because the details are where the theory
claims live.

---

## 1. What was plugged in (the plumbing was all the work)

Everything below went through stock extension points. This matters:
the critique that follows is aimed at *defaults doing theoretical
work*, not at the code — the architecture allowed its own repair,
which is to its credit.

| Socket | What we hung on it |
|---|---|
| duck-typed SensorModule contract (`step / update_state / reset / state_dict`) | `AudioSM` (CARFAC+SAI ear), `RetinaSM` (log-polar eye), `TouchSM` (depth-only fingertip), `CamSM` (reference-eye control arm) |
| `Environment` wrapping | `AudioEnvironment` — objects get voices; the inner MuJoCo sim never knows |
| `monty_class` config key | `MontyConsensus` — a subclass overriding one decision (`check_terminal_conditions`) |
| `match_criterion` config | consensus size N rides the existing `count` plumbing |
| `lm_to_lm_vote_matrix` | 3 LMs, all-to-all — Monty's own `_combine_votes` runs the federation |
| hydra conf tree | our yamls symlinked in; conditions (dark/mute/numb/scaled/rotated/tilted/foils) are config swaps, never code |

The three senses share one design recipe, proven three times:
**pick the reference frame where the transformation you fear becomes
a translation; keep the features few and physical; selftest against
the raw simulator; make every claim as a Monty verdict; ship the
ablation matrix.**

- **Ear**: location = (log-lag octaves, cochlear place). Transposition
  is translation → *key is pose* (LM verdict, real guitars).
- **Eye**: location = (log-r octaves, theta) about a fixated fovea.
  Scale and image rotation are translations → *scale is pose,
  rotation is pose*. The LM recognized 1.5× and 30°-rotated objects
  as sole hypotheses and reported `scale 1.0` — no scale code exists
  anywhere to cheat with; the rotated banana's detected pose read
  ≈ −27°, a rotation the LM measured without being told.
- **Touch**: location = true meters via pinhole; depth only
  (photon-free by construction), reach-limited to 0.45 m. The reach
  limit is physics we measured, not decoration: our real-room sonar
  takes (SM57, click track, tape measure) showed the ear jams itself
  inside ~0.5 m — its own emission owns the near field. Touch owns
  the donut hole.

Triad result: 7 conditions × 2 creatures = **14/14**, including each
sense alone — touch recognized both creatures in dark silence.

## 2. The duel (and the audit that made it unimpeachable)

Same 64×64 patch, same LM settings, same saccade privilege — only
the coordinate choice differed between our log-polar eye and a
faithful stand-in for the reference CameraSM (3D metric locations,
normal/curvature pose, HSV). Why a stand-in and not the class
itself: `ObservationProcessor.process` is welded to the transform
stack's observation shape (`semantic_3d`, `sensor_frame_data`,
`cam_to_world`) and reads the simulator's *semantic channel* for
on-object — an oracle our sensors don't get. The stand-in
reimplements its percept recipe on raw rgba+depth with the same
walk ours uses. Known limitation, stated plainly: we have not
benchmarked the stand-in against the original on their stock rig;
until then, "their eye" means "their percept recipe on our skull."

Audit trail, in order: pose sampling verified stock
(`initial_possible_poses='informed'`; `max_nneighbors` raised 3→10
*in their favor*). HSV tolerance verified stock verbatim. Feature
weights found to be OURS — at their stock `[2, 0.5, 0.5]` a false
identification we'd initially reported **vanished**, and we retracted
it on the figure itself. Final, fully stock: **ours 12/12, theirs
5/12, zero false IDs on either side**, across in-plane scale,
rotation, their composition, and out-of-plane tilt to 60° (which was
pre-registered as the metric eye's home turf; it refused all four
tilt cells instead).

The claim that survives: *the coordinate choice is the eye.* One
remap buys translation, scale, and rotation invariance, certified by
the learning module rather than by patch correlation.

## 3. Then we made it refuse, and found the real problem

A perfect score on a test that cannot fail is not evidence of
honesty (we learned this on saxophones). So: three foils the system
never learned — bowl (round, mug-family), apple (mug-red, round),
wrench (elongated). Refusal is the only honest verdict.

Three findings, each a receipt against the verdict machinery:

**Finding 1 — the quorum counts confidence, not consensus.**
On the silent bowl, the eye matched "mug" and touch matched
"banana". Stock `AnyLMsMatch(count=2)` declared MATCH. Two witnesses
naming different suspects convicted. The criterion never asks what
the LMs are confident *about*.

**Finding 2 — first-winner termination forecloses slow evidence.**
Verdicts gate at `min_eval_steps`; fast senses adjourn the episode
at the gate. Our ear could only identify the mug when run alone —
its pose-ambiguous drone needs the symmetry route, and in mixed runs
zero symmetry events fire before the meeting ends. Episodes that
*stop at the first confident answer* structurally cannot hear
dissent that takes longer than confidence.

**Finding 3 — absence has no vote.**
The apple: round to the eye, round to the finger — both matched
"mug", honest cross-modal agreement on a wrong answer. The only
witness against it was the ear, and the ear had nothing to say,
because a sense that detects nothing salient sends nothing at all.
But the mug hypothesis *predicts a 147 Hz tower*, and the ear was
listening to a scene that did not contain one. That is testimony,
and the machinery had no way to hear it.

## 4. The fixes (both are subclass-sized)

**Consensus** (`audiomonty/consensus.py`): match ⇔ N matched LMs
name the *same* object; confident dissent blocks; a lone opinion
waits. Bowl: blocked (mug-vs-banana disagreement). Wrench: withheld
(lone opinion never seconded). Creatures: still recognized.

**Expected-signal veto** (same file): silence isn't silence — it's
no sound *compared to the sound that is present*. A functioning
sense (not dark, not numb) that has a learned graph for the
candidate, but whose evidence sits below 25% of the supporting
senses' level, testifies AGAINST. The absolute-zero version failed
first and the failure proved the thesis: even our "silent" world
gives the ear 1.08 of junk evidence off the CARFAC startup
transient. Absence is relative. Apple: ear sees 1.08 for "mug" vs
supporters' 6–11 — **vetoed**.

Scorecard across the three criteria: **3 false → 1 false → 0
false.** Costs on the table: a *muted* mug is now withheld too — a
working ear that hears a scene with no hum in it vetoes, because the
hum was part of what "mug" meant. A *dark* eye still abstains
cleanly. The taxonomy doing the work is three-way and it is the
whole point:

> **channel absent** (no data) ≠ **sensed absence** (data of no) ≠
> **sensed presence**. Conflating the first two is what let the
> apple through.

Bonus: the relative floor retired Finding 2's race for free — a slow
ear that is genuinely hearing the drone clears 25% easily and never
vetoes; only an ear hearing nothing of the sort testifies.

## 5. What we think this means for the theory

1. **Brains don't guess and stop; they guess and update.** The
   terminal state is a harness convenience doing theoretical work it
   hasn't earned. Every failure above is a symptom of stopping:
   first-winner races, dissent foreclosed, absence unheard. A
   recognizer that is not allowed to terminate (we're planning one
   as a realtime audio plugin, where the stream never ends) is
   forced to hold verdicts as decaying, dethronable state — which is
   what the theory says cortex does.
2. **The decisive sense is pair-relative.** The apple was caught by
   the ear, the bowl by touch's dissent, the wrench by the eye's
   refusal — in each case, the channel where the false hypothesis
   made its sharpest falsifiable prediction. Identity lives where
   the hypothesis sticks its neck out. This requires testing a
   hypothesis against senses that did *not* vote for it, which the
   current message flow has no path for.
3. **Voiced-vs-voiceless worlds are platonic.** Our own sonar
   campaign ranged voiceless things all night (a moth, a wall, a
   human at 23 inches) — everything scatters the acoustic
   illuminant. Ambient-noise acoustics with all objects as
   reflectors is queued; after that, the ear discriminates apple
   from mug by scattering signature instead of by veto, and the
   platonic lie is never told in the first place.
4. **Degeneracy is frame-relative and every frame has its mugs.**
   Their eye's mug (surface of revolution) and our eye's mug
   (theta-uniform ring) are the same object being maximally
   self-similar in two different coordinates. Novel blobs fall into
   the most degenerate learned object in *any* single frame — which
   is exactly why the fix has to live at the federation level, not
   in a better sensor.

## 6. Open questions we'd genuinely like answers to

- Where should expected-signal dissent live? We bolted it onto the
  termination decision because that's the socket we had. It smells
  like it belongs in the voting protocol — a vote *against* a
  hypothesis from an LM whose modality that hypothesis makes
  predictions for.
- Does anything in the roadmap give an LM a notion of absence — a
  prediction error for "the feature I expected at this location is
  not here" that carries evidence weight, not just no update?
- LM-level time is the same question wearing another hat: tempo is
  currently invisible to our audio graphs (same object at any
  speed), and absence-over-time (the rest, the gap, the mute) is
  where music keeps most of its information.

## Reproduce

```
# senses + triad
uv run python run.py experiment=triad_pretrain_2creatures
uv run python run.py experiment=triad_eval_2creatures          # 14/14 matrix via dark/mute/numb overrides
# duel
uv run python run.py experiment=cam_pretrain_2creatures
uv run python run.py experiment=cam_eval_2creatures env_interface=retina_eval_{scaled,rotated,rotscale,tilted,tilted60}
# the verdict-machinery findings and fixes
uv run python run.py experiment=triad_eval_2creatures env_interface=retina_eval_foils   # finding 1
uv run python run.py experiment=triad_eval_consensus  env_interface=retina_eval_foils   # 0 false
```

Figures: `figures/triad.png`, `figures/eye_vs_eye.png`,
`figures/where_refusal_lives.png`,
`figures/novelty_by_disagreement.png`, `figures/real_sonar.png`.
