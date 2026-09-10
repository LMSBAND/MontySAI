# Recon: what the Monty tree actually asks of us (2026-09-10)

Read on day zero, before any install. Shallow clone of
thousandbrainsproject/tbp.monty lives in `tbp.monty/` (gitignored --
upstream, not ours).

## The environment surface (Stage 1 target)

`src/tbp/monty/frameworks/environments/environment.py` -- Environment is
a small Protocol, NOT a framework to inherit from:

    step(actions) -> (Observations, ProprioceptiveState)
    reset() -> (Observations, ProprioceptiveState)
    close()
    add_object(name, position, rotation, scale, ...)   # ObjectEnvironment
    remove_all_objects()

Replacing Habitat = implementing this over a socket to Godot. Precedent
in-tree: `frameworks/environment_utils/server.py` is the "Monty Meets
World" demo -- an iPad camera streamed into Monty over plain HTTP PUT.
Non-Habitat sensors are a beaten path.

## The sensor module surface (Stage 2 target)

`src/tbp/monty/frameworks/models/sensor_modules.py` -- CameraSM is the
template. The contract:

    update_state(agent)   # agent pose x sensor offset -> SensorState
    step(ctx, observation, motor_only_step) -> Message   # the percept
    reset(), state_dict()

CameraSM = ObservationProcessor (features at locations: on_object,
rgba, surface_normal, curvatures) + MessageNoise + PerceptFilter
(FeatureChangeFilter with delta thresholds) + telemetry. An AudioSM is
the same skeleton with (channel, lag) as location and ridge descriptors
as features. No audio anything exists in the tree.

## Assets already in hand (LMS repo)

- `tools/ear_rig/carfac_ref.py` -- vendored google/carfac NumPy
  reference, verified against lms_sai.jsfx. This IS the sensor math.
- `tools/ear_rig/sai_ref.py` -- SAI at plugin constants, frame
  iterator, first-peak pitch rule.
- lms_sai.jsfx -- realtime CARFAC+SAI, bit-identical schedule, note log,
  certain-only chord quality, 83/42-channel density switch.

## The theory question Stage 2 must answer (flagged, not solved)

Monty's loop is move -> location changes -> same object, new view. An
SAI ridge does NOT move when the listener walks; pitch is
source-intrinsic. The auditory object is pose-stable in its own frame.
So "movement" for the AudioSM is either (a) the pose-dependent features
around a stable frame (level, timbre-through-distance, room), or (b)
time itself -- successive strobe frames as saccades. Pick explicitly
when writing the SM; do not let it happen by accident.

## Next actions (Stage 1)

1. `uv sync` in tbp.monty (uv.lock present), run a stock experiment;
   benchmark data downloads are the expected grind.
2. Godot wilderness project: socket server emitting
   {observations, pose}, accepting {action}.
3. Python adapter class implementing the Protocol; stock recognition
   experiment with Godot as the world = Stage 1 done.

## Day zero outcome (same day, evening)

Stage-0 DONE. `uv sync --extra dev --extra simulator_mujoco` works
(their UV_PROTOTYPE.md undersells it; 5.3G venv, python 3.13).
MuJoCo unit tests 51/52 (one Hypothesis deadline flake). YCB-for-MuJoCo
data comes from the CI recipe, NOT the habitat downloader:

    mkdir -p ~/tbp/data/mujoco/objects/ycb
    curl -L https://tbp-data-public-5e789bd48e75350c.s3.us-east-2.amazonaws.com/tbp.monty/ycb_objects_1.0.tgz \
      | tar -xzf - -C ~/tbp/data/mujoco/objects/ycb

Then:

    uv run python run.py experiment=tutorial/first_experiment_mujoco

ran to completion: surface agent on a mug, graph built, model saved
under ~/tbp/results/monty/pretrained_models/my_trained_models/.

Also learned: every experiment is Hydra config groups; a Godot
environment = a `/environment: godot_*` yaml + a GodotSimulator class
implementing the same Protocol MuJoCoSimulator does (624 lines,
src/tbp/monty/simulators/mujoco/simulator.py -- THE template).

Next: the Godot side. Needs the wilderness project exhumed; then the
socket server scene + GodotSimulator adapter.
