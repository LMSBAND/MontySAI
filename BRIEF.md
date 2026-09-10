# Project Brief: Monty + Godot + Real-Time SAI

## Goal

Connect Numenta's **Monty** (Thousand Brains Project, `thousandbrainsproject/tbp.monty`) to a **Godot 4** environment, then add a **Lyon CARFAC / Stabilized Auditory Image (SAI)** sensor module so Monty can learn auditory "objects" in an auditory reference frame the same way it currently learns visual/tactile objects.

Thesis: the SAI is already a spatial reference frame (lag × cochlear channel), computed by the cochlea's own mechanism. Monty's sensor modules emit *feature-at-location + pose*; an SAI frame is exactly that. Monty has never had a first-class audio sensor. We're providing the one it's missing.

## Existing assets (do not rebuild)

- **Real-time CARFAC+SAI plugin** — JSFX for REAPER, trigger-at-peak strobing, CPU-optimized, output matches the `google/carfac` Python reference (reported zero difference). Can stream frames.
- **Godot 4 wilderness world** — the paused hunting game, built on real wildlife telemetry. Has terrain, physics, raycasts, animals.
- **gmem blackboard architecture** (LMS suite) — fixed-offset shared-memory bus with a TYPE_REGISTRY. Monty's Cortical Messaging Protocol (CMP) is the same idea; treat them as analogous.
- **`google/carfac` Python reference** — use for the sensor module math; the JSFX is already validated against it.

## Approach (in order — do not skip ahead)

### Stage 1 — Godot as Monty's environment (replace Habitat)
- Monty talks to its environment through a data-loader / environment interface: **observations in, actions out**.
- Build a Godot scene that:
  - exposes a movable sensor (camera patch first — matches what Monty already expects)
  - emits observations + sensor pose over a **socket** (TCP or WebSocket, JSON or msgpack)
  - accepts movement commands from Monty
- Write the Python-side environment adapter so Monty runs **unchanged** against Godot instead of Habitat.
- Success = Monty completes an existing object-recognition experiment with Godot as the world.
- Log everything (observations, poses, actions, evidence scores) to disk from day one.

### Stage 2 — SAI as a Monty sensor module
- Implement a Python `SensorModule` subclass that takes an SAI frame (channels × lags) and emits Monty's feature-at-location format.
  - "location" = (channel, lag) coordinates
  - "pose" = listener position / orientation in the Godot world
  - features = local SAI values / ridge descriptors
- Source of frames: the JSFX stream (preferred, real-time) or the `google/carfac` Python SAI on Godot's audio output (simpler first).
- Sound sources are Godot nodes; the agent "moves" by turning/walking, which changes the auditory frame — this is the sensorimotor loop.
- Success = Monty learns and recognizes a small set of sound "objects" (e.g. distinct instruments or animal calls) in the auditory frame.

### Stage 3 — Native reimplementation (ONLY if Stage 1–2 measurements justify it)
- Profile first. Port only the hot path of the learning module.
- Target: **Rust via gdext** or **C++ via GDExtension**. Not GDScript (too slow for hot loops), not Lua (no native Godot binding).

## Principles

- **Wrap first, port later.** Get the reference running in our environment, log, then rewrite what the logs say to rewrite.
- **Match the reference.** Every reimplemented stage gets diffed against the Python original on a fixed test signal; report max |Δ|.
- **Log streams, not snapshots.** Frames over time are the dataset. Godot/DAW state (which source is active, where the agent is) are free labels.
- **Keep the modules dumb and honest.** Sensor modules sense; learning modules judge. No module should be confident about something the sensor didn't support.
- **Commit by habit.** Small commits, working state always recoverable.

## First tasks

1. Clone `tbp.monty`, run an existing experiment, read the environment/data-loader interface and the sensor-module base class.
2. Sketch the socket protocol (observation schema, action schema, pose format).
3. Minimal Godot scene: one object, one movable camera sensor, socket server.
4. Python adapter → Monty runs one episode in Godot.
5. Then Stage 2.
