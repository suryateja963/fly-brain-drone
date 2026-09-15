# Fly-Brain Drone — Build Guide

An autonomous drone agent whose control architecture is derived from the
*Drosophila* central brain. Runs entirely in simulation. No training, no
dataset, no GPU required.

This document is the working spec. Work through it phase by phase. Each phase
has an explicit exit test — do not move on until it passes.

---

## 0. Project constraints

These are fixed. Any proposed solution that violates one of these is wrong.

| Constraint | Value |
|---|---|
| Cost | $0. Free / open-source / local only. |
| OS | Windows (native, no WSL required) |
| GPU | RTX 5060, 8GB VRAM — do not assume more |
| Simulator | Webots (Apache 2.0) |
| Language | Python 3 |
| Sensors | Camera + IMU only |
| Training | None. Every stage is hand-wired. |
| Goal | A working demo, filmable |

**Why no training:** the fly's obstacle-avoidance circuit is not learned. It is
genetically wired. Reproducing it means writing arithmetic, not running a
training loop. This is the single biggest reason this project is finishable.

---

## 1. Architecture overview

Signal path, camera to motors, every timestep:

```
Camera frame (RGB, full res)
        |
        v
[ RETINA ]        downsample to ~40x20 greyscale, blur
        |
        v
[ MEDULLA / EMD ] Reichardt correlators -> local optic flow field
        |
        v
[ LOBULA PLATE ]  pool flow -> 4 wide-field channels
        |
        +-------------------+
        |                   |
        v                   v
[ AVOIDANCE ]        [ CENTRAL COMPLEX ]  <- IMU yaw
  reflex, hardwired    ring attractor, heading + goal vector
        |                   |
        +-------------------+
                  |
                  v
          [ ARBITRATION ]   sigmoid blend, not a hard switch
                  |
                  v
     [ DESCENDING BOTTLENECK ]  4 values only
                  |
                  v
        roll / pitch / yaw_rate / thrust
```

### Biological mapping

| Module | Fly structure | Function |
|---|---|---|
| Retina | Ommatidia | Coarse, blurry sampling |
| EMD layer | Medulla | Local motion detection |
| Wide-field pooling | Lobula plate tangential cells | Collapse flow to few channels |
| Avoidance | Innate reflex pathway | Steer from high flow |
| Ring attractor | Ellipsoid body / protocerebral bridge | Heading estimate |
| Goal vector | Fan-shaped body | Where we want to go |
| Bottleneck | Descending neurons | Narrow command channel |

### The two-camera principle

The scene is rendered realistically for the **viewer**. The agent sees a
deliberately degraded version — small, blurry, greyscale.

This is not a compromise. Fly vision is genuinely low-resolution, and optic
flow is more tractable on blurred input because fine texture creates aliasing
in the correlators. Render beautifully, perceive crudely.

It is also the best shot in the video: photoreal 3D on one side, a 40x20
motion field on the other, both driving the same flight.

---

## 2. Environment setup

### 2.1 Install Webots

Download the Windows installer from the official Cyberbotics site. Native
install — no WSL, no build from source, no Visual Studio.

### 2.2 Python

Webots runs external Python controllers. Use Python 3 with:

- `numpy` — all the neural computation
- `opencv-python` — downsampling, blur, flow visualisation
- `matplotlib` — offline plots and figures

No torch. No CUDA. Nothing in this project needs a GPU.

### 2.3 Confirm the toolchain

Open the bundled Mavic 2 Pro sample world. Fly it with the stock controller.
Confirm you can read camera frames and IMU values from Python.

**Exit test for Phase 0:** you can print the camera frame shape and the current
yaw value, every timestep, from your own Python controller.

Do not write a single line of neural code until this passes.

---

## 3. Phase 1 — Waypoint flight, no vision

Get a drone flying to a coordinate using ground-truth position. No optic flow,
no fly brain. This exists purely to prove the flight stack works and to give
you a baseline to beat.

**Build:**
- Read position and yaw
- Compute bearing to a hardcoded target
- Proportional controller on yaw and pitch
- Hold altitude

**Exit test:** drone takes off, flies to the target coordinate, hovers there,
in an empty world.

**Do not skip this.** If flight is unstable later, you need to know whether the
bug is in the fly brain or the flight control. This phase separates them.

---

## 4. Phase 2 — The EMD layer (the hard one)

This is where the project lives or dies. Reichardt detectors are easy to
implement in a way that looks plausible but points the wrong direction. Budget
real time here.

### 4.1 Retina

```
frame -> greyscale -> gaussian blur -> resize to (40, 20) -> float [0,1]
```

Keep the previous frame. The EMD needs two timepoints.

### 4.2 Reichardt correlator

For each horizontally adjacent pixel pair (A, B):

```
response = (A_delayed * B_now) - (B_delayed * A_now)
```

The delay is a low-pass filter over the previous frame, not simply the raw
previous frame. Implement it as an exponential moving average with a time
constant you can tune:

```
delayed = tau * delayed_prev + (1 - tau) * current
```

Sign convention matters. Decide which sign means rightward motion and write it
down in a comment. You will forget.

Do the same vertically for vertical flow.

### 4.3 Standalone validation — mandatory

Before connecting anything to a motor, build a test harness:

1. Feed the EMD a synthetic drifting sinusoid grating. Known direction, known
   speed. The output must have the correct sign.
2. Feed it a grating drifting the other way. Sign must flip.
3. Feed it a static image. Output must be near zero.
4. Only then feed it real camera frames while you fly the drone manually with
   the Phase 1 controller.

**Exit test:** a live side-by-side window — camera view on the left, optic
flow field as arrows or a heatmap on the right — and the flow visibly and
correctly tracks your motion. Fly forward: outward expansion. Yaw left:
uniform rightward flow.

If the flow looks noisy but directionally correct, that is fine and expected.
If it looks clean but points the wrong way, stop and fix it now.

---

## 5. Phase 3 — Wide-field pooling

Collapse the flow field into four numbers. This is the lobula plate and it is
the conceptual heart of the design: thousands of local detectors become four
channels.

| Channel | Computation |
|---|---|
| `left_expansion` | Mean outward horizontal flow, left half |
| `right_expansion` | Mean outward horizontal flow, right half |
| `vertical_flow` | Mean vertical flow, whole field |
| `rotation` | Mean uniform horizontal flow (yaw signal) |

Note that rotation and expansion are confounded: yawing produces uniform
horizontal flow across the whole field, which contaminates the expansion
estimate. Subtract the rotational component before computing expansion. The
fly does something equivalent using haltere feedback — you have an IMU, so use
the measured yaw rate to cancel it.

**Exit test:** print the four channels live. Fly toward a wall — both expansion
values climb. Fly past a pillar on the left — left expansion spikes alone.

---

## 6. Phase 4 — Avoidance reflex

Two lines of real logic. This is the bee centring response.

```
turn_command = gain * (left_expansion - right_expansion)
```

Positive difference means more flow on the left, so the obstacle is on the
left, so turn right. Plus a braking term: if total expansion exceeds a
threshold, reduce forward speed.

Hardwired. No learning. This is the innate pathway.

**Exit test:** build a corridor world with pillars. Drone flies forward, no
goal, and does not hit anything. It may wander. That is correct — it has no
goal yet.

---

## 7. Phase 5 — Central complex

A ring attractor holding heading.

### 5.1 Structure

- N units in a ring (16 or 32 is plenty)
- Local excitation: each unit excites its neighbours
- Global inhibition: all units inhibit all others
- Result: a single stable bump of activity

### 5.2 Dynamics

Each timestep, shift the bump by the IMU yaw rate. Then apply one step of the
excitation/inhibition update to re-sharpen it.

The bump position is the drone's estimated heading. It will drift — that is
biologically accurate and acceptable over short flights.

### 5.3 Goal vector

Store a target heading as a second ring, or simply as an angle. The error
signal is the circular difference between bump position and goal.

**Exit test:** visualise the ring as a bar chart. Yaw the drone manually. The
bump rotates smoothly and tracks true heading. No splitting, no fading.

---

## 8. Phase 6 — Arbitration

Two pathways compete: avoidance and goal-seeking.

```
urgency = sigmoid((total_expansion - threshold) * sharpness)
turn = urgency * avoidance_turn + (1 - urgency) * goal_turn
```

**Use a sigmoid, not an if-statement.** A hard switch produces oscillation:
the drone dodges, the obstacle leaves view, it turns back, the obstacle
returns. Smooth blending prevents this limit cycle.

This mirrors the fly: the innate pathway dominates when threat is immediate,
the goal pathway when the air is clear.

**Exit test:** obstacles placed directly between start and target. Drone
reaches the target without collisions.

---

## 9. Phase 7 — Output bottleneck

Four values only: `roll`, `pitch`, `yaw_rate`, `thrust`.

Enforce this as an actual interface — a single function that takes the
arbitration output and returns exactly four floats, with nothing else in the
codebase touching the motors. The constraint is the point. It is the
descending-neuron bottleneck, and keeping it honest keeps the architecture
honest.

---

## 10. Phase 8 — The demo

Worlds to build, in ascending difficulty:

1. Empty — sanity
2. Corridor — wall following
3. Pillar forest — the money shot
4. Target behind obstacles — full task
5. Cluttered room — stress test

### Recording

Webots has a built-in movie recorder. Capture:

- Third-person chase view of the flight
- The agent's 40x20 view, upscaled for legibility
- The optic flow field
- The four wide-field channels as live bars
- The ring attractor bump rotating

### Shorts that exist before the project is finished

- The EMD flow visualisation alone (Phase 2)
- Four numbers replacing thousands of detectors (Phase 3)
- Two lines of code dodging a pillar (Phase 4)
- The heading bump rotating (Phase 5)
- Full autonomous flight (Phase 6)

---

## 11. Known failure modes

| Symptom | Likely cause |
|---|---|
| Flow looks clean, drone turns into obstacles | EMD sign inverted |
| Flow is pure noise | Retina resolution too high, or blur too weak |
| Expansion spikes when yawing in open space | Rotation not subtracted from expansion |
| Drone oscillates near obstacles | Hard switch in arbitration; use sigmoid |
| Ring bump splits or fades | Excitation/inhibition ratio wrong |
| Heading drifts badly over time | Expected. Shorten flights or add a visual reset cue. |
| Drone dodges but never reaches target | Avoidance gain too high relative to goal gain |

---

## 12. Repository layout

```
fly-brain-drone/
  controllers/
    fly_brain/
      fly_brain.py          # main Webots controller loop
      retina.py             # downsample + blur
      emd.py                # Reichardt correlators
      lobula.py             # wide-field pooling
      central_complex.py    # ring attractor
      arbitration.py        # sigmoid blend
      output.py             # the 4-value bottleneck
  worlds/
    01_empty.wbt
    02_corridor.wbt
    03_pillars.wbt
    04_target_behind.wbt
  tests/
    test_emd_grating.py     # synthetic validation, Phase 2
    test_ring_attractor.py  # bump stability
  tools/
    visualise_flow.py       # standalone flow viewer
  README.md
```

---

## 13. Build order summary

| Phase | Deliverable | Exit test |
|---|---|---|
| 0 | Toolchain | Read camera + IMU from Python |
| 1 | Waypoint flight | Reaches coordinate, empty world |
| 2 | EMD layer | Flow visibly correct on gratings and live |
| 3 | Pooling | Four channels respond correctly |
| 4 | Avoidance | Survives corridor with no goal |
| 5 | Ring attractor | Bump tracks yaw |
| 6 | Arbitration | Reaches target through obstacles |
| 7 | Bottleneck | Four-value interface enforced |
| 8 | Demo | Recorded flight in pillar forest |

**Phase 2 is the risk.** Everything after it is straightforward if the flow is
right. Do not let it slide.

---

## 14. Later extension — the LLM layer

Not part of this build. Noted so the architecture stays compatible.

A language model cannot fly the drone. Flight control needs decisions every
10–20ms; a local LLM takes hundreds of milliseconds at best. Anything in the
control loop will hit the wall before the first token arrives.

The correct structure is two-tier:

- **LLM above the loop**, at human timescales, issuing goals: a target heading
  written into the central complex's goal vector.
- **Fly brain inside the loop**, at 50Hz, doing all flying, dodging and
  heading-holding.

To stay compatible, keep the goal heading as a single clean input to the
central complex module. That is the only seam the LLM ever touches.
