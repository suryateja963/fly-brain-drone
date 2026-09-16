# Phase 1 — Waypoint flight, no vision

Goal: the drone takes off, flies to a coordinate, and hovers there. No optic
flow, no fly brain. Ground-truth position from GPS.

**Do not skip this.** If flight is unstable in Phase 4, you need to already
know the flight stack is sound so the fault can only be in the fly brain.
This phase is what separates them.

Prerequisite: Phase 0 passing — in particular, yaw that visibly *changes*
when you rotate the drone. The yaw loop below trusts it completely.

## Run it

`PHASE = 1` at the top of `fly_brain.py` (set it to `0` to fall back to the
sensor check). Open the Mavic 2 Pro world, set the robot's `controller` field
to `fly_brain`, reload.

The world needs a `gps` device on the robot. The Mavic 2 Pro sample has one.
If yours does not, add a GPS node to the robot's `extensionSlot`.

## Expected output

```
[phase1] target=(5.0, 3.0) altitude=1.5m
[phase1] pos=(+0.00,+0.00) alt=0.42m dist= 5.83m bearing_err=+0.540rad FLY
[phase1] pos=(+1.836,+1.09) alt=1.48m dist= 3.71m bearing_err=+0.012rad FLY
[phase1] ARRIVED at (+4.81, +2.93) after 1184 steps
[phase1] pos=(+4.81,+2.93) alt=1.50m dist= 0.21m bearing_err=-0.204rad HOLD
```

## Exit test

Takes off, flies to the target, and **hovers stably for 30 s** — position
holding within roughly 0.3 m, no drift, no wobble, no slow circling.

### Do not save the world while the simulation is running

Webots writes the drone's *current* position into the `.wbt` when you save,
along with `hidden linearVelocity` / `angularVelocity` fields. Save mid-flight
and the drone's start position becomes wherever it happened to be — the next
run then begins at the target and prints `ARRIVED after 1 steps`, which tests
nothing. A save can also silently drop the floor's `appearance`, leaving an
untextured floor that generates **zero optic flow** and would send you
debugging the EMD layer in Phase 2 when the world is at fault.

If it happens: `git checkout worlds/01_empty.wbt` restores a clean start.

To reset between runs use **revert** (`Ctrl+Shift+R`), never save.

This has now happened twice. The usual cause is `Ctrl+S` out of habit, or
answering "Save" to the prompt Webots shows when you close a world or quit.
**Answer "Discard".** Nothing in the world is worth keeping between runs —
every change you want lives in the `.wbt` in git.

A run that begins with `ARRIVED after 1 steps`, or whose first printed
position is already near the target, is a clobbered world, not a result.
Restore it and re-run before reading anything into the numbers.

## Tuning order

One gain at a time, in this order. Changing two at once makes it impossible
to attribute what happened.

1. **`K_VERTICAL_I`** — set `TARGET_X/Y` to the start position so it only
   hovers, and watch the `alt=` column. It must reach `TARGET_ALTITUDE` and
   stay there. **This is the one gain that most needs tuning**: it is
   currently an untested guess.
   - settles *below* target and sits flat → too low
   - overshoots, then drifts slowly up and down → too high
2. **`K_VERTICAL_P`** — only if the climb itself is wrong. Oscillating
   rapidly means too high; a sluggish climb means too low.
3. **`K_YAW_P`** — restore the real target. It should turn to face the
   target and stop. Overshooting past the bearing and swinging back means
   too high.
4. **`K_FORWARD_P`** and **`K_BRAKE`** — last, as a pair. Overshooting the
   waypoint means `K_FORWARD_P` too high or `K_BRAKE` too low; crawling to a
   stop short of it means the reverse.

### Why altitude needs an integral

A proportional term on a cubic error **cannot** close a steady-state offset:
thrust and gravity find an equilibrium below target and stay there. The first
measured run commanded 1.5 m and sat flat at 0.90 m for the whole flight. The
integral accumulates that residual error and trims it out.

The gain is a starting guess and has never been flown. Expect to move it.

## The up-axis

Webots worlds use one of two coordinate systems, and which one decides where
altitude lives in the GPS triple:

- **ENU** (`z` up) — Webots' default for worlds saved by R2022 and later
- **NUE** (`y` up) — used by several bundled sample worlds

Reading the wrong component makes the altitude loop track a *horizontal*
coordinate: the drone never climbs, and it looks like a thrust problem.

The controller detects this on the first step and prints what it found:

```
[phase1] coordinate system detected: ENU (z up) gps=(0.0, 0.0, 0.12)
```

**Check that line against reality.** If the drone will not climb, the
detection is wrong — read the world's `coordinateSystem` field and set
`self.up_axis` (2 for ENU, 1 for NUE) and `self.horizontal_axes` by hand.

Phase 0 also prints the raw GPS triple, which is the cheapest way to confirm
the up-axis before Phase 1 depends on it: the component holding a small
positive resting altitude is the vertical one.

## Design notes

**Three independent loops.** Altitude→thrust, bearing→yaw, distance→pitch
are deliberately uncoupled, which is what makes them tunable one at a time.

**`wrap_angle`.** A bearing error of 359° is really −1°. Without wrapping,
the drone turns the long way round. Same circular-difference logic as
`central_complex.goal_error`.

**Hysteresis on arrival.** `ARRIVE_RADIUS` 0.35 m to stop, `RESUME_RADIUS`
0.60 m to start again. With a single threshold the drone hunts back and
forth across the boundary forever. This is the same reason Phase 6's
arbitration uses a sigmoid rather than an if-statement — a hard boundary in
a feedback loop produces a limit cycle.

**Facing gate.** No forward lean until within `FACING_TOLERANCE` of the
bearing, then scaled by `cos(bearing_error)`. Without it the drone crabs
sideways along a curved path instead of flying the bearing.

## If it fails

| Symptom | Cause |
|---|---|
| `alt=` settles flat, below `TARGET_ALTITUDE` | `K_VERTICAL_I` too low — a P-only loop cannot close this |
| `alt=` overshoots then drifts up and down | `K_VERTICAL_I` too high |
| Repeated `ARRIVED`, `bearing_err` winding through π | Overshooting past the target and orbiting — raise `K_BRAKE` |
| Never leaves ground, `dist` barely changes | Up-axis detected wrong — see below |
| Climbs then flips | Motor sign wrong — the diagonal pairs counter-rotate |
| Never leaves ground | `K_VERTICAL_THRUST` below hover for this airframe |
| Slow circle around the target | `K_FORWARD_P` too high relative to `K_YAW_P` |
| Turns the long way round | `wrap_angle` not applied to bearing error |
| Hunts in and out at the target | Hysteresis band too narrow |
| Drifts sideways constantly | Roll stabilisation gain too low |
| `getDevice('gps')` is None | No GPS node on the robot in this world |

## When it passes

Keep this controller. From Phase 2 you fly it manually to generate the motion
that validates the EMD layer — the flow visualisation needs real camera
movement to check against.
