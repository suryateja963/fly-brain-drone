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

## Tuning order

One gain at a time, in this order. Changing two at once makes it impossible
to attribute what happened.

1. **`K_VERTICAL_P`** — set `TARGET_X/Y` to the start position so it only
   hovers. Climbs and holds altitude? Good. Oscillating up and down means
   too high; sagging means too low.
2. **`K_YAW_P`** — restore the real target. It should turn to face the
   target and stop. Overshooting past the bearing and swinging back means
   too high.
3. **`K_FORWARD_P`** — last. Too high overshoots the waypoint and flies
   back; too low crawls or never arrives.

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
