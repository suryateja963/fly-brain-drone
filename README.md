# Fly-Brain Drone

An autonomous drone agent whose control architecture is derived from the
*Drosophila* central brain. Runs entirely in simulation. No training, no
dataset, no GPU required.

The fly's obstacle-avoidance circuit is not learned — it is genetically wired.
Reproducing it means writing arithmetic, not running a training loop.

## Architecture

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

| Module | Fly structure | Function |
|---|---|---|
| Retina | Ommatidia | Coarse, blurry sampling |
| EMD layer | Medulla | Local motion detection |
| Wide-field pooling | Lobula plate tangential cells | Collapse flow to few channels |
| Avoidance | Innate reflex pathway | Steer from high flow |
| Ring attractor | Ellipsoid body / protocerebral bridge | Heading estimate |
| Goal vector | Fan-shaped body | Where we want to go |
| Bottleneck | Descending neurons | Narrow command channel |

The scene is rendered realistically for the viewer. The agent sees a
deliberately degraded version — small, blurry, greyscale. Fly vision is
genuinely low-resolution, and optic flow is more tractable on blurred input
because fine texture creates aliasing in the correlators. Render beautifully,
perceive crudely.

## Status

Scaffold. Modules are stubs; see the phase table below for what is implemented.

| Phase | Deliverable | Exit test | Done |
|---|---|---|---|
| 0 | Toolchain | Read camera + IMU from Python | [ ] |
| 1 | Waypoint flight | Reaches coordinate, empty world | [ ] |
| 2 | EMD layer | Flow visibly correct on gratings and live | [ ] |
| 3 | Pooling | Four channels respond correctly | [ ] |
| 4 | Avoidance | Survives corridor with no goal | [ ] |
| 5 | Ring attractor | Bump tracks yaw | [ ] |
| 6 | Arbitration | Reaches target through obstacles | [ ] |
| 7 | Bottleneck | Four-value interface enforced | [ ] |
| 8 | Demo | Recorded flight in pillar forest | [ ] |

**Phase 2 is the risk.** Everything after it is straightforward if the flow is
right.

## Setup

Requires [Webots](https://cyberbotics.com/) (Apache 2.0) and Python 3.

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Point Webots at the interpreter under *Tools → Preferences → Python command*.

Run the tests (these need no simulator):

```sh
pytest
```

Open a world from `worlds/` in Webots to fly.

## Layout

```
controllers/fly_brain/   the controller and its neural modules
worlds/                  Webots scenes, ascending difficulty
tests/                   standalone validation, no simulator needed
tools/                   visualisers
```

The full build spec, with per-phase exit tests and known failure modes, is in
[FLY_BRAIN_DRONE.md](FLY_BRAIN_DRONE.md).

## License

MIT — see [LICENSE](LICENSE).
