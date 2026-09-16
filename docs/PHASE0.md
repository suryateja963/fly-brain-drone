# Phase 0 — Toolchain

Goal: prove you can read real sensor data from your own Python controller.
Nothing else. No neural code until this passes.

## 1. Install Webots

Download the Windows installer from [cyberbotics.com](https://cyberbotics.com/)
and run it. Native install — no WSL, no source build, no Visual Studio.

## 2. Python environment

Already done if `.venv` exists in the repo root. Otherwise:

```sh
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Point Webots at the interpreter

*Tools → Preferences → General → Python command*, set to the absolute path:

```
D:\Fly-Brain Drone\.venv\Scripts\python.exe
```

Webots ships its own `controller` module and injects it on the path when it
launches a controller. That is why `fly_brain.py` cannot be run directly from
a terminal — it only imports inside Webots.

## 4. Sanity-check the install before your code is involved

*File → Open Sample World → robots → dji → mavic → mavic2pro.wbt*, then press
play. Fly it with the stock controller using the arrow keys. If this does not
work, the problem is the install, not your code.

## 5. Run your controller

**Open `worlds/01_empty.wbt` from this repository** — *File → Open World*,
then browse to `D:\Fly-Brain Drone\worlds\01_empty.wbt`.

That world already has `controller "fly_brain"` set on the drone, so there is
nothing to select and nothing to save. Press play and it runs.

### Why not the bundled sample world

Webots only offers controllers belonging to the *same project* as the open
world. A project is any directory holding `worlds/` and `controllers/`
side by side. The bundled `mavic_2_pro.wbt` lives inside the Webots install
tree, whose project contains only `mavic2pro` and `mavic2pro_patrol` — so
`fly_brain` never appears in its dropdown, and cannot be selected there.

This repository is its own project: `worlds/` and `controllers/` sit beside
each other at the root, which is what makes `fly_brain` resolvable.

## Exit test

The console prints, every 16th timestep:

```
[phase0] timestep=32ms
[phase0] camera 400x240
[phase0] frame=    16 shape=(240, 400, 4) bytes=384000 yaw=+0.001rad yaw_rate=+0.000rad/s
```

Yaw must **change when you rotate the drone**. A frozen value means the IMU is
enabled but not being read correctly — fix that now, because every later phase
trusts it.

## If it fails

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: controller` | Running outside Webots, or the Python command path is wrong |
| `getDevice() returned None` | Device name differs in your world — check the scene tree for exact names |
| No output at all | Controller field not set to `fly_brain`, or world not reloaded |
| `bytes=0` or `image is None` | Camera not enabled, or enabled with a zero timestep |

Device names in the snippet above are the Mavic 2 Pro's (`camera`,
`inertial unit`, `gyro`). If you build your own robot, they will differ.
