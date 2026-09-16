"""Main Webots controller loop.

Signal path, camera to motors, every timestep:

    retina -> emd -> lobula -> (avoidance | central_complex) -> arbitration
           -> output -> motors

PHASE 1 (current): waypoint flight on ground-truth position, no vision.
This is the bug-isolation layer. When the drone misbehaves in Phase 4, you
need to already know the flight stack is sound, so the fault can only be in
the fly brain. Keep this controller working — it is also the manual-flight
rig you fly while validating the EMD layer in Phase 2.

DELIBERATELY MINIMAL. An earlier version layered on an arrival hysteresis
band, a closing-speed brake, and a distance-scaled bearing fade. Two of the
three introduced bugs worse than what they fixed — each suppressed a signal
rather than bounding its effect, and each suppression created a stable wrong
state (a false altitude equilibrium, a permanent orbit). They are gone.

Add complexity back only when a measured failure demands it, one term at a
time, and only after the telemetry below shows which term is at fault.

Set PHASE = 0 to fall back to the toolchain sensor check.
"""

import math

from controller import Robot

PHASE = 2

# ---- Waypoint ------------------------------------------------------------
TARGET_X = 5.0
TARGET_Y = 3.0
TARGET_ALTITUDE = 1.5

# Diagnostic mode for worlds/01c_hover.wbt. Puts the waypoint at the drone's
# own start position, so distance is ~0 from the first step: no translation,
# no forward lean, no bearing to chase, and the altitude loop is the only
# thing doing work.
#
# Every altitude measurement so far was taken while the drone was also
# translating. The equilibrium (vin ~ +0.648, alt ~ 1.01m against a 1.5m
# command) held identical across integral ceilings of 1.2, 2.5, 6.0 and 3.0,
# which says the integral is not the constraint — but that reading is only
# trustworthy with the position loop removed. This flag removes it.
HOVER_TEST = False

# Stop translating inside this radius.
#
# Measured: with this at 0.35 the drone settled into a permanent orbit at
# d=0.33-0.35 — straddling the threshold, so `arrived` flickered, and every
# flip to FLY fired yaw at full gain (yin=+-3.0) on a bearing that means
# nothing at 0.34m, kicking it sideways into the next lap. The radius has to
# sit outside the radius the drone actually holds, not on top of it.
ARRIVE_RADIUS = 0.60

# Yaw authority fades to zero as distance falls to this range, so a bearing
# that is noise-dominated close in produces a small command rather than a
# full-gain kick. This attenuates the signal's EFFECT; it never freezes the
# signal itself, which is what caused an earlier orbit on a stale heading.
YAW_FADE_START = 1.5  # metres

# REVERTED to 0.0 (no floor). Introduced at 0.35 alongside K_YAW_P 1.6->2.4
# as a single change, and that pair crashed the drone. Reverting the gain but
# keeping the floor would be an untested combination, so both go back.
#
# The problem it was aimed at is real and still open: faded to ya=0.13 the
# yaw command could not rotate the airframe, so the drone held ~71deg off
# target and circled at a fixed radius. The fix has to raise authority
# WITHOUT feeding a large yaw_input into the propeller mix — most likely by
# rotating first and gating forward drive on heading, rather than by turning
# harder while already translating.
MIN_YAW_AUTHORITY = 0.0

# Forward command retained once inside ARRIVE_RADIUS. Not zero: cutting the
# command entirely left the drone with no way to hold station, so it coasted
# out of the radius and orbited. Small enough not to overshoot back through
# the target, large enough to resist drift.
HOLD_DRIVE_SCALE = 0.35

# ---- Gains ---------------------------------------------------------------
# Tune in this order, one at a time: altitude, then yaw, then forward.
K_VERTICAL_P = 3.0          # altitude error -> thrust
# REVERTED to 68.5. Raising this to 69.3 drove the controller to command
# NEGATIVE vertical input (vin=-0.221) to fight an over-powered base.
#
# The reasoning that led there was wrong: vin=+0.648 at equilibrium looked
# like evidence the base was too low, but that was the equilibrium of a
# SATURATED integral. A saturated integral tells you nothing about the base
# — only that the integral could not do its job.
K_VERTICAL_THRUST = 68.5

# A proportional term on a cubic error cannot close a steady-state offset:
# thrust and gravity reach equilibrium below target and stay there (measured
# at 0.90m against a 1.5m command). The integral trims that residual.
#
# Windup is bounded by the clamp plus a saturation back-off — never by gating
# where the term may act. Gating it produced false equilibria twice.
# Altitude is a PID, and the D term is what makes it settle rather than
# oscillate. History, all measured:
#
#   clamp 1.2 -> integral pinned every sample, alt stuck at 1.08m
#   clamp 4.0 -> sustained limit cycle, alt swinging 1.0-2.3m for 515 lines,
#                integral swinging 0.6-4.0, 27 ARRIVED events, no decay
#
# Both are the same loop. A PI controller on a second-order plant (thrust ->
# acceleration -> velocity -> altitude) oscillates; the 1.2 clamp was merely
# suppressing the swing by starving it, which read as a steady-state offset.
# The giveaway in the second run: I peaked at +3.065 while aerr was +0.001 —
# maximum authority at zero error, a quarter cycle out of phase, which is an
# undamped integrator.
#
# The D term opposes vertical RATE, so it brakes the drone as it approaches
# the target altitude instead of sailing through it.
# With the D term damping the loop, the ceiling can rise without bringing
# the oscillation back. Measured at clamp 2.5: I pinned at +2.500 on every
# sample with aerr steady at +0.367 and vrate at 0.00 — saturated and
# perfectly damped, so the ceiling was the only thing capping authority.
# THE CEILING WAS NEVER THE CONSTRAINT. Three clamp settings — 1.2, 2.5 and
# 6.0 — all produced the SAME equilibrium output, vin=+0.648, with the
# integral pinned and aerr steady near +0.25. An integral that saturates at
# every ceiling and still cannot close the error is not short of headroom:
# something downstream of it is capping the thrust.
#
# That something is K_VERTICAL_THRUST. Base 68.5 plus vin lands at T=69.15,
# and 69.15 is simply not enough thrust to hold 1.5m on this airframe — the
# drone settles wherever 69.15 happens to balance gravity, which is ~1.25m.
# The integral was compensating for a base thrust set too low, which is not
# an integral's job.
#
# Raise the base to where hover actually is, and let I do what it is for:
# trimming a small residual, not supplying the bulk of the lift.
K_VERTICAL_I = 0.10
K_VERTICAL_D = 3.0
MAX_VERTICAL_INTEGRAL = 3.0

K_ROLL_P = 50.0             # attitude stabilisation, not steering
K_PITCH_P = 30.0

# REVERTED to 1.6. Raising this to 2.4 (with a 0.35 authority floor) crashed
# the drone: yin reached +5.602 and the aircraft went down at (4.21, 0.85),
# alt=0.02. yaw_input feeds the propeller mix directly — it subtracts from
# two rotors and adds to the other two — so a large yaw command tips the
# airframe rather than just rotating it. The heading-stall it was meant to
# fix is real, but the cure has to come from somewhere other than raw gain.
K_YAW_P = 1.6               # bearing error -> yaw rate
K_FORWARD_P = 0.22          # distance -> pitch (forward lean)

# Lean angle is what actually limits speed. Too much and the camera points at
# the floor, which matters from Phase 2 onward.
MAX_PITCH_DISTURBANCE = 0.6

# TURN/CRUISE thresholds, in radians. Hysteresis: enter CRUISE only when
# well aligned, drop back to TURN only when badly misaligned. A single
# threshold makes the drone chatter between states on bearing noise.
#
# Replaces the old FACING_TOLERANCE gate, which had no hysteresis and, more
# importantly, no way to ever CLOSE a heading error — it merely refused to
# translate while misaligned, so a drone yawed 71deg off target sat there
# indefinitely with its drive gated to zero.
TURN_ENTER = 0.35   # above this misalignment, stop and turn
TURN_EXIT = 0.12    # below this, resume cruising

# The Mavic worlds use basicTimeStep 8, so the loop runs at 125Hz.
PRINT_EVERY = 125

# ---- Startup guard -------------------------------------------------------
# Webots writes the drone's CURRENT position into the .wbt when the world is
# saved, so saving mid-flight makes the next run start wherever the drone
# happened to be. That has happened four times, and each time the run tested
# nothing while looking superficially normal. Refuse to fly rather than
# produce another misleading log.
EXPECTED_START = (0.0, 0.0)
START_TOLERANCE = 1.0  # metres


def wrap_angle(radians):
    """Wrap to [-pi, pi], so an error of 359 degrees reads as -1 degree.

    A naive subtraction of two headings makes the drone turn the long way
    round the circle. Same circular-difference logic as
    central_complex.goal_error.
    """
    return math.atan2(math.sin(radians), math.cos(radians))


def clamp(value, low, high):
    return max(low, min(high, value))


class Phase1Controller:
    """Waypoint flight from ground-truth position.

    Three independent loops, deliberately not coupled:
      altitude -> thrust
      bearing  -> yaw rate
      distance -> pitch (forward lean)

    Keeping them separate is what makes the gains tunable one at a time.
    """

    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep

        self.imu = robot.getDevice("inertial unit")
        self.imu.enable(timestep)
        self.gps = robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(timestep)

        self.camera = robot.getDevice("camera")
        self.camera.enable(timestep)

        self.motors = []
        for name in (
            "front left propeller",
            "front right propeller",
            "rear left propeller",
            "rear right propeller",
        ):
            motor = robot.getDevice(name)
            motor.setPosition(float("inf"))  # velocity control
            motor.setVelocity(1.0)
            self.motors.append(motor)

        self.arrived = False
        self.altitude_integral = 0.0

        # TURN/CRUISE state. Starts True so the drone squares up to the target
        # before moving, whatever heading the world gave it.
        self.turning = True

        # HOVER_TEST waypoint, captured on the first step rather than here:
        # the GPS reads nothing until after the first robot.step(), the same
        # ordering constraint the up-axis detection has.
        self.hover_x = 0.0
        self.hover_y = 0.0
        self.hover_captured = False

        # Previous altitude, for the vertical-rate estimate the D term needs.
        # None until the first step has run.
        self.prev_altitude = None

        # Which GPS component is "up" depends on the world's coordinateSystem:
        # ENU (Webots' modern default) puts altitude at index 2, NUE (used by
        # several bundled sample worlds) at index 1. Detected on first step.
        self.up_axis = None
        self.horizontal_axes = None

    def _detect_up_axis(self):
        """Pick the GPS component that is vertical, by elimination.

        Webots only ever uses ENU or NUE, so the up-axis is index 2 or 1.
        On the ground the vertical component holds a small positive resting
        altitude, while a horizontal start coordinate is near zero.
        """
        values = self.gps.getValues()
        enu_alt = values[2]
        nue_alt = values[1]

        if nue_alt > 0.01 and enu_alt <= 0.01:
            self.up_axis, self.horizontal_axes = 1, (0, 2)
        elif enu_alt > 0.01 and nue_alt <= 0.01:
            self.up_axis, self.horizontal_axes = 2, (0, 1)
        else:
            self.up_axis, self.horizontal_axes = 2, (0, 1)

        system = "ENU (z up)" if self.up_axis == 2 else "NUE (y up)"
        print(
            f"[phase1] coordinate system: {system} "
            f"gps={tuple(round(v, 3) for v in values)}",
            flush=True,
        )

    def check_start_position(self):
        """Return an error string if the world was saved mid-flight."""
        gps = self.gps.getValues()
        x = gps[self.horizontal_axes[0]]
        y = gps[self.horizontal_axes[1]]
        drift = math.hypot(x - EXPECTED_START[0], y - EXPECTED_START[1])
        if drift > START_TOLERANCE:
            return (
                f"start position is ({x:+.2f}, {y:+.2f}), "
                f"{drift:.2f}m from the expected origin"
            )
        return None

    def step(self):
        """One control cycle. Returns a status dict for logging."""
        if self.up_axis is None:
            self._detect_up_axis()

        roll, pitch, yaw = self.imu.getRollPitchYaw()
        gps = self.gps.getValues()
        x = gps[self.horizontal_axes[0]]
        y = gps[self.horizontal_axes[1]]
        altitude = gps[self.up_axis]
        roll_rate, pitch_rate, _yaw_rate = self.gyro.getValues()

        # In HOVER_TEST the waypoint is the drone's own start position, so the
        # position loop contributes nothing and only altitude is under test.
        # Captured here rather than in __init__ because the GPS reads nothing
        # until after the first robot.step().
        if HOVER_TEST and not self.hover_captured:
            self.hover_x = x
            self.hover_y = y
            self.hover_captured = True
            print(
                f"[phase1] hover waypoint pinned at "
                f"({self.hover_x:+.2f}, {self.hover_y:+.2f})",
                flush=True,
            )

        target_x = self.hover_x if HOVER_TEST else TARGET_X
        target_y = self.hover_y if HOVER_TEST else TARGET_Y

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        self.arrived = distance < ARRIVE_RADIUS

        bearing_error = wrap_angle(math.atan2(dy, dx) - yaw)

        # --- Altitude -> thrust (PI)
        altitude_error = clamp(TARGET_ALTITUDE - altitude, -1.0, 1.0)

        # Integrate always; bound the term, never gate where it may act.
        # Back off only when already saturated in the error's direction —
        # unlike a gate or a decay, that cannot invent an equilibrium.
        integral_step = altitude_error * (self.timestep / 1000.0)
        at_high = self.altitude_integral >= MAX_VERTICAL_INTEGRAL
        at_low = self.altitude_integral <= -MAX_VERTICAL_INTEGRAL
        if not (at_high and integral_step > 0) and not (at_low and integral_step < 0):
            self.altitude_integral = clamp(
                self.altitude_integral + integral_step,
                -MAX_VERTICAL_INTEGRAL,
                MAX_VERTICAL_INTEGRAL,
            )

        # Vertical rate from GPS altitude deltas. Opposing it damps the loop:
        # without this the integral reaches peak authority at zero error and
        # the altitude cycles indefinitely (measured, 1.0-2.3m, 515 lines).
        if self.prev_altitude is None:
            vertical_rate = 0.0
        else:
            vertical_rate = (altitude - self.prev_altitude) / (
                self.timestep / 1000.0
            )
        self.prev_altitude = altitude

        vertical_input = (
            K_VERTICAL_P * (altitude_error ** 3.0)
            + K_VERTICAL_I * self.altitude_integral
            - K_VERTICAL_D * vertical_rate
        )

        # --- Bearing -> yaw rate
        # Fade authority as distance shrinks. Bearing is computed fresh every
        # step and never frozen — an earlier version froze it close in and the
        # drone orbited on a stale heading. Here the SIGNAL stays live and only
        # its authority is attenuated, so a meaningless close-range bearing
        # produces a small command instead of a full-gain kick.
        # Yaw stays live in HOLD. Zeroing it on arrival meant the drone could
        # not correct heading while station-keeping, and the fade below
        # already attenuates the noisy close-range bearing to near nothing.
        # The floor matters: faded to 0.13 the command could not rotate the
        # airframe at all, so the drone sat 71deg off target indefinitely.
        # Attenuate the noisy close-range bearing, but never below the
        # authority needed to actually turn.
        yaw_authority = clamp(
            distance / YAW_FADE_START, MIN_YAW_AUTHORITY, 1.0
        )
        yaw_input = K_YAW_P * bearing_error * yaw_authority

        # --- Distance -> forward lean
        # Station-keeping matters as much as the approach. Measured: cutting
        # the forward command to zero on arrival left the drone with NO way
        # to hold position — it coasted outward until d exceeded the arrival
        # radius, flipped back to FLY, took a yaw kick, and orbited. 27
        # arrivals in one run, none of them stable.
        #
        # So keep steering at the target inside the radius, just gently: the
        # command stays proportional to distance and is simply scaled down.
        #
        # The facing gate applies only OUTSIDE the arrival radius. Measured:
        # with the gate active in HOLD the drone sat yawed ~72deg off target
        # (brg pinned at +1.252 > FACING_TOLERANCE), so the drive was gated to
        # zero every step and the station-keeping never ran at all. Close in
        # the gate is pointless anyway — cos(bearing) already scales the
        # drive, and at 0.2m there is no curved-path problem to prevent.
        # TURN before CRUISE, with hysteresis between them.
        #
        # yaw_input enters all four rotors with alternating sign, so it is the
        # one term that directly fights the roll/pitch stabilisation: a large
        # yaw command unbalances the diagonal pairs faster than K_ROLL_P and
        # K_PITCH_P can correct, which is why raising K_YAW_P tipped the
        # aircraft (yin +5.602, crashed at (4.21, 0.85)).
        #
        # So the fix is WHEN yaw acts, not how hard. Turning with zero forward
        # lean lets the heading converge while the airframe is otherwise
        # settled; cruising then starts with bearing_error already small, so
        # yaw_input stays naturally tiny and never destabilises anything.
        #
        # Two thresholds, not one: entering CRUISE needs a tight heading, but
        # leaving it needs a loose one, or the drone chatters between states
        # on bearing noise mid-flight.
        # The turn logic runs in HOLD too. Forcing turning=False on arrival
        # locked the drone out of the very fix it needed: it sat 71deg off
        # target (brg pinned at +1.245) with no way to square up, because the
        # only code that closes a heading error was disabled exactly where the
        # error persisted. Station-keeping still runs while it turns — the
        # drive is scaled by cos(bearing_error), which handles the rest.
        if self.turning:
            if abs(bearing_error) < TURN_EXIT:
                self.turning = False
        elif abs(bearing_error) > TURN_ENTER:
            self.turning = True

        # Turning suppresses forward lean only OUTSIDE the arrival radius,
        # where the point is to finish rotating before building speed. In
        # HOLD it must not: the drone would lose station-keeping the moment
        # it began squaring up, and drift — the same stranding that forcing
        # turning=False on arrival used to cause, arriving by another route.
        # cos(bearing_error) already shrinks the drive while badly aligned.
        if self.turning and not self.arrived:
            pitch_disturbance = 0.0
        else:
            drive = K_FORWARD_P * distance * math.cos(bearing_error)
            if self.arrived:
                drive *= HOLD_DRIVE_SCALE
            pitch_disturbance = clamp(drive, 0.0, MAX_PITCH_DISTURBANCE)

        # --- Attitude stabilisation (holds the aircraft level; not steering)
        roll_input = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_rate
        pitch_input = (
            K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_rate - pitch_disturbance
        )

        # --- Mix to four propellers
        base = K_VERTICAL_THRUST + vertical_input
        fl = base - roll_input + pitch_input - yaw_input
        fr = base + roll_input + pitch_input + yaw_input
        rl = base - roll_input - pitch_input + yaw_input
        rr = base + roll_input - pitch_input - yaw_input

        self.motors[0].setVelocity(fl)
        self.motors[1].setVelocity(-fr)
        self.motors[2].setVelocity(-rl)
        self.motors[3].setVelocity(rr)

        return {
            "x": x,
            "y": y,
            "altitude": altitude,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "distance": distance,
            "bearing_error": bearing_error,
            "arrived": self.arrived,
            "turning": self.turning,
            # Internal terms, so the next failure is diagnosable without
            # guessing which one is at fault.
            "alt_err": altitude_error,
            "integral": self.altitude_integral,
            "vertical_input": vertical_input,
            "vertical_rate": vertical_rate,
            "yaw_input": yaw_input,
            "yaw_authority": yaw_authority,
            "pitch_disturbance": pitch_disturbance,
            "thrust": base,
        }


def run_phase0(robot, timestep):
    """Toolchain check: print frame shape and yaw every timestep."""
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    gps = robot.getDevice("gps")
    if gps is not None:
        gps.enable(timestep)

    print(f"[phase0] timestep={timestep}ms", flush=True)
    print(f"[phase0] camera {camera.getWidth()}x{camera.getHeight()}", flush=True)
    if gps is not None:
        raw = gps.getValues()
        print(
            f"[phase0] gps raw={tuple(round(v, 3) for v in raw)} "
            "— the component holding resting altitude is the up-axis",
            flush=True,
        )

    frames = 0
    while robot.step(timestep) != -1:
        frames += 1
        image = camera.getImage()
        if image is None:
            continue
        shape = (camera.getHeight(), camera.getWidth(), 4)
        _roll, _pitch, yaw = imu.getRollPitchYaw()
        yaw_rate = gyro.getValues()[2]
        if frames % PRINT_EVERY == 0:
            print(
                f"[phase0] frame={frames:6d} shape={shape} bytes={len(image)} "
                f"yaw={yaw:+.3f}rad yaw_rate={yaw_rate:+.3f}rad/s",
                flush=True,
            )


def run_phase1(robot, timestep):
    """Waypoint flight to a hardcoded coordinate."""
    controller = Phase1Controller(robot, timestep)

    print(f"[phase1] timestep={timestep}ms", flush=True)
    print(
        f"[phase1] target=({TARGET_X}, {TARGET_Y}) altitude={TARGET_ALTITUDE}m",
        flush=True,
    )
    if HOVER_TEST:
        print(
            "[phase1] HOVER_TEST: waypoint pinned to the start position — "
            "altitude loop under test in isolation, no translation expected",
            flush=True,
        )

    # One step to populate the sensors before reading the start position.
    if robot.step(timestep) == -1:
        return
    controller.step()

    problem = controller.check_start_position()
    if problem is not None:
        print("", flush=True)
        print("=" * 68, flush=True)
        print(f"[phase1] REFUSING TO FLY: {problem}.", flush=True)
        print("", flush=True)
        print("The world was saved while the simulation was running, which", flush=True)
        print("wrote the drone's then-current position into 01_empty.wbt.", flush=True)
        print("A run from here tests nothing.", flush=True)
        print("", flush=True)
        print("  Fix:  git checkout worlds/01_empty.wbt", flush=True)
        print("  Then: revert the world in Webots (Ctrl+Shift+R)", flush=True)
        print("", flush=True)
        print("When Webots asks to save on close, answer DISCARD.", flush=True)
        print("=" * 68, flush=True)
        print("", flush=True)
        return

    print("[phase1] start position OK", flush=True)
    print(
        "[phase1] columns: pos alt dist bearing | "
        "alt_err integral vert_in yaw_in yaw_auth pitch_dist thrust",
        flush=True,
    )

    frames = 0
    announced = False
    while robot.step(timestep) != -1:
        frames += 1
        s = controller.step()

        if s["arrived"] and not announced:
            print(
                f"[phase1] ARRIVED at ({s['x']:+.2f}, {s['y']:+.2f}) "
                f"after {frames} steps",
                flush=True,
            )
            announced = True
        elif not s["arrived"]:
            announced = False

        if frames % PRINT_EVERY == 0:
            print(
                f"[phase1] ({s['x']:+.2f},{s['y']:+.2f}) "
                f"alt={s['altitude']:.2f} "
                f"d={s['distance']:5.2f} "
                f"brg={s['bearing_error']:+.3f} "
                f"{'HOLD' if s['arrived'] else ('TURN' if s['turning'] else 'CRUZ')} | "
                f"aerr={s['alt_err']:+.3f} "
                f"I={s['integral']:+.3f} "
                f"vin={s['vertical_input']:+.3f} "
                f"vrate={s['vertical_rate']:+.2f} "
                f"yin={s['yaw_input']:+.3f} "
                f"ya={s['yaw_authority']:.2f} "
                f"pd={s['pitch_disturbance']:.3f} "
                f"T={s['thrust']:.2f}",
                flush=True,
            )


def run_phase2(robot, timestep):
    """Full pipeline: vision, avoidance, navigation, compliance, failsafes.

    Phase 1 above is kept deliberately and stays selectable. It is the only
    flight mode with a known-good history, so it is the fallback if this
    misbehaves, and the comparison baseline when tuning.
    """
    import numpy as np

    from config import get_config
    from pipeline import FlyBrainPipeline

    cfg = get_config()

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    motors = []
    for name in (
        "front left propeller",
        "front right propeller",
        "rear left propeller",
        "rear right propeller",
    ):
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(1.0)
        motors.append(motor)

    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([TARGET_X, TARGET_Y]))

    print(f"[phase2] timestep={timestep}ms", flush=True)
    print(
        f"[phase2] flow front-end: {cfg.optical_flow.method}", flush=True
    )
    print(
        f"[phase2] goal=({TARGET_X}, {TARGET_Y}) "
        f"altitude={cfg.flight.target_altitude}m "
        f"ceiling={cfg.altitude_cap.ceiling_m}m",
        flush=True,
    )

    dt = timestep / 1000.0
    last_motors = np.full(4, 68.5, dtype=np.float64)
    previous_position = None
    frames = 0

    # One step so the sensors read before the first pipeline call.
    if robot.step(timestep) == -1:
        return

    while robot.step(timestep) != -1:
        frames += 1

        raw = camera.getImage()
        if raw is None:
            continue
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (camera.getHeight(), camera.getWidth(), 4)
        )

        gps_values = gps.getValues()
        position = np.array([gps_values[0], gps_values[1]], dtype=np.float64)
        altitude = float(gps_values[2])

        roll, pitch, yaw = imu.getRollPitchYaw()
        rates = gyro.getValues()
        roll_rate = float(rates[0])
        pitch_rate = float(rates[1])
        yaw_rate = float(rates[2])

        if previous_position is None:
            actual_velocity = np.zeros(2, dtype=np.float64)
        else:
            actual_velocity = (position - previous_position) / dt
        previous_position = position.copy()
        speed = float(np.linalg.norm(actual_velocity))

        state = pipeline.step(
            frame=frame,
            position=position,
            altitude=altitude,
            yaw=yaw,
            yaw_rate=yaw_rate,
            speed=speed,
            acceleration=np.array([0.0, 0.0, 9.81]),
            commanded_velocity=np.zeros(2, dtype=np.float64),
            actual_velocity=actual_velocity,
            motor_feedback=last_motors,
            last_motors=last_motors,
            dt=dt,
            roll=roll,
            pitch=pitch,
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
        )

        commands = state.motors.motors
        motors[0].setVelocity(float(commands[0]))
        motors[1].setVelocity(float(-commands[1]))
        motors[2].setVelocity(float(-commands[2]))
        motors[3].setVelocity(float(commands[3]))
        last_motors = commands.copy()

        if frames % PRINT_EVERY == 0:
            channels = state.channels
            print(
                f"[phase2] ({position[0]:+.2f},{position[1]:+.2f}) "
                f"alt={altitude:.2f} "
                f"yaw={yaw:+.2f} | "
                f"L={channels.left_expansion:+.3f} "
                f"R={channels.right_expansion:+.3f} "
                f"rot={channels.rotation:+.3f} "
                f"urg={state.avoidance.urgency:.2f} "
                f"aw={state.arbitration.avoidance_weight:.2f} | "
                f"batt={state.battery_state.charge_pct:.0f}% "
                f"{state.override_reason or 'mission'}",
                flush=True,
            )

    print("", flush=True)
    print(pipeline.telemetry.format_report(), flush=True)


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    if PHASE == 0:
        run_phase0(robot, timestep)
    elif PHASE == 2:
        run_phase2(robot, timestep)
    else:
        run_phase1(robot, timestep)


if __name__ == "__main__":
    main()
