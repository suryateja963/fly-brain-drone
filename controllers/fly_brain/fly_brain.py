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

PHASE = 1

# ---- Waypoint ------------------------------------------------------------
TARGET_X = 5.0
TARGET_Y = 3.0
TARGET_ALTITUDE = 1.5

# Stop translating inside this radius. No outer resume band: a single
# threshold is simpler, and the orbiting it was meant to prevent turned out
# to be caused by a frozen bearing, not by boundary hunting.
ARRIVE_RADIUS = 0.35

# ---- Gains ---------------------------------------------------------------
# Tune in this order, one at a time: altitude, then yaw, then forward.
K_VERTICAL_P = 3.0          # altitude error -> thrust
K_VERTICAL_THRUST = 68.5    # base thrust, roughly hover for the Mavic 2 Pro

# A proportional term on a cubic error cannot close a steady-state offset:
# thrust and gravity reach equilibrium below target and stay there (measured
# at 0.90m against a 1.5m command). The integral trims that residual.
#
# Windup is bounded by the clamp plus a saturation back-off — never by gating
# where the term may act. Gating it produced false equilibria twice.
K_VERTICAL_I = 0.35
MAX_VERTICAL_INTEGRAL = 1.2

K_ROLL_P = 50.0             # attitude stabilisation, not steering
K_PITCH_P = 30.0

K_YAW_P = 1.6               # bearing error -> yaw rate
K_FORWARD_P = 0.22          # distance -> pitch (forward lean)

# Lean angle is what actually limits speed. Too much and the camera points at
# the floor, which matters from Phase 2 onward.
MAX_PITCH_DISTURBANCE = 0.6

# Don't translate until roughly facing the target, or the drone crabs
# sideways along a curved path instead of flying the bearing.
FACING_TOLERANCE = 0.5  # radians

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

        dx = TARGET_X - x
        dy = TARGET_Y - y
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

        vertical_input = (
            K_VERTICAL_P * (altitude_error ** 3.0)
            + K_VERTICAL_I * self.altitude_integral
        )

        # --- Bearing -> yaw rate
        yaw_input = 0.0 if self.arrived else K_YAW_P * bearing_error

        # --- Distance -> forward lean
        if self.arrived or abs(bearing_error) > FACING_TOLERANCE:
            pitch_disturbance = 0.0
        else:
            pitch_disturbance = clamp(
                K_FORWARD_P * distance * math.cos(bearing_error),
                0.0,
                MAX_PITCH_DISTURBANCE,
            )

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
            # Internal terms, so the next failure is diagnosable without
            # guessing which one is at fault.
            "alt_err": altitude_error,
            "integral": self.altitude_integral,
            "vertical_input": vertical_input,
            "yaw_input": yaw_input,
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
        "alt_err integral vert_in yaw_in pitch_dist thrust",
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
                f"{'HOLD' if s['arrived'] else 'FLY '} | "
                f"aerr={s['alt_err']:+.3f} "
                f"I={s['integral']:+.3f} "
                f"vin={s['vertical_input']:+.3f} "
                f"yin={s['yaw_input']:+.3f} "
                f"pd={s['pitch_disturbance']:.3f} "
                f"T={s['thrust']:.2f}",
                flush=True,
            )


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    if PHASE == 0:
        run_phase0(robot, timestep)
    else:
        run_phase1(robot, timestep)


if __name__ == "__main__":
    main()
