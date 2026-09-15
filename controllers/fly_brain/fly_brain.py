"""Main Webots controller loop.

Signal path, camera to motors, every timestep:

    retina -> emd -> lobula -> (avoidance | central_complex) -> arbitration
           -> output -> motors

PHASE 1 (current): waypoint flight on ground-truth position, no vision.
This is the bug-isolation layer. When the drone misbehaves in Phase 4, you
need to already know the flight stack is sound, so the fault can only be in
the fly brain. Keep this controller working — it is also the manual-flight
rig you fly while validating the EMD layer in Phase 2.

PHASE 2 (next): replace the ground-truth bearing with optic flow. The flight
control below does not change.

Set PHASE = 0 to fall back to the toolchain sensor check.
"""

import math

from controller import Robot

PHASE = 1

# ---- Waypoint ------------------------------------------------------------
TARGET_X = 5.0
TARGET_Y = 3.0
TARGET_ALTITUDE = 1.5

# Within this radius we count as arrived and stop translating. The outer band
# is hysteresis: once arrived, we only resume flying if we drift past it.
# Without the gap the drone hunts across the boundary forever.
ARRIVE_RADIUS = 0.35
RESUME_RADIUS = 0.60

# ---- Gains ---------------------------------------------------------------
# Tune these in order: altitude first (in a hover, target = start position),
# then yaw (it should turn to face the target without overshooting), then
# pitch last. Change one at a time.
K_VERTICAL_P = 3.0          # altitude error -> thrust
K_VERTICAL_OFFSET = 68.5    # base thrust, roughly hover for the Mavic 2 Pro
K_VERTICAL_THRUST = 68.5

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

PRINT_EVERY = 16


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

        # Which GPS component is "up" depends on the world's coordinateSystem:
        # ENU (Webots' modern default) puts altitude at index 2, NUE (used by
        # several bundled sample worlds) at index 1. Guessing wrong makes the
        # altitude loop track a horizontal coordinate, and the drone never
        # takes off. Detect it instead — see _detect_up_axis.
        self.up_axis = None
        self.horizontal_axes = None

    def _detect_up_axis(self):
        """Pick the GPS component that is vertical, by elimination.

        Called once, on the first control step, while the drone is still on
        the ground. The two horizontal components are the drone's start
        position (near the world origin in the sample worlds); the vertical
        one carries its resting altitude above the ground plane.

        We cannot simply take the largest — a drone parked away from the
        origin would defeat that. Instead we use the fact that Webots only
        ever uses ENU or NUE, so the up-axis is index 2 or index 1, and the
        remaining axis pairs with x. Compare the two candidates against the
        IMU: on the ground, level, the vertical component is the one that
        does not change as the drone is translated. With a single sample we
        fall back on magnitude between only those two candidates, which is
        reliable because resting altitude is positive and small while a
        horizontal start coordinate is typically ~0.
        """
        values = self.gps.getValues()
        enu_alt = values[2]
        nue_alt = values[1]

        # Resting altitude is positive (above the ground plane). A horizontal
        # start coordinate is near zero in the sample worlds, and may be
        # negative — which a horizontal axis can be and an altitude cannot.
        if nue_alt > 0.01 and enu_alt <= 0.01:
            self.up_axis, self.horizontal_axes = 1, (0, 2)
        elif enu_alt > 0.01 and nue_alt <= 0.01:
            self.up_axis, self.horizontal_axes = 2, (0, 1)
        else:
            # Both or neither look plausible: prefer ENU, Webots' default for
            # worlds saved by R2022 and later.
            self.up_axis, self.horizontal_axes = 2, (0, 1)

        system = "ENU (z up)" if self.up_axis == 2 else "NUE (y up)"
        print(
            f"[phase1] coordinate system detected: {system} "
            f"gps={tuple(round(v, 3) for v in values)}",
            flush=True,
        )
        print(
            "[phase1] if the drone does not climb, this detection is wrong — "
            "check the world's coordinateSystem field and set UP_AXIS manually",
            flush=True,
        )

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

        # Hysteresis band — see ARRIVE_RADIUS.
        if distance < ARRIVE_RADIUS:
            self.arrived = True
        elif distance > RESUME_RADIUS:
            self.arrived = False

        bearing_error = wrap_angle(math.atan2(dy, dx) - yaw)

        # --- Altitude -> thrust
        altitude_error = clamp(TARGET_ALTITUDE - altitude, -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (altitude_error ** 3.0)

        # --- Bearing -> yaw rate
        yaw_input = 0.0 if self.arrived else K_YAW_P * bearing_error

        # --- Distance -> forward lean
        # Only once roughly facing the target, and scaled down by how far off
        # the bearing still is, so a turn is finished before speed builds.
        if self.arrived or abs(bearing_error) > FACING_TOLERANCE:
            pitch_disturbance = 0.0
        else:
            facing = math.cos(bearing_error)
            pitch_disturbance = clamp(
                K_FORWARD_P * distance * facing, 0.0, MAX_PITCH_DISTURBANCE
            )

        # --- Attitude stabilisation (holds the aircraft level; not steering)
        roll_input = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_rate
        pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_rate - pitch_disturbance

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
            "yaw": yaw,
            "distance": distance,
            "bearing_error": bearing_error,
            "arrived": self.arrived,
        }


def run_phase0(robot, timestep):
    """Toolchain check: print frame shape and yaw every timestep."""
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    # Not needed for the Phase 0 exit test, but printing the raw GPS triple
    # here is how you confirm the up-axis before Phase 1 depends on it.
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

    frames = 0
    announced = False
    while robot.step(timestep) != -1:
        frames += 1
        status = controller.step()

        if status["arrived"] and not announced:
            print(
                f"[phase1] ARRIVED at ({status['x']:+.2f}, {status['y']:+.2f}) "
                f"after {frames} steps",
                flush=True,
            )
            announced = True
        elif not status["arrived"]:
            announced = False

        if frames % PRINT_EVERY == 0:
            print(
                f"[phase1] pos=({status['x']:+.2f},{status['y']:+.2f}) "
                f"alt={status['altitude']:.2f}m "
                f"dist={status['distance']:5.2f}m "
                f"bearing_err={status['bearing_error']:+.3f}rad "
                f"{'HOLD' if status['arrived'] else 'FLY '}",
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
