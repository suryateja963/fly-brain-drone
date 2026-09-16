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
K_VERTICAL_THRUST = 68.5    # base thrust, roughly hover for the Mavic 2 Pro

# A pure proportional term on a cubic error cannot close a steady-state
# offset: thrust and gravity reach equilibrium below target and stay there
# (measured: commanded 1.5m, settled flat at 0.90m). The integral term
# accumulates the residual error and trims it out.
# UNTUNED. The integral term is structurally right — a P-only loop on a cubic
# error cannot close a steady-state offset, which is why the measured run sat
# flat at 0.90m against a 1.5m command — but this gain is a starting guess,
# not a tuned value. It must be tuned against the real airframe:
#
#   too low  -> settles short of TARGET_ALTITUDE (the bug this replaces)
#   too high -> overshoots, then oscillates slowly about the target
#
# The clamp must stay well clear of the integral the loop needs at
# equilibrium, or it becomes the binding constraint and the drone settles
# short with the integral pinned at the clamp.
# Measured: 0.45 with a 3.0 clamp overshot to 2.29m against a 1.5m command,
# then sagged back through target and kept falling — the integral wound up
# during the climb, and had to unwind again. The clamp also let it accumulate
# far more than equilibrium needs.
# UNTUNED — third iteration, still a guess. Flown history, all at 1.5m
# commanded:
#   P-only                      -> settled 0.90m (no integral authority)
#   I=0.45, integrate always    -> peaked 2.29m, sagged (windup)
#   I=0.15, integrate in band   -> settled 0.90m (band deadlocked the term)
# Now: integrate always, bleed while far, back off when saturated.
K_VERTICAL_I = 0.35
MAX_VERTICAL_INTEGRAL = 1.2

# NOTE: INTEGRAL_BAND and INTEGRAL_DECAY are both gone. Each was an attempt
# to suppress windup by restricting the integral, and each created a false
# equilibrium where the restriction balanced the error term:
#   band gate (0.4m) -> drone pinned at 0.90m, integral never engaged
#   decay    (0.995) -> drone pinned at 1.00m, integral bled every step
# Windup is now handled solely by the saturation back-off in step(), which
# bounds the term without inventing an equilibrium of its own.

K_ROLL_P = 50.0             # attitude stabilisation, not steering
K_PITCH_P = 30.0

K_YAW_P = 1.6               # bearing error -> yaw rate
K_FORWARD_P = 0.22          # distance -> pitch (forward lean)

# Braking. Without this the drone carries its momentum past the target when
# pitch_disturbance drops to zero, overshoots, turns round, and orbits
# forever. Opposing lean proportional to closing speed stops it on the mark.
K_BRAKE = 0.9
MAX_BRAKE = 0.5

# Lean angle is what actually limits speed. Too much and the camera points at
# the floor, which matters from Phase 2 onward.
MAX_PITCH_DISTURBANCE = 0.6

# Don't translate until roughly facing the target, or the drone crabs
# sideways along a curved path instead of flying the bearing.
FACING_TOLERANCE = 0.5  # radians

# Yaw authority fades linearly to zero as distance falls to zero, reaching
# full gain at this range. Bearing is noise-dominated close in, so the
# command is attenuated rather than the signal frozen — freezing it produced
# a permanent orbit on a stale heading (see the note in step()).
#
# Must stay comfortably above RESUME_RADIUS: authority has to be near zero
# by the time the drone is close enough to leave HOLD on drift alone.
BEARING_FADE = 1.2  # metres

# The Mavic worlds use basicTimeStep 8, so the control loop runs at 125Hz.
# Printing every 16th step is 8 lines/second — faster than you can read.
# 125 gives roughly one line per second.
PRINT_EVERY = 125


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

        # Altitude PI state. The integral trims the steady-state offset a
        # proportional term alone leaves (see K_VERTICAL_I).
        self.altitude_integral = 0.0

        # Previous horizontal position, for the closing-speed estimate the
        # braking term needs. None until the first step has run.
        self.prev_position = None

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

        # Bearing is always computed fresh. An earlier version froze it inside
        # a 0.5m deadzone and reused the last value, which produced a closed
        # orbit: BEARING_DEADZONE (0.5m) was larger than RESUME_RADIUS (0.6m)
        # minus the drift, so the drone left HOLD and began translating while
        # still inside the deadzone — flying a STALE compass heading unrelated
        # to where the target actually was. Measured: bearing_err pinned at
        # -0.859 for ~40 samples while the drone traced the same loop forever.
        #
        # The real problem the deadzone was aimed at is that bearing is noisy
        # at close range. That is handled below by scaling the yaw command
        # down as distance shrinks, which bounds the bad signal's EFFECT
        # rather than freezing the signal itself.
        bearing_error = wrap_angle(math.atan2(dy, dx) - yaw)

        # --- Altitude -> thrust (PI, not P)
        altitude_error = clamp(TARGET_ALTITUDE - altitude, -1.0, 1.0)

        # Always integrate; bound the integral instead of gating where it may
        # act. An earlier version only accumulated inside a 0.4m band, which
        # deadlocked: the airframe's natural equilibrium sits ~0.6m BELOW
        # target, outside the band, so the integral could never accumulate the
        # authority to climb into the band, and the drone settled at exactly
        # the P-only equilibrium (0.90m measured, twice).
        #
        # Windup is prevented by the clamp plus back-off: when the output is
        # already saturated in the direction the error points, stop adding to
        # it. That bounds overshoot without ever locking the term out.
        integral_step = altitude_error * (self.timestep / 1000.0)
        saturated_high = self.altitude_integral >= MAX_VERTICAL_INTEGRAL
        saturated_low = self.altitude_integral <= -MAX_VERTICAL_INTEGRAL
        if not (saturated_high and integral_step > 0) and not (
            saturated_low and integral_step < 0
        ):
            self.altitude_integral = clamp(
                self.altitude_integral + integral_step,
                -MAX_VERTICAL_INTEGRAL,
                MAX_VERTICAL_INTEGRAL,
            )

        # NO DECAY TERM HERE — deliberately. An earlier version bled the
        # integral toward zero whenever the error exceeded INTEGRAL_BAND, and
        # that created a second deadlock one layer up from the band gate it
        # replaced: the drone pinned at exactly 1.00m, where the error (0.50m)
        # sits just outside the 0.4m band, so the integral was bled every step
        # and balanced the error term at a false equilibrium. Eleven
        # consecutive samples read 1.00m.
        #
        # The saturation back-off above is sufficient for windup on its own,
        # and unlike a decay it cannot manufacture an equilibrium of its own.
        vertical_input = (
            K_VERTICAL_P * (altitude_error ** 3.0)
            + K_VERTICAL_I * self.altitude_integral
        )

        # --- Bearing -> yaw rate
        # Scale authority down as distance shrinks. At close range the bearing
        # to target is dominated by noise — a 5cm position wobble 20cm out
        # swings it through a radian — so acting on it at full gain makes the
        # drone chase an angle that means nothing. Fading the gain bounds that
        # signal's effect without freezing the signal (see the note above).
        bearing_authority = clamp(distance / BEARING_FADE, 0.0, 1.0)
        yaw_input = (
            0.0 if self.arrived else K_YAW_P * bearing_error * bearing_authority
        )

        # --- Distance -> forward lean, minus a braking term
        # Closing speed along the bearing, from GPS deltas. Braking against it
        # is what stops the drone on the mark instead of orbiting the target.
        dt = self.timestep / 1000.0
        if self.prev_position is None:
            closing_speed = 0.0
        else:
            moved_x = x - self.prev_position[0]
            moved_y = y - self.prev_position[1]
            # Project movement onto the unit vector pointing at the target.
            if distance > 1e-6:
                closing_speed = (moved_x * dx + moved_y * dy) / (distance * dt)
            else:
                closing_speed = 0.0
        self.prev_position = (x, y)

        if self.arrived or abs(bearing_error) > FACING_TOLERANCE:
            pitch_disturbance = 0.0
        else:
            facing = math.cos(bearing_error)
            drive = K_FORWARD_P * distance * facing
            brake = clamp(K_BRAKE * closing_speed, 0.0, MAX_BRAKE)
            pitch_disturbance = clamp(
                drive - brake, 0.0, MAX_PITCH_DISTURBANCE
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
