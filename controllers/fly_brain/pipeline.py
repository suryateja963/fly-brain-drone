"""The full pipeline — every layer wired into one call.

    camera frame
        -> retina (coarse, blurred)
        -> optic flow (Lucas-Kanade, or EMD)
        -> wide-field pooling (rotation subtracted FIRST)
        -> avoidance reflex          \
                                       -> arbitration (sigmoid blend)
        -> ring attractor + goal     /
        -> compliance (geofence, height zones, wind)
        -> failsafes (battery, motors, vibration, payload)
        -> output bottleneck (altitude cap enforced HERE)
        -> four motor values

Kept separate from fly_brain.py deliberately. This module is pure
computation: it takes sensor values and returns motor commands, touching no
Webots API. That means the whole architecture is testable end to end without
a simulator, which is what test_integration.py does.

PRIORITY ORDER, and why it is this way. Several subsystems can override the
goal, and when two want different things the more urgent must win:

    1. motor failure      — the aircraft is damaged; nothing else matters
    2. geofence violation — legal hard constraint, overrides mission
    3. battery emergency  — cannot reach home; ping and land
    4. vibration anomaly  — degrading; return to base at reduced speed
    5. battery return     — can still reach home, so go now
    6. avoidance          — blended, not overriding (see arbitration.py)
    7. goal               — the mission

Anything above avoidance REPLACES the goal. Avoidance itself blends, because
a hard switch there produces the dodge-return limit cycle.
"""

from typing import List, Optional

import numpy as np

from arbitration import Arbitrator
from avoidance import AvoidanceReflex
from battery import BatteryModel
from contracts import (
    ArbitrationOutput,
    GoalCommand,
    MotorCommands,
)
from geofence import Geofence
from height_zones import HeightZoneManager
from motor_failure import MotorFailureHandler
from optical_flow import make_flow_frontend
from output import OutputBottleneck
from payload import PayloadManager
from retina import sample as retina_sample
from ring_attractor import PathIntegrator, RingAttractor, circular_difference
from telemetry import TelemetryRecorder
from vibration import VibrationMonitor
from wide_field import WideFieldPooling
from wind import WindEstimator


class PipelineState:
    """Everything the pipeline computed this step, for logging and the demo.

    A plain attribute holder rather than a contract dataclass: this is
    diagnostic output, not an inter-module interface, and the demo panels
    want whatever happens to be interesting rather than a fixed schema.
    """

    __slots__ = (
        "flow_field",
        "channels",
        "avoidance",
        "heading_state",
        "path_state",
        "goal",
        "arbitration",
        "motors",
        "geofence_status",
        "battery_state",
        "motor_health",
        "vibration_state",
        "wind_estimate",
        "payload_state",
        "override_reason",
        "target_altitude",
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            setattr(self, slot, None)


class FlyBrainPipeline:
    """The whole architecture, as one object.

    Constructed from config; stepped with sensor readings; returns motor
    commands. No Webots imports, so it runs anywhere.
    """

    __slots__ = (
        "_cfg",
        "_flow",
        "_pooling",
        "_avoidance",
        "_ring",
        "_path",
        "_arbitrator",
        "_geofence",
        "_heights",
        "_wind",
        "_battery",
        "_motors",
        "_vibration",
        "_payload",
        "_bottleneck",
        "_telemetry",
        "_target_altitude",
        "_goal_position",
        "_elapsed",
        "_returning_home",
        "_altitude_integral",
        "_previous_altitude",
        "_turning",
    )

    def __init__(self, cfg, goal_position: np.ndarray = None) -> None:
        self._cfg = cfg

        self._flow = make_flow_frontend(cfg)
        self._pooling = WideFieldPooling.from_config(cfg)
        self._avoidance = AvoidanceReflex.from_config(cfg)
        self._ring = RingAttractor.from_config(cfg)
        self._path = PathIntegrator.from_config(cfg)
        self._arbitrator = Arbitrator.from_config(cfg)

        self._geofence = Geofence.from_config(cfg)
        self._heights = HeightZoneManager.from_config(cfg)
        self._wind = WindEstimator.from_config(cfg)

        self._battery = BatteryModel.from_config(cfg)
        self._motors = MotorFailureHandler.from_config(cfg)
        self._vibration = VibrationMonitor.from_config(cfg)
        self._payload = PayloadManager.from_config(cfg)

        self._bottleneck = OutputBottleneck.from_config(cfg)
        self._telemetry = TelemetryRecorder.from_config(cfg)

        self._target_altitude = cfg.flight.target_altitude
        self._goal_position = (
            np.zeros(2, dtype=np.float64)
            if goal_position is None
            else np.asarray(goal_position, dtype=np.float64)[:2]
        )
        self._elapsed = 0.0
        self._returning_home = False

        # Flight control state, carried between steps. Starts turning so
        # the drone squares up to the goal before translating.
        self._altitude_integral = 0.0
        self._previous_altitude = None
        self._turning = True

    # -----------------------------------------------------------------
    # The step
    # -----------------------------------------------------------------

    def step(
        self,
        frame: np.ndarray,
        position: np.ndarray,
        altitude: float,
        yaw: float,
        yaw_rate: float,
        speed: float,
        acceleration: np.ndarray,
        commanded_velocity: np.ndarray,
        actual_velocity: np.ndarray,
        motor_feedback: np.ndarray,
        last_motors: np.ndarray,
        dt: float,
        gps_position: np.ndarray = None,
        obstacles: List[np.ndarray] = None,
        roll: float = 0.0,
        pitch: float = 0.0,
        roll_rate: float = 0.0,
        pitch_rate: float = 0.0,
    ) -> PipelineState:
        """One control cycle, sensors to motor commands.

        Args:
            frame: raw camera image, HxWx3 or HxWx4 uint8.
            position: (2,) world position.
            altitude: metres.
            yaw, yaw_rate: radians, radians/second.
            speed: forward speed, m/s.
            acceleration: (3,) IMU accelerometer.
            commanded_velocity, actual_velocity: (2,) for wind estimation.
            motor_feedback: (4,) achieved rotor speeds.
            last_motors: (4,) previously commanded rotor speeds.
            dt: timestep in seconds.
            gps_position: (2,) or None when GPS is unavailable or distrusted.
            obstacles: detected positions near the drop zone.

        Returns:
            PipelineState, whose `.motors` carries the commands to apply.
        """
        state = PipelineState()
        self._elapsed += dt
        p = np.asarray(position, dtype=np.float64)[:2]

        # --- Vision ----------------------------------------------------
        retina = retina_sample(
            frame,
            shape=(self._cfg.retina.rows, self._cfg.retina.cols),
            blur_sigma=self._cfg.retina.blur_sigma,
        )
        state.flow_field = self._flow.compute(retina, timestamp=self._elapsed)

        # Rotation subtracted BEFORE expansion — see wide_field.py.
        state.channels = self._pooling.pool(state.flow_field, yaw_rate)
        state.avoidance = self._avoidance.react(state.channels)

        # --- Navigation ------------------------------------------------
        state.heading_state = self._ring.step(yaw_rate, dt)
        state.path_state = self._path.update(
            heading_rad=state.heading_state.heading_rad,
            speed=speed,
            dt=dt,
            gps_position=gps_position,
        )

        # --- Environment and health ------------------------------------
        state.geofence_status = self._geofence.check(p)
        state.wind_estimate = self._wind.update(
            commanded_velocity, actual_velocity, dt
        )
        state.motor_health = self._motors.detect(last_motors, motor_feedback)
        state.vibration_state = self._vibration.update(acceleration)
        state.battery_state = self._battery.update(
            total_thrust=float(np.sum(last_motors)),
            distance_to_home=state.path_state.distance_to_home,
            dt=dt,
        )

        if obstacles is not None:
            state.payload_state = self._payload.assess(
                drone_position=p,
                drop_target=self._goal_position,
                altitude=altitude,
                obstacles=obstacles,
            )

        # --- Goal, and the overrides that can replace it ---------------
        state.target_altitude = self._resolve_altitude(p)
        goal, override = self._resolve_goal(p, state, altitude)
        state.goal = goal
        state.override_reason = override

        # --- Arbitration: blend, never switch --------------------------
        state.arbitration = self._arbitrator.arbitrate(
            avoidance=state.avoidance,
            goal=goal,
            current_heading=yaw,
            total_expansion=state.channels.total_expansion,
        )

        # --- Motors ----------------------------------------------------
        state.motors = self._to_motors(
            state, yaw, altitude, dt,
            roll=roll, pitch=pitch,
            roll_rate=roll_rate, pitch_rate=pitch_rate,
        )

        self._record_incidents(state)
        self._telemetry.update(
            position=p,
            altitude=altitude,
            charge_pct=state.battery_state.charge_pct,
            dt=dt,
        )

        return state

    # -----------------------------------------------------------------
    # Goal resolution and the priority ladder
    # -----------------------------------------------------------------

    def _resolve_altitude(self, position: np.ndarray) -> float:
        """Target altitude here, respecting height zones and the cap."""
        zone_target = self._heights.target_for(position)
        zone_ceiling = self._heights.ceiling_for(position)
        target = min(self._target_altitude, zone_target, zone_ceiling)
        return self._bottleneck.clamp_target_altitude(target)

    def _resolve_goal(self, position, state, altitude):
        """Apply the priority ladder. Returns (GoalCommand, reason or None).

        Each branch REPLACES the goal rather than blending, because these
        are conditions where continuing the mission is wrong, not merely
        inconvenient. Avoidance is deliberately absent: it blends downstream.
        """
        cruise = self._cfg.battery.cruise_speed_ms
        home_heading = self._path.heading_home()

        # 1. Motor failure — the aircraft is damaged.
        if state.motor_health.degraded_mode:
            if not self._motors.can_hold_position(state.motor_health):
                return (
                    GoalCommand(
                        desired_heading=home_heading,
                        desired_speed=0.0,
                        desired_altitude=self._motors.descent_target(
                            altitude, self._cfg.simulation.timestep_ms / 1000.0
                        ),
                    ),
                    "motor_failure_descent",
                )
            return (
                GoalCommand(
                    desired_heading=home_heading,
                    desired_speed=cruise * 0.5,
                    desired_altitude=state.target_altitude,
                ),
                "motor_failure_return",
            )

        # 2. Geofence — a legal hard constraint.
        if not state.geofence_status.inside:
            direction = state.geofence_status.violation_direction
            escape_heading = (
                float(np.arctan2(direction[1], direction[0]))
                if direction is not None
                else home_heading
            )
            return (
                GoalCommand(
                    desired_heading=escape_heading,
                    desired_speed=cruise,
                    desired_altitude=state.target_altitude,
                ),
                "geofence_escape",
            )

        # 3. Battery emergency — cannot reach home at all.
        if state.battery_state.emergency:
            return (
                GoalCommand(
                    desired_heading=home_heading,
                    desired_speed=0.0,
                    desired_altitude=max(0.0, altitude - 0.3),
                ),
                "battery_emergency_land",
            )

        # 4. Vibration — degrading, return slowly.
        if state.vibration_state.anomaly_detected:
            scale = self._vibration.speed_scale(state.vibration_state)
            return (
                GoalCommand(
                    desired_heading=home_heading,
                    desired_speed=cruise * scale,
                    desired_altitude=state.target_altitude,
                ),
                "vibration_return",
            )

        # 5. Battery reserve — can still reach home, so go now.
        if not state.battery_state.can_return_home or self._returning_home:
            self._returning_home = True
            return (
                GoalCommand(
                    desired_heading=home_heading,
                    desired_speed=cruise,
                    desired_altitude=state.target_altitude,
                ),
                "battery_return_home",
            )

        # 7. The mission.
        to_goal = self._goal_position - position
        return (
            GoalCommand(
                desired_heading=float(np.arctan2(to_goal[1], to_goal[0])),
                desired_speed=cruise,
                desired_altitude=state.target_altitude,
            ),
            None,
        )

    # -----------------------------------------------------------------
    # Motors
    # -----------------------------------------------------------------

    def _to_motors(
        self,
        state: PipelineState,
        yaw: float,
        altitude: float,
        dt: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        roll_rate: float = 0.0,
        pitch_rate: float = 0.0,
    ) -> MotorCommands:
        """Convert the arbitrated command into four rotor values.

        This reuses the flight control that Phase 1 arrived at over twelve
        measured debugging runs. An earlier version of this method was a
        naive reimplementation and it CRASHED ON TAKEOFF in the first flight
        test — altitude 0.02m, yaw spun to -2.78 rad. It repeated three
        mistakes Phase 1 had already paid for:

          - a proportional-only altitude term, ignoring the tuned I and D
            gains sitting in config. A P-only loop on a cubic error cannot
            close a steady-state offset (measured: settled 0.90m against a
            1.5m command) and a PI loop without damping oscillates.
          - full yaw gain applied while simultaneously leaning forward.
            yaw_input enters all four rotors with alternating sign, so it
            fights the attitude stabilisation directly; Phase 1 solved this
            by turning BEFORE translating, not by turning harder.
          - roll_input hardcoded to zero, so nothing held the aircraft level.

        The lesson is in the architecture now: the flight control layer is
        not something to rewrite from first principles per pipeline.
        """
        arbitration: ArbitrationOutput = state.arbitration
        flight = self._cfg.flight

        heading_error = circular_difference(arbitration.final_heading, yaw)

        # --- Altitude: full PID, using the gains that were actually tuned.
        altitude_error = float(np.clip(arbitration.final_altitude - altitude, -1.0, 1.0))

        integral_step = altitude_error * dt
        at_high = self._altitude_integral >= flight.max_vertical_integral
        at_low = self._altitude_integral <= -flight.max_vertical_integral
        if not (at_high and integral_step > 0) and not (at_low and integral_step < 0):
            self._altitude_integral = float(
                np.clip(
                    self._altitude_integral + integral_step,
                    -flight.max_vertical_integral,
                    flight.max_vertical_integral,
                )
            )

        if self._previous_altitude is None:
            vertical_rate = 0.0
        else:
            vertical_rate = (altitude - self._previous_altitude) / dt
        self._previous_altitude = altitude

        vertical_input = (
            flight.k_vertical_p * (altitude_error ** 3.0)
            + flight.k_vertical_i * self._altitude_integral
            - flight.k_vertical_d * vertical_rate
        )

        # --- TURN before CRUISE, with hysteresis between the states.
        if self._turning:
            if abs(heading_error) < flight.turn_exit:
                self._turning = False
        elif abs(heading_error) > flight.turn_enter:
            self._turning = True

        # --- Yaw authority fades close to the goal, where bearing is noise.
        to_goal = float(
            np.linalg.norm(self._goal_position - state.path_state.position_estimate)
        )
        yaw_authority = float(np.clip(to_goal / flight.yaw_fade_start, 0.0, 1.0))
        yaw_input = flight.k_yaw_p * heading_error * yaw_authority

        # --- Forward lean: suppressed entirely while turning, so the
        # rotation finishes before speed builds.
        if self._turning:
            pitch_disturbance = 0.0
        else:
            facing = max(0.0, float(np.cos(heading_error)))
            pitch_disturbance = float(
                np.clip(
                    flight.k_forward_p * arbitration.final_speed * facing,
                    0.0,
                    flight.max_pitch_disturbance,
                )
            )

        # --- Attitude stabilisation. Without this nothing holds the
        # aircraft level and it tips at the first lateral command.
        roll_input = flight.k_roll_p * float(np.clip(roll, -1.0, 1.0)) + roll_rate
        pitch_input = (
            flight.k_pitch_p * float(np.clip(pitch, -1.0, 1.0))
            + pitch_rate
            - pitch_disturbance
        )

        commands = self._bottleneck.descend(
            roll_input=roll_input,
            pitch_input=pitch_input,
            yaw_input=yaw_input,
            vertical_input=vertical_input,
            altitude=altitude,
            vertical_rate=vertical_rate,
        )

        if state.motor_health.degraded_mode:
            rebalanced = self._motors.rebalance(
                commands.motors, state.motor_health
            )
            commands = MotorCommands(
                motors=rebalanced,
                clamped=commands.clamped,
                altitude_capped=commands.altitude_capped,
            )

        return commands

    # -----------------------------------------------------------------
    # Incidents
    # -----------------------------------------------------------------

    def _record_incidents(self, state: PipelineState) -> None:
        """Log anything noteworthy. Deduplicated by the incident log."""
        if not state.geofence_status.inside:
            self._telemetry.record_incident(
                "geofence_violation",
                f"within {state.geofence_status.nearest_boundary_dist:.1f}m "
                "of a no-fly zone",
            )

        if state.battery_state.emergency:
            self._telemetry.record_incident(
                "battery_emergency",
                f"charge {state.battery_state.charge_pct:.1f}% — "
                "cannot reach home",
            )
        elif not state.battery_state.can_return_home:
            self._telemetry.record_incident(
                "battery_return",
                f"return-to-home triggered at "
                f"{state.battery_state.charge_pct:.1f}%",
            )

        if state.motor_health.degraded_mode:
            dead = np.flatnonzero(~state.motor_health.rotor_status)
            self._telemetry.record_incident(
                "motor_failure",
                f"rotor(s) {dead.tolist()} failed; rebalanced",
            )

        if state.vibration_state.anomaly_detected:
            self._telemetry.record_incident(
                "vibration_anomaly",
                f"{state.vibration_state.dominant_freq:.0f}Hz at "
                f"amplitude {state.vibration_state.amplitude:.2f}",
            )

        if state.motors.altitude_capped:
            self._telemetry.record_incident(
                "altitude_cap", f"ceiling {self._bottleneck.ceiling:.1f}m enforced"
            )

        if state.wind_estimate.gust_detected:
            self._telemetry.record_incident(
                "wind_gust", f"{state.wind_estimate.speed:.1f} m/s"
            )

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    @property
    def telemetry(self) -> TelemetryRecorder:
        return self._telemetry

    @property
    def battery(self) -> BatteryModel:
        return self._battery

    @property
    def motor_handler(self) -> MotorFailureHandler:
        return self._motors

    @property
    def vibration_monitor(self) -> VibrationMonitor:
        return self._vibration

    @property
    def payload(self) -> PayloadManager:
        return self._payload

    @property
    def path(self) -> PathIntegrator:
        return self._path

    @property
    def ring(self) -> RingAttractor:
        return self._ring

    def set_goal(self, position: np.ndarray) -> None:
        self._goal_position = np.asarray(position, dtype=np.float64)[:2]

    def set_home(self, position: np.ndarray) -> None:
        self._path.set_home(position)

    def report(self):
        return self._telemetry.report()
