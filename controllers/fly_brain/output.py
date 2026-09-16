"""Descending bottleneck — the only path to the motors.

Four values reach the rotors: roll, pitch, yaw_rate, thrust. Nothing else in
the codebase touches them. The constraint is the point — it is the
descending-neuron bottleneck, and keeping it honest keeps the architecture
honest.

THE ALTITUDE CEILING IS ENFORCED HERE, DELIBERATELY.

Not in the planner, not in arbitration, not in the goal generator — here, at
the final stage, after every other layer has had its say. A safety limit
placed upstream can be bypassed by any downstream change, including one made
months later by someone who never read the planner. A limit at the
bottleneck cannot be bypassed without editing this file, which is exactly
the review surface a safety-critical constraint should have.
"""

import numpy as np

from contracts import MotorCommands

# Motor index convention, matching the Webots Mavic 2 Pro rotor order.
FRONT_LEFT = 0
FRONT_RIGHT = 1
REAR_LEFT = 2
REAR_RIGHT = 3


class OutputBottleneck:
    """Collapse the whole architecture into four motor values."""

    __slots__ = (
        "_base_thrust",
        "_ceiling_m",
        "_soft_limit_m",
        "_cap_enabled",
        "_min_motor",
        "_max_motor",
        "_k_roll",
        "_k_pitch",
    )

    def __init__(
        self,
        base_thrust: float = 68.5,
        ceiling_m: float = 10.0,
        soft_limit_m: float = 8.0,
        cap_enabled: bool = True,
        min_motor: float = 0.0,
        max_motor: float = 576.0,
        k_roll: float = 50.0,
        k_pitch: float = 30.0,
    ) -> None:
        if soft_limit_m > ceiling_m:
            raise ValueError(
                f"soft limit {soft_limit_m} must not exceed ceiling {ceiling_m}"
            )

        self._base_thrust = base_thrust
        self._ceiling_m = ceiling_m
        self._soft_limit_m = soft_limit_m
        self._cap_enabled = cap_enabled
        self._min_motor = min_motor
        self._max_motor = max_motor
        self._k_roll = k_roll
        self._k_pitch = k_pitch

    def descend(
        self,
        roll_input: float,
        pitch_input: float,
        yaw_input: float,
        vertical_input: float,
        altitude: float,
        vertical_rate: float = 0.0,
    ) -> MotorCommands:
        """Produce the four motor values, with the ceiling enforced.

        Args:
            roll_input: attitude stabilisation term.
            pitch_input: attitude term, including any forward lean.
            yaw_input: yaw rate command.
            vertical_input: altitude controller output.
            altitude: current altitude in metres — the cap is checked
                against this, not against whatever the planner intended.
            vertical_rate: current climb rate, m/s.

        Returns:
            MotorCommands, with flags recording whether the cap or the motor
            limits engaged.
        """
        capped = False

        if self._cap_enabled:
            vertical_input, capped = self._apply_ceiling(
                vertical_input, altitude, vertical_rate
            )

        base = self._base_thrust + vertical_input

        # Standard quadrotor mix. Diagonal pairs counter-rotate, so yaw
        # enters with alternating sign — which is why a large yaw command
        # unbalances the aircraft rather than simply turning it.
        motors = np.array(
            [
                base - roll_input + pitch_input - yaw_input,  # front left
                base + roll_input + pitch_input + yaw_input,  # front right
                base - roll_input - pitch_input + yaw_input,  # rear left
                base + roll_input - pitch_input - yaw_input,  # rear right
            ],
            dtype=np.float64,
        )

        clamped = bool(
            np.any(motors < self._min_motor) or np.any(motors > self._max_motor)
        )
        np.clip(motors, self._min_motor, self._max_motor, out=motors)

        return MotorCommands(
            motors=motors,
            clamped=clamped,
            altitude_capped=capped,
        )

    def _apply_ceiling(
        self, vertical_input: float, altitude: float, vertical_rate: float
    ):
        """Enforce the altitude limit. Returns (vertical_input, capped).

        Two stages, because a hard stop at the ceiling alone would let the
        drone arrive there at speed and overshoot:

          - above the soft limit, climb authority is progressively reduced
          - at the ceiling, climb is refused outright and any upward rate is
            actively opposed
        """
        if altitude >= self._ceiling_m:
            # At or above the hard cap: no climb, and brake any ascent.
            limited = min(vertical_input, 0.0)
            if vertical_rate > 0.0:
                limited = min(limited, -vertical_rate)
            return limited, True

        if altitude > self._soft_limit_m and vertical_input > 0.0:
            # Between soft and hard: scale climb authority linearly to zero.
            span = self._ceiling_m - self._soft_limit_m
            remaining = (self._ceiling_m - altitude) / span
            return vertical_input * remaining, True

        return vertical_input, False

    def would_exceed_ceiling(self, target_altitude: float) -> bool:
        """Whether a requested altitude is above the cap.

        Lets the planner avoid commanding something the bottleneck will
        refuse — an optimisation, never the enforcement itself.
        """
        return self._cap_enabled and target_altitude > self._ceiling_m

    def clamp_target_altitude(self, target_altitude: float) -> float:
        """Reduce a target to the legal maximum."""
        if not self._cap_enabled:
            return target_altitude
        return min(target_altitude, self._ceiling_m)

    @property
    def ceiling(self) -> float:
        return self._ceiling_m

    @classmethod
    def from_config(cls, cfg) -> "OutputBottleneck":
        return cls(
            base_thrust=cfg.flight.base_thrust,
            ceiling_m=cfg.altitude_cap.ceiling_m,
            soft_limit_m=cfg.altitude_cap.soft_limit_m,
            cap_enabled=cfg.altitude_cap.enabled,
            k_roll=cfg.flight.k_roll_p,
            k_pitch=cfg.flight.k_pitch_p,
        )
