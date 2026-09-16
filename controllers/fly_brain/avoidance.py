"""Avoidance reflex — the innate pathway.

The bee centring response, hardwired. No learning: this circuit is genetically
specified in the insect, and reproducing it means writing arithmetic.

    turn_rate = gain * (left_expansion - right_expansion)

More flow on the left means the obstacle is on the left, so turn right. Plus
a braking term when total expansion crosses a threshold — something is close
ahead and speed must come down.

The urgency output is a SIGMOID of total expansion, not a boolean. It is what
the arbitration layer blends on, and a boolean there produces a dodge-return
limit cycle: dodge, obstacle leaves view, turn back, obstacle returns.
"""

import numpy as np

from contracts import AvoidanceCommand, PooledChannels


def sigmoid(x: float) -> float:
    """Numerically stable logistic function.

    The naive form overflows for large negative x; branching on the sign
    keeps exp() arguments non-positive.
    """
    if x >= 0.0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


class AvoidanceReflex:
    """Steer away from high flow, brake when closing fast."""

    __slots__ = (
        "_turn_gain",
        "_brake_threshold",
        "_brake_gain",
        "_max_turn_rate",
        "_urgency_midpoint",
        "_urgency_steepness",
    )

    def __init__(
        self,
        turn_gain: float = 1.0,
        brake_threshold: float = 1.0,
        brake_gain: float = 0.8,
        max_turn_rate: float = 1.5,
        urgency_midpoint: float = 1.0,
        urgency_steepness: float = 4.0,
    ) -> None:
        self._turn_gain = turn_gain
        self._brake_threshold = brake_threshold
        self._brake_gain = brake_gain
        self._max_turn_rate = max_turn_rate
        self._urgency_midpoint = urgency_midpoint
        self._urgency_steepness = urgency_steepness

    def react(self, channels: PooledChannels) -> AvoidanceCommand:
        """Two lines of real logic, plus bounds.

        Args:
            channels: pooled wide-field channels, rotation already subtracted.

        Returns:
            AvoidanceCommand with turn rate, brake force and urgency.
        """
        # Positive difference = more flow left = obstacle left = turn right,
        # which is a negative turn rate under the right-handed convention.
        turn_rate = -self._turn_gain * channels.expansion_difference
        turn_rate = float(
            np.clip(turn_rate, -self._max_turn_rate, self._max_turn_rate)
        )

        total = channels.total_expansion

        excess = total - self._brake_threshold
        brake_force = float(max(0.0, self._brake_gain * excess))

        # Sigmoid, never a step. This is the value arbitration blends on.
        urgency = sigmoid((total - self._urgency_midpoint) * self._urgency_steepness)

        return AvoidanceCommand(
            turn_rate=turn_rate,
            brake_force=brake_force,
            urgency=urgency,
        )

    @classmethod
    def from_config(cls, cfg) -> "AvoidanceReflex":
        return cls(
            turn_gain=cfg.avoidance.turn_gain,
            brake_threshold=cfg.avoidance.brake_threshold,
            brake_gain=cfg.avoidance.brake_gain,
            max_turn_rate=cfg.avoidance.max_turn_rate,
            urgency_midpoint=cfg.avoidance.urgency_midpoint,
            urgency_steepness=cfg.avoidance.urgency_steepness,
        )
