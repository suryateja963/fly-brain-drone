"""Arbitration — two pathways compete.

    urgency = sigmoid((total_expansion - threshold) * steepness)
    output  = urgency * avoidance + (1 - urgency) * goal

USE A SIGMOID, NOT AN IF-STATEMENT.

This is the single most important rule in the file, and it is not stylistic.
A hard switch produces a limit cycle: the drone dodges, the obstacle leaves
the field of view, urgency drops to zero, it turns back toward the goal, the
obstacle re-enters view, urgency jumps to one, it dodges again — forever,
without ever passing the obstacle.

Smooth blending means the drone commits partially, the obstacle stays
partially in view, and the two pathways find an equilibrium that carries it
around. This mirrors the fly: the innate pathway dominates when threat is
immediate, the goal pathway when the air is clear, and neither is ever fully
switched off.
"""

import numpy as np

from contracts import ArbitrationOutput, AvoidanceCommand, GoalCommand
from ring_attractor import circular_difference


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


def blend_headings(a: float, b: float, weight_b: float) -> float:
    """Circular interpolation between two headings.

    Linear interpolation is wrong on a circle: blending 179 degrees and -179
    degrees linearly gives 0, pointing the drone in exactly the opposite
    direction from either input. Interpolating on the unit circle gives 180,
    which is correct.
    """
    x = (1.0 - weight_b) * np.cos(a) + weight_b * np.cos(b)
    y = (1.0 - weight_b) * np.sin(a) + weight_b * np.sin(b)
    return float(np.arctan2(y, x))


class Arbitrator:
    """Blend the avoidance reflex with goal-seeking."""

    __slots__ = ("_midpoint", "_steepness", "_min_goal_weight")

    def __init__(
        self,
        sigmoid_midpoint: float = 1.0,
        sigmoid_steepness: float = 4.0,
        min_goal_weight: float = 0.05,
    ) -> None:
        self._midpoint = sigmoid_midpoint
        self._steepness = sigmoid_steepness
        self._min_goal_weight = min_goal_weight

    def arbitrate(
        self,
        avoidance: AvoidanceCommand,
        goal: GoalCommand,
        current_heading: float,
        total_expansion: float,
    ) -> ArbitrationOutput:
        """Produce a single command from two competing pathways.

        Args:
            avoidance: reflex output, carrying its own urgency.
            goal: where the goal pathway wants to go.
            current_heading: present heading, the base for the avoidance turn.
            total_expansion: threat signal driving the blend.

        Returns:
            ArbitrationOutput with both weights retained for display.
        """
        # The reflex computes its own urgency from pooled channels; the
        # arbitrator recomputes from total expansion so the blend can be
        # tuned independently of the reflex's braking behaviour.
        threat = sigmoid((total_expansion - self._midpoint) * self._steepness)

        # The goal never fully vanishes. At urgency 1.0 a drone with no goal
        # component at all will happily dodge sideways forever; retaining a
        # floor means it keeps making progress even while avoiding.
        avoidance_weight = float(min(threat, 1.0 - self._min_goal_weight))
        goal_weight = 1.0 - avoidance_weight

        # The avoidance pathway expresses itself as a turn away from current
        # heading, not as an absolute bearing: the reflex knows which way to
        # turn, not where it wants to end up.
        avoidance_heading = current_heading + avoidance.turn_rate

        final_heading = blend_headings(
            goal.desired_heading, avoidance_heading, avoidance_weight
        )

        # Braking scales speed down; the goal's desired speed is the ceiling.
        speed_scale = max(0.0, 1.0 - avoidance.brake_force)
        final_speed = float(goal.desired_speed * speed_scale)

        # Altitude is not contested by the reflex — the fly's avoidance is
        # lateral — so the goal's altitude passes through.
        final_altitude = goal.desired_altitude

        return ArbitrationOutput(
            final_heading=final_heading,
            final_speed=final_speed,
            final_altitude=final_altitude,
            avoidance_weight=avoidance_weight,
            goal_weight=goal_weight,
        )

    def heading_error(self, output: ArbitrationOutput, current: float) -> float:
        """Shortest turn from current heading to the arbitrated one."""
        return circular_difference(output.final_heading, current)

    @classmethod
    def from_config(cls, cfg) -> "Arbitrator":
        return cls(
            sigmoid_midpoint=cfg.arbitration.sigmoid_midpoint,
            sigmoid_steepness=cfg.arbitration.sigmoid_steepness,
            min_goal_weight=cfg.arbitration.min_goal_weight,
        )
