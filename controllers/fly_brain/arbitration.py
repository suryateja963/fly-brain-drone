"""Arbitration.

Two pathways compete: avoidance and goal-seeking.

    urgency = sigmoid((total_expansion - threshold) * sharpness)
    turn    = urgency * avoidance_turn + (1 - urgency) * goal_turn

USE A SIGMOID, NOT AN IF-STATEMENT. A hard switch produces oscillation: the
drone dodges, the obstacle leaves view, it turns back, the obstacle returns.
Smooth blending prevents this limit cycle.

This mirrors the fly: the innate pathway dominates when threat is immediate,
the goal pathway when the air is clear.

Phase 6.
"""

THRESHOLD = 1.0
SHARPNESS = 4.0


def blend(avoidance_turn, goal_turn, total_expansion,
          threshold=THRESHOLD, sharpness=SHARPNESS):
    """Blend the two pathways by threat urgency.

    Args:
        avoidance_turn: turn command from the reflex pathway.
        goal_turn: turn command from the goal pathway.
        total_expansion: summed expansion, the threat signal.
        threshold: expansion at which urgency is 0.5.
        sharpness: steepness of the blend.

    Returns:
        (turn, urgency) — blended turn, and the urgency in [0, 1] for logging
        and visualisation.
    """
    raise NotImplementedError("Phase 6")
