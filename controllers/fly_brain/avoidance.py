"""Avoidance reflex — innate pathway.

The bee centring response. Two lines of real logic, hardwired, no learning.

Phase 4.
"""

DEFAULT_GAIN = 1.0
BRAKE_THRESHOLD = 1.0


def reflex(wide_field, gain=DEFAULT_GAIN, brake_threshold=BRAKE_THRESHOLD):
    """Steer away from high flow, and brake when closing fast.

        turn = gain * (left_expansion - right_expansion)

    A positive difference means more flow on the left, so the obstacle is on
    the left, so turn right. If total expansion exceeds the threshold, reduce
    forward speed.

    Args:
        wide_field: lobula.WideField.
        gain: turn gain.
        brake_threshold: total expansion above which forward speed is cut.

    Returns:
        (turn, speed_scale) — turn command, and a forward-speed multiplier
        in [0, 1].
    """
    raise NotImplementedError("Phase 4")
