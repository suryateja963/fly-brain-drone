"""Descending bottleneck.

Four values only: roll, pitch, yaw_rate, thrust.

This is an actual enforced interface. `descend` is the only function that may
produce motor commands, and nothing else in the codebase touches the motors.
The constraint is the point — it is the descending-neuron bottleneck, and
keeping it honest keeps the architecture honest.

Phase 7.
"""

from collections import namedtuple

Command = namedtuple("Command", ["roll", "pitch", "yaw_rate", "thrust"])


def descend(turn, speed_scale, altitude_error):
    """Collapse the whole architecture into exactly four floats.

    Args:
        turn: blended turn command from arbitration.
        speed_scale: forward-speed multiplier from the reflex.
        altitude_error: target altitude minus current, in metres.

    Returns:
        Command — exactly four floats, nothing else.
    """
    raise NotImplementedError("Phase 7")
