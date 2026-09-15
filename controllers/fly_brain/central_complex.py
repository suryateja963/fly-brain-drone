"""Central complex — ellipsoid body / protocerebral bridge / fan-shaped body.

A ring attractor holding heading, plus a goal vector.

Structure:
  - N units in a ring (16 or 32 is plenty)
  - Local excitation: each unit excites its neighbours
  - Global inhibition: all units inhibit all others
  - Result: a single stable bump of activity

The bump will drift. That is biologically accurate and acceptable over short
flights.

This module is also the seam for a future LLM layer (spec §14): the goal
heading is the single clean input it would ever touch. Keep it that way.

Phase 5.
"""

import math

N_UNITS = 32


class RingAttractor:
    """A ring of units supporting one stable bump of activity."""

    def __init__(self, n_units=N_UNITS, excitation=1.0, inhibition=0.5):
        """Args:
        n_units: units around the ring.
        excitation: local neighbour excitation strength.
        inhibition: global inhibition strength.

        If the bump splits or fades, this ratio is wrong.
        """
        raise NotImplementedError("Phase 5")

    def step(self, yaw_rate, dt):
        """Shift the bump by the IMU yaw rate, then re-sharpen it.

        One step of the excitation/inhibition update after the shift.

        Returns:
            Estimated heading in radians, the bump position.
        """
        raise NotImplementedError("Phase 5")

    @property
    def heading(self):
        """Current bump position, in radians."""
        raise NotImplementedError("Phase 5")


def goal_error(heading, goal_heading):
    """Circular difference between current and goal heading.

    Returns:
        Signed error in radians, wrapped to [-pi, pi].
    """
    return math.atan2(
        math.sin(goal_heading - heading), math.cos(goal_heading - heading)
    )
