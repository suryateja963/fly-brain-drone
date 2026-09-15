"""Ring attractor stability — Phase 5.

The bump must stay single and sharp, and track integrated yaw. Needs no
simulator; run with `pytest`.
"""

import pytest


@pytest.mark.skip(reason="Phase 5 not implemented")
def test_bump_is_single_and_stable():
    """With zero yaw rate, activity settles to one bump and stays put.

    If it splits or fades, the excitation/inhibition ratio is wrong.
    """


@pytest.mark.skip(reason="Phase 5 not implemented")
def test_bump_tracks_constant_yaw():
    """Under a constant yaw rate, the bump rotates at that rate."""


@pytest.mark.skip(reason="Phase 5 not implemented")
def test_bump_wraps_around_the_ring():
    """Yawing past the ring's seam must not break tracking."""


@pytest.mark.skip(reason="Phase 5 not implemented")
def test_goal_error_wraps_to_shortest_turn():
    """Error between headings either side of the seam takes the short way."""
