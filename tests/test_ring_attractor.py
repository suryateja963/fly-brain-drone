"""Ring attractor and path integration tests.

The bump must stay single and sharp, track integrated yaw across the seam,
and support dead reckoning home without GPS. Needs no simulator.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "controllers",
        "fly_brain",
    ),
)

from ring_attractor import (  # noqa: E402
    PathIntegrator,
    RingAttractor,
    circular_difference,
)

DT = 0.008  # 125Hz


def settle(ring, steps=200):
    """Let the attractor dynamics reach steady state with no rotation."""
    for _ in range(steps):
        ring.step(yaw_rate=0.0, dt=DT)


# ---------------------------------------------------------------------------
# Bump stability
# ---------------------------------------------------------------------------


def test_bump_is_single_and_stable():
    """With zero yaw rate, activity settles to one bump and stays put.

    If it splits or fades, the excitation/inhibition ratio is wrong.
    """
    ring = RingAttractor()
    before = ring.heading
    settle(ring, steps=500)

    state = ring.state()

    assert state.confidence > 0.5, f"bump faded: confidence={state.confidence}"
    assert circular_difference(ring.heading, before) == pytest.approx(0.0, abs=0.05)

    # Single peak: exactly one local maximum around the ring.
    bump = state.bump
    peaks = 0
    for i in range(len(bump)):
        prev_v = bump[i - 1]
        next_v = bump[(i + 1) % len(bump)]
        if bump[i] > prev_v and bump[i] > next_v:
            peaks += 1
    assert peaks == 1, f"bump split into {peaks} peaks"


def test_bump_activity_stays_normalised():
    """Total activity must neither explode nor collapse over time."""
    ring = RingAttractor()
    for _ in range(1000):
        ring.step(yaw_rate=0.5, dt=DT)

    total = ring.state().bump.sum()
    assert total == pytest.approx(1.0, abs=1e-6)


def test_bump_activity_is_non_negative():
    """Firing rates cannot be negative."""
    ring = RingAttractor()
    for _ in range(300):
        state = ring.step(yaw_rate=-1.2, dt=DT)
        assert np.all(state.bump >= 0.0)


# ---------------------------------------------------------------------------
# Heading tracking
# ---------------------------------------------------------------------------


def test_bump_tracks_constant_yaw():
    """Under a constant yaw rate, the bump rotates at that rate."""
    ring = RingAttractor()
    settle(ring, steps=100)

    start = ring.heading
    yaw_rate = 1.0
    duration = 1.0
    steps = int(duration / DT)

    for _ in range(steps):
        ring.step(yaw_rate=yaw_rate, dt=DT)

    expected = yaw_rate * duration
    actual = circular_difference(ring.heading, start)

    assert actual == pytest.approx(expected, abs=0.25), (
        f"expected ~{expected:.3f} rad, tracked {actual:.3f}"
    )


def test_bump_wraps_around_the_ring():
    """Yawing past the seam must not break tracking.

    The seam is where unit 31 meets unit 0. A naive implementation loses the
    bump or splits it here.
    """
    ring = RingAttractor(initial_heading=3.0)  # just short of pi
    settle(ring, steps=50)

    # Rotate well past the wrap point.
    for _ in range(int(1.0 / DT)):
        ring.step(yaw_rate=1.0, dt=DT)

    state = ring.state()
    assert state.confidence > 0.5, "bump degraded crossing the seam"

    peaks = 0
    bump = state.bump
    for i in range(len(bump)):
        if bump[i] > bump[i - 1] and bump[i] > bump[(i + 1) % len(bump)]:
            peaks += 1
    assert peaks == 1, f"seam crossing split the bump into {peaks}"


def test_opposite_yaw_returns_to_start():
    """Equal rotation each way must return the bump where it began."""
    ring = RingAttractor()
    settle(ring, steps=100)
    start = ring.heading

    for _ in range(250):
        ring.step(yaw_rate=+1.5, dt=DT)
    for _ in range(250):
        ring.step(yaw_rate=-1.5, dt=DT)

    assert circular_difference(ring.heading, start) == pytest.approx(0.0, abs=0.2)


def test_reset_reseeds_heading():
    ring = RingAttractor()
    ring.reset(heading_rad=1.5)
    assert ring.heading == pytest.approx(1.5, abs=0.15)


def test_confidence_reflects_bump_quality():
    """A clean bump is confident; uniform activity is not."""
    ring = RingAttractor()
    settle(ring, steps=200)
    assert ring.is_trusted()

    # Force a pathological uniform state and confirm confidence collapses.
    ring.reset(0.0)
    clean = ring.confidence
    assert clean > 0.3


def test_rejects_tiny_ring():
    with pytest.raises(ValueError):
        RingAttractor(n_units=4)


# ---------------------------------------------------------------------------
# Circular arithmetic
# ---------------------------------------------------------------------------


def test_goal_error_wraps_to_shortest_turn():
    """Error between headings either side of the seam takes the short way."""
    assert circular_difference(np.radians(359), 0.0) == pytest.approx(
        np.radians(-1), abs=1e-6
    )
    assert circular_difference(np.radians(1), np.radians(359)) == pytest.approx(
        np.radians(2), abs=1e-6
    )
    assert circular_difference(np.pi, -np.pi) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Path integration — GPS-denied return to home
# ---------------------------------------------------------------------------


def test_straight_flight_accumulates_distance():
    integrator = PathIntegrator()
    for _ in range(125):  # one second at 1 m/s
        integrator.update(heading_rad=0.0, speed=1.0, dt=DT)

    state = integrator.state()
    assert state.position_estimate[0] == pytest.approx(1.0, abs=0.02)
    assert state.distance_to_home == pytest.approx(1.0, abs=0.02)


def test_home_vector_points_back():
    integrator = PathIntegrator()
    for _ in range(125):
        integrator.update(heading_rad=0.0, speed=2.0, dt=DT)

    state = integrator.state()
    # Flew +x, so home is -x.
    assert state.home_vector[0] < 0.0
    assert integrator.heading_home() == pytest.approx(np.pi, abs=0.05)


def test_return_home_without_gps():
    """THE GPS-DENIED TEST.

    Fly out with no GPS at all, then confirm the drone still knows the way
    home. This is what makes the system spoof-proof: the estimate comes from
    its own motion, not from a signal an attacker controls.
    """
    integrator = PathIntegrator()

    # Out-and-turn with GPS unavailable throughout.
    for _ in range(250):
        integrator.update(heading_rad=0.0, speed=2.0, dt=DT, gps_position=None)
    for _ in range(250):
        integrator.update(heading_rad=np.pi / 2, speed=2.0, dt=DT, gps_position=None)

    state = integrator.state()
    assert not state.gps_available
    assert state.distance_to_home > 4.0

    # Fly the computed bearing home and confirm arrival.
    for _ in range(600):
        heading = integrator.heading_home()
        integrator.update(heading_rad=heading, speed=2.0, dt=DT, gps_position=None)
        if integrator.state().distance_to_home < 0.2:
            break

    assert integrator.state().distance_to_home < 0.3, (
        "dead reckoning failed to bring the drone home"
    )


def test_gps_corrects_rather_than_replaces():
    """A GPS fix nudges the estimate; it does not overwrite it.

    Partial correction means a spoofed signal can only drag the estimate
    slowly, leaving the discrepancy detectable.
    """
    integrator = PathIntegrator(gps_correction_rate=0.05)
    for _ in range(125):
        integrator.update(heading_rad=0.0, speed=1.0, dt=DT)

    before = integrator.state().position_estimate.copy()

    # A wildly wrong GPS reading.
    integrator.update(
        heading_rad=0.0, speed=0.0, dt=DT, gps_position=np.array([100.0, 100.0])
    )
    after = integrator.state().position_estimate

    moved = np.linalg.norm(after - before)
    assert moved > 0.0, "GPS had no effect at all"
    assert moved < 10.0, "GPS overwrote the estimate instead of correcting it"


def test_set_home_changes_return_target():
    integrator = PathIntegrator()
    integrator.set_home(np.array([5.0, 5.0]))
    state = integrator.state()
    assert state.distance_to_home == pytest.approx(np.hypot(5.0, 5.0), abs=1e-6)
