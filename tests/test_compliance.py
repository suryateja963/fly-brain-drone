"""Compliance tests: geofence, height zones, wind, and the altitude cap.

The altitude cap tests matter most. It is enforced in the output bottleneck
rather than the planner precisely so it cannot be bypassed, and these tests
attempt to bypass it.
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

from geofence import Geofence, Polygon  # noqa: E402
from height_zones import HeightZone, HeightZoneManager  # noqa: E402
from output import OutputBottleneck  # noqa: E402
from wind import WindEstimator  # noqa: E402

SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
DT = 0.008


# ===========================================================================
# Polygon geometry
# ===========================================================================


def test_point_inside_square():
    poly = Polygon("test", SQUARE)
    assert poly.contains(np.array([5.0, 5.0]))


def test_point_outside_square():
    poly = Polygon("test", SQUARE)
    assert not poly.contains(np.array([50.0, 50.0]))


def test_point_just_outside_edge():
    poly = Polygon("test", SQUARE)
    assert not poly.contains(np.array([10.5, 5.0]))


def test_concave_polygon_handled():
    """Real no-fly zones are rarely convex; ray casting must cope.

    An L-shape: the notch must read as outside.
    """
    l_shape = [
        [0.0, 0.0], [10.0, 0.0], [10.0, 4.0],
        [4.0, 4.0], [4.0, 10.0], [0.0, 10.0],
    ]
    poly = Polygon("L", l_shape)

    assert poly.contains(np.array([2.0, 2.0]))
    assert poly.contains(np.array([2.0, 8.0]))
    assert not poly.contains(np.array([8.0, 8.0])), "notch read as inside"


def test_distance_to_boundary():
    poly = Polygon("test", SQUARE)
    assert poly.distance_to_boundary(np.array([5.0, 15.0])) == pytest.approx(5.0)


def test_escape_vector_points_outward_from_inside():
    poly = Polygon("test", SQUARE)
    # Near the right edge; the way out is +x.
    escape = poly.escape_vector(np.array([9.0, 5.0]))

    assert escape[0] > 0.9
    assert np.linalg.norm(escape) == pytest.approx(1.0, abs=1e-6)


def test_rejects_degenerate_polygon():
    with pytest.raises(ValueError):
        Polygon("bad", [[0.0, 0.0], [1.0, 1.0]])


# ===========================================================================
# Geofence
# ===========================================================================


def test_position_far_away_is_legal():
    fence = Geofence([Polygon("zone", SQUARE)], margin=2.0)
    status = fence.check(np.array([100.0, 100.0]))

    assert status.inside
    assert status.violation_direction is None


def test_position_inside_zone_is_illegal():
    fence = Geofence([Polygon("zone", SQUARE)], margin=2.0)
    status = fence.check(np.array([5.0, 5.0]))

    assert not status.inside
    assert status.violation_direction is not None


def test_margin_creates_standoff():
    """A control system aiming exactly at a line will cross it."""
    fence = Geofence([Polygon("zone", SQUARE)], margin=3.0)
    status = fence.check(np.array([12.0, 5.0]))  # 2m outside, inside margin

    assert not status.inside, "margin not enforced"


def test_escape_direction_points_away_when_in_margin():
    """Outside but too close: push AWAY, not toward the boundary."""
    fence = Geofence([Polygon("zone", SQUARE)], margin=3.0)
    status = fence.check(np.array([12.0, 5.0]))

    assert status.violation_direction is not None
    assert status.violation_direction[0] > 0.0, "steering into the zone"


def test_disabled_geofence_permits_everything():
    fence = Geofence([Polygon("zone", SQUARE)], margin=2.0, enabled=False)
    assert fence.check(np.array([5.0, 5.0])).inside


def test_path_crossing_zone_is_detected():
    """Route planning must see a violation before flying it."""
    fence = Geofence([Polygon("zone", SQUARE)], margin=1.0)

    assert fence.path_crosses_zone(np.array([-5.0, 5.0]), np.array([20.0, 5.0]))


def test_path_avoiding_zone_is_clear():
    fence = Geofence([Polygon("zone", SQUARE)], margin=1.0)

    assert not fence.path_crosses_zone(
        np.array([-5.0, 50.0]), np.array([20.0, 50.0])
    )


def test_multiple_zones_all_checked():
    fence = Geofence(
        [
            Polygon("a", SQUARE),
            Polygon("b", [[50.0, 50.0], [60.0, 50.0], [60.0, 60.0], [50.0, 60.0]]),
        ],
        margin=1.0,
    )

    assert not fence.check(np.array([5.0, 5.0])).inside
    assert not fence.check(np.array([55.0, 55.0])).inside
    assert fence.check(np.array([30.0, 30.0])).inside


# ===========================================================================
# Height zones
# ===========================================================================


def test_default_altitude_outside_zones():
    manager = HeightZoneManager([], default_altitude=1.5, global_ceiling=10.0)
    assert manager.target_for(np.array([0.0, 0.0])) == 1.5


def test_zone_target_applies_inside():
    zone = HeightZone("low", SQUARE, target_altitude=1.0, max_altitude=2.0)
    manager = HeightZoneManager([zone], default_altitude=5.0, global_ceiling=10.0)

    assert manager.target_for(np.array([5.0, 5.0])) == 1.0
    assert manager.target_for(np.array([50.0, 50.0])) == 5.0


def test_overlapping_zones_take_most_restrictive():
    """Two zones overlapping means two reasons to stay low."""
    a = HeightZone("a", SQUARE, target_altitude=3.0, max_altitude=4.0)
    b = HeightZone("b", SQUARE, target_altitude=1.0, max_altitude=2.0)
    manager = HeightZoneManager([a, b], default_altitude=8.0, global_ceiling=10.0)

    assert manager.target_for(np.array([5.0, 5.0])) == 1.0
    assert manager.ceiling_for(np.array([5.0, 5.0])) == 2.0


def test_zone_cannot_raise_global_ceiling():
    """A zone may lower the ceiling, never raise it."""
    zone = HeightZone("high", SQUARE, target_altitude=50.0, max_altitude=99.0)
    manager = HeightZoneManager([zone], default_altitude=1.5, global_ceiling=10.0)

    assert manager.ceiling_for(np.array([5.0, 5.0])) == 10.0
    assert manager.target_for(np.array([5.0, 5.0])) == 10.0


def test_violation_detected():
    zone = HeightZone("low", SQUARE, target_altitude=1.0, max_altitude=2.0)
    manager = HeightZoneManager([zone], global_ceiling=10.0)

    assert manager.violates(np.array([5.0, 5.0]), altitude=3.0)
    assert not manager.violates(np.array([5.0, 5.0]), altitude=1.5)


def test_active_zone_named_for_telemetry():
    zone = HeightZone("corridor", SQUARE, target_altitude=1.0, max_altitude=2.0)
    manager = HeightZoneManager([zone])

    assert manager.active_zone(np.array([5.0, 5.0])).name == "corridor"
    assert manager.active_zone(np.array([99.0, 99.0])) is None


def test_rejects_max_below_target():
    with pytest.raises(ValueError):
        HeightZone("bad", SQUARE, target_altitude=5.0, max_altitude=2.0)


# ===========================================================================
# Altitude cap — SAFETY CRITICAL, enforced at the bottleneck
# ===========================================================================


def test_climb_permitted_below_soft_limit():
    bottleneck = OutputBottleneck(ceiling_m=10.0, soft_limit_m=8.0)
    result = bottleneck.descend(0.0, 0.0, 0.0, vertical_input=2.0, altitude=3.0)

    assert not result.altitude_capped


def test_climb_authority_reduced_above_soft_limit():
    """Between soft limit and ceiling, authority scales toward zero."""
    bottleneck = OutputBottleneck(
        ceiling_m=10.0, soft_limit_m=8.0, base_thrust=0.0
    )

    low = bottleneck.descend(0.0, 0.0, 0.0, vertical_input=2.0, altitude=8.5)
    high = bottleneck.descend(0.0, 0.0, 0.0, vertical_input=2.0, altitude=9.5)

    assert low.altitude_capped and high.altitude_capped
    assert float(np.mean(high.motors)) < float(np.mean(low.motors))


def test_climb_refused_at_ceiling():
    """THE SAFETY TEST. A large climb command at the cap must not climb."""
    bottleneck = OutputBottleneck(
        ceiling_m=10.0, soft_limit_m=8.0, base_thrust=0.0
    )
    result = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=100.0, altitude=10.0
    )

    assert result.altitude_capped
    assert float(np.mean(result.motors)) <= 0.0, "climbed through the ceiling"


def test_ascent_actively_opposed_at_ceiling():
    """Arriving at the cap with upward momentum must be braked.

    Compared against a hovering baseline rather than against zero: real
    rotors cannot spin backwards, so min_motor clamps commands at 0 and the
    braking shows up as thrust BELOW hover, not as a negative number.
    """
    bottleneck = OutputBottleneck(
        ceiling_m=10.0, soft_limit_m=8.0, base_thrust=60.0
    )

    hovering = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=0.0, altitude=5.0, vertical_rate=0.0
    )
    climbing_at_cap = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=0.0, altitude=10.5, vertical_rate=3.0
    )

    assert float(np.mean(climbing_at_cap.motors)) < float(
        np.mean(hovering.motors)
    ), "ascent not opposed at the ceiling"


def test_descent_always_permitted():
    """The cap restricts climbing, never descending."""
    bottleneck = OutputBottleneck(
        ceiling_m=10.0, soft_limit_m=8.0, base_thrust=60.0
    )

    hovering = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=0.0, altitude=15.0
    )
    descending = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=-5.0, altitude=15.0
    )

    assert float(np.mean(descending.motors)) < float(np.mean(hovering.motors))


def test_planner_cannot_bypass_cap():
    """A target above the ceiling is clamped, and the cap still enforces.

    The clamp is a courtesy to the planner; the enforcement is independent.
    """
    bottleneck = OutputBottleneck(ceiling_m=10.0, soft_limit_m=8.0)

    assert bottleneck.would_exceed_ceiling(50.0)
    assert bottleneck.clamp_target_altitude(50.0) == 10.0

    result = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=100.0, altitude=12.0
    )
    assert result.altitude_capped


def test_disabled_cap_permits_climb():
    bottleneck = OutputBottleneck(
        ceiling_m=10.0, soft_limit_m=8.0, cap_enabled=False
    )
    result = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=5.0, altitude=50.0
    )

    assert not result.altitude_capped


def test_rejects_soft_limit_above_ceiling():
    with pytest.raises(ValueError):
        OutputBottleneck(ceiling_m=5.0, soft_limit_m=10.0)


# ===========================================================================
# Output bottleneck — the motor mix
# ===========================================================================


def test_produces_exactly_four_motors():
    bottleneck = OutputBottleneck()
    result = bottleneck.descend(0.0, 0.0, 0.0, 0.0, altitude=1.0)

    assert result.motors.shape == (4,)


def test_yaw_enters_with_alternating_sign():
    """Diagonal pairs counter-rotate; this is why large yaw tips the airframe."""
    bottleneck = OutputBottleneck(base_thrust=100.0)
    result = bottleneck.descend(
        0.0, 0.0, yaw_input=10.0, vertical_input=0.0, altitude=1.0
    )

    motors = result.motors
    # front-left and rear-right get -yaw; front-right and rear-left get +yaw.
    assert motors[0] < motors[1]
    assert motors[3] < motors[2]


def test_roll_differentiates_left_from_right():
    bottleneck = OutputBottleneck(base_thrust=100.0)
    result = bottleneck.descend(
        roll_input=10.0, pitch_input=0.0, yaw_input=0.0,
        vertical_input=0.0, altitude=1.0,
    )

    motors = result.motors
    assert motors[1] > motors[0]
    assert motors[3] > motors[2]


def test_motors_clamped_to_limits():
    bottleneck = OutputBottleneck(base_thrust=100.0, max_motor=120.0)
    result = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=500.0, altitude=1.0
    )

    assert result.clamped
    assert np.all(result.motors <= 120.0)


def test_motors_never_negative():
    bottleneck = OutputBottleneck(base_thrust=10.0, min_motor=0.0)
    result = bottleneck.descend(
        0.0, 0.0, 0.0, vertical_input=-500.0, altitude=5.0
    )

    assert np.all(result.motors >= 0.0)


# ===========================================================================
# Wind
# ===========================================================================


def test_no_wind_when_commanded_matches_actual():
    estimator = WindEstimator(window=10)
    for _ in range(20):
        state = estimator.update(
            commanded_velocity=np.array([1.0, 0.0]),
            actual_velocity=np.array([1.0, 0.0]),
            dt=DT,
        )

    assert state.speed == pytest.approx(0.0, abs=1e-6)


def test_steady_drift_estimated_as_wind():
    """Commanded to hold, drifted east: the wind is blowing east."""
    estimator = WindEstimator(window=10)
    for _ in range(20):
        state = estimator.update(
            commanded_velocity=np.array([0.0, 0.0]),
            actual_velocity=np.array([2.0, 0.0]),
            dt=DT,
        )

    assert state.speed == pytest.approx(2.0, abs=0.1)
    assert state.direction == pytest.approx(0.0, abs=0.1)


def test_wind_direction_recovered():
    estimator = WindEstimator(window=10)
    for _ in range(20):
        state = estimator.update(
            commanded_velocity=np.array([0.0, 0.0]),
            actual_velocity=np.array([0.0, 3.0]),
            dt=DT,
        )

    assert state.direction == pytest.approx(np.pi / 2, abs=0.1)


def test_compensation_opposes_wind():
    estimator = WindEstimator(window=10)
    for _ in range(20):
        estimator.update(
            commanded_velocity=np.array([0.0, 0.0]),
            actual_velocity=np.array([2.0, 0.0]),
            dt=DT,
        )

    compensation = estimator.compensation()
    assert compensation[0] < 0.0, "compensation reinforces the wind"


def test_gust_detected_on_sudden_change():
    estimator = WindEstimator(window=5, gust_threshold=1.5)

    for _ in range(10):
        estimator.update(np.array([0.0, 0.0]), np.array([0.1, 0.0]), DT)

    state = None
    for _ in range(5):
        state = estimator.update(np.array([0.0, 0.0]), np.array([8.0, 0.0]), DT)

    assert state.gust_detected


def test_steady_wind_is_not_a_permanent_gust():
    """The drone already compensates for steady wind; it needn't keep reacting."""
    estimator = WindEstimator(
        window=5, gust_threshold=1.5, recovery_timeout_s=0.02
    )

    for _ in range(200):
        state = estimator.update(
            np.array([0.0, 0.0]), np.array([3.0, 0.0]), dt=0.01
        )

    assert not state.gust_detected, "steady wind latched as a gust"


def test_window_smooths_transients():
    """A single control transient must not read as wind."""
    estimator = WindEstimator(window=20)

    for _ in range(19):
        estimator.update(np.array([1.0, 0.0]), np.array([1.0, 0.0]), DT)
    state = estimator.update(np.array([1.0, 0.0]), np.array([9.0, 0.0]), DT)

    assert state.speed < 1.0, f"transient dominated the estimate: {state.speed}"


def test_reset_clears_history():
    estimator = WindEstimator(window=5)
    for _ in range(10):
        estimator.update(np.array([0.0, 0.0]), np.array([5.0, 0.0]), DT)

    estimator.reset()
    state = estimator.update(np.array([0.0, 0.0]), np.array([0.0, 0.0]), DT)

    assert state.speed == pytest.approx(0.0, abs=1e-6)


def test_rejects_tiny_window():
    with pytest.raises(ValueError):
        WindEstimator(window=1)
