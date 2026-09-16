"""Wide-field pooling tests, with rotation subtraction as the centrepiece.

The confound these guard against: yawing produces uniform horizontal flow
across the whole field, which reads as expansion and creates a phantom
obstacle every time the drone turns in open space.
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

from contracts import FlowField  # noqa: E402
from wide_field import WideFieldPooling  # noqa: E402

ROWS, COLS = 20, 40


def make_flow(horizontal=0.0, vertical=0.0):
    """A uniform flow field, the simplest case to reason about."""
    flow = np.zeros((ROWS, COLS, 2), dtype=np.float32)
    flow[:, :, 0] = horizontal
    flow[:, :, 1] = vertical
    return FlowField(timestamp=0.0, flow=flow, valid=True)


def make_expansion_flow(left_magnitude, right_magnitude):
    """Outward flow: leftward on the left half, rightward on the right."""
    flow = np.zeros((ROWS, COLS, 2), dtype=np.float32)
    half = COLS // 2
    flow[:, :half, 0] = -left_magnitude
    flow[:, half:, 0] = +right_magnitude
    return FlowField(timestamp=0.0, flow=flow, valid=True)


def test_invalid_flow_returns_zero_channels():
    """A flow field flagged invalid must not produce phantom signal."""
    pooling = WideFieldPooling()
    flow = make_flow(horizontal=5.0)
    flow.valid = False

    channels = pooling.pool(flow, yaw_rate=0.0)

    assert channels.left_expansion == 0.0
    assert channels.right_expansion == 0.0
    assert channels.total_expansion == 0.0


def test_symmetric_expansion_gives_zero_difference():
    """Equal obstacles both sides produce no turn signal."""
    pooling = WideFieldPooling()
    channels = pooling.pool(make_expansion_flow(1.0, 1.0), yaw_rate=0.0)

    assert channels.left_expansion == pytest.approx(1.0, abs=1e-5)
    assert channels.right_expansion == pytest.approx(1.0, abs=1e-5)
    assert channels.expansion_difference == pytest.approx(0.0, abs=1e-5)


def test_left_obstacle_raises_left_channel_alone():
    """Flying past a pillar on the left spikes left expansion only."""
    pooling = WideFieldPooling()
    channels = pooling.pool(make_expansion_flow(2.0, 0.0), yaw_rate=0.0)

    assert channels.left_expansion > channels.right_expansion
    assert channels.expansion_difference > 0.0


def test_yaw_in_open_space_does_not_create_expansion():
    """THE CRITICAL TEST.

    Pure yaw in open space produces uniform horizontal flow. Without
    rotation subtraction this reads as an obstacle, and the drone swerves
    away from nothing. With subtraction, expansion must stay near zero.
    """
    pooling = WideFieldPooling(yaw_rate_to_flow=1.0)

    yaw_rate = 3.0
    # A pure yaw of this rate produces exactly this uniform flow.
    flow = make_flow(horizontal=yaw_rate)

    channels = pooling.pool(flow, yaw_rate=yaw_rate)

    assert channels.total_expansion == pytest.approx(0.0, abs=1e-5), (
        f"phantom obstacle from yaw: total={channels.total_expansion}"
    )
    assert channels.expansion_difference == pytest.approx(0.0, abs=1e-5)


def test_rotation_subtraction_is_auditable():
    """The pre-subtraction value is retained so the fix is visible."""
    pooling = WideFieldPooling(yaw_rate_to_flow=1.0)
    yaw_rate = 2.0
    channels = pooling.pool(make_flow(horizontal=yaw_rate), yaw_rate=yaw_rate)

    assert channels.raw_rotation_before_subtraction == pytest.approx(2.0, abs=1e-5)
    assert channels.rotation == pytest.approx(0.0, abs=1e-5)


def test_uncorrected_yaw_would_have_produced_phantom_signal():
    """Prove the correction is doing work, not merely present.

    Pooling the same flow with a zero yaw-rate input models the
    uncorrected case; expansion must then be non-zero, which is exactly the
    failure mode the subtraction exists to prevent.
    """
    pooling = WideFieldPooling(yaw_rate_to_flow=1.0)
    flow = make_flow(horizontal=3.0)

    corrected = pooling.pool(flow, yaw_rate=3.0)
    uncorrected = pooling.pool(flow, yaw_rate=0.0)

    assert abs(uncorrected.expansion_difference) > 1.0
    assert abs(corrected.expansion_difference) < 1e-5


def test_obstacle_survives_simultaneous_yaw():
    """A real obstacle must still register while the drone is yawing.

    Over-subtraction would be as bad as none: the correction must remove the
    rotational component without removing the obstacle with it.
    """
    pooling = WideFieldPooling(yaw_rate_to_flow=1.0)

    yaw_rate = 1.5
    flow = make_expansion_flow(2.0, 0.0)
    # Superimpose the rotational component on the obstacle flow.
    flow.flow[:, :, 0] += yaw_rate

    channels = pooling.pool(flow, yaw_rate=yaw_rate)

    assert channels.expansion_difference > 0.5, (
        "obstacle was subtracted away along with the rotation"
    )


def test_vertical_channel_responds_to_vertical_flow():
    pooling = WideFieldPooling()
    channels = pooling.pool(make_flow(vertical=1.5), yaw_rate=0.0)

    assert channels.vertical_expansion == pytest.approx(1.5, abs=1e-5)


def test_horizon_rows_excluded_from_expansion():
    """Sky rows must not contribute to lateral expansion."""
    pooling = WideFieldPooling(horizon_row_start=10)

    flow = np.zeros((ROWS, COLS, 2), dtype=np.float32)
    # Signal ONLY in the excluded band.
    flow[:10, : COLS // 2, 0] = -5.0
    field = FlowField(timestamp=0.0, flow=flow, valid=True)

    channels = pooling.pool(field, yaw_rate=0.0)

    assert channels.left_expansion == pytest.approx(0.0, abs=1e-5)


def test_rejects_invalid_split_fraction():
    with pytest.raises(ValueError):
        WideFieldPooling(split_fraction=0.0)
    with pytest.raises(ValueError):
        WideFieldPooling(split_fraction=1.0)


def test_rejects_horizon_beyond_field():
    with pytest.raises(ValueError):
        WideFieldPooling(rows=20, horizon_row_start=20)
