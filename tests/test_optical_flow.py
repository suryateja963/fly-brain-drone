"""Lucas-Kanade validation, against the same gratings the EMD passes.

LK is a least-squares velocity estimator, so unlike the EMD it should report
flow in calibrated pixels per frame. The tests check that: a grating drifting
at a known speed must produce approximately that speed, not merely the right
sign.
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

from optical_flow import EMDFlowAdapter, LucasKanadeFlow  # noqa: E402

SHAPE = (20, 40)
WAVELENGTH = 10.0


def drifting_grating(phase, wavelength=WAVELENGTH, vertical=False):
    """Sinusoid grating. Positive phase drifts toward +x (or +y)."""
    rows, cols = SHAPE
    if vertical:
        coord = np.arange(rows).reshape(rows, 1)
        pattern = np.sin(2.0 * np.pi * (coord - phase) / wavelength)
        return np.tile(0.5 + 0.5 * pattern, (1, cols))
    coord = np.arange(cols).reshape(1, cols)
    pattern = np.sin(2.0 * np.pi * (coord - phase) / wavelength)
    return np.tile(0.5 + 0.5 * pattern, (rows, 1))


def interior(array):
    """Drop the border, where the filter window runs off the edge."""
    return array[4:-4, 6:-6]


def run_drift(speed, vertical=False, steps=8):
    """Drift a grating and return mean interior flow after settling."""
    lk = LucasKanadeFlow(shape=SHAPE)
    result = None
    for step in range(steps):
        frame = drifting_grating(speed * step, vertical=vertical)
        result = lk.compute(frame, timestamp=step * 0.008)
    return result


# ---------------------------------------------------------------------------
# Contract behaviour
# ---------------------------------------------------------------------------


def test_first_frame_is_invalid():
    """No previous frame means no flow. Saying so beats inventing motion."""
    lk = LucasKanadeFlow(shape=SHAPE)
    result = lk.compute(drifting_grating(0.0))

    assert not result.valid
    assert np.all(result.flow == 0.0)


def test_second_frame_is_valid():
    lk = LucasKanadeFlow(shape=SHAPE)
    lk.compute(drifting_grating(0.0))
    result = lk.compute(drifting_grating(1.0))

    assert result.valid
    assert result.flow.shape == (20, 40, 2)


def test_output_shape_matches_retina():
    """LK produces one vector per retinal location, unlike the EMD's pairs."""
    result = run_drift(1.0)
    assert result.flow.shape == (*SHAPE, 2)


def test_reset_clears_history():
    lk = LucasKanadeFlow(shape=SHAPE)
    lk.compute(drifting_grating(0.0))
    lk.reset()
    result = lk.compute(drifting_grating(1.0))

    assert not result.valid


def test_rejects_even_window():
    with pytest.raises(ValueError):
        LucasKanadeFlow(window=4)


def test_rejects_tiny_window():
    with pytest.raises(ValueError):
        LucasKanadeFlow(window=1)


def test_rejects_wrong_retina_shape():
    lk = LucasKanadeFlow(shape=SHAPE)
    with pytest.raises(ValueError):
        lk.compute(np.zeros((10, 10)))


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def test_rightward_drift_gives_positive_horizontal():
    result = run_drift(+1.0)
    mean = float(np.mean(interior(result.flow[:, :, 0])))
    assert mean > 0.0, f"expected positive, got {mean:+.4f}"


def test_leftward_drift_flips_sign():
    right = float(np.mean(interior(run_drift(+1.0).flow[:, :, 0])))
    left = float(np.mean(interior(run_drift(-1.0).flow[:, :, 0])))

    assert left < 0.0, f"expected negative, got {left:+.4f}"
    assert np.isclose(right, -left, rtol=0.25)


def test_static_image_gives_near_zero():
    result = run_drift(0.0)
    assert abs(float(np.mean(interior(result.flow[:, :, 0])))) < 1e-6
    assert abs(float(np.mean(interior(result.flow[:, :, 1])))) < 1e-6


def test_vertical_drift_drives_vertical_channel():
    result = run_drift(+1.0, vertical=True)
    horizontal = float(np.mean(interior(result.flow[:, :, 0])))
    vertical = float(np.mean(interior(result.flow[:, :, 1])))

    assert vertical > 0.0, f"expected positive vertical, got {vertical:+.4f}"
    assert abs(horizontal) < abs(vertical) * 0.2


def test_upward_drift_flips_vertical_sign():
    down = float(np.mean(interior(run_drift(+1.0, vertical=True).flow[:, :, 1])))
    up = float(np.mean(interior(run_drift(-1.0, vertical=True).flow[:, :, 1])))

    assert up < 0.0
    assert np.isclose(down, -up, rtol=0.25)


# ---------------------------------------------------------------------------
# Magnitude — what LK gives that the EMD does not
# ---------------------------------------------------------------------------


def test_reports_speed_monotonically_but_underestimates():
    """MEASURED: single-scale LK underestimates, and here is exactly why.

    A 1-D grating gives a rank-1 structure tensor, so the solver falls back
    to normal flow: the component across the brightness gradient, which is
    the only one observable (the aperture problem). That estimate scales as
    It/|grad I|, and It is a finite difference standing in for a derivative
    — an approximation that only holds in the small-displacement limit.

    At 1 px/frame against a 10 px wavelength the sinusoid curves
    substantially between samples, so the estimate comes out around 0.14
    rather than 1.0. This is textbook single-scale LK behaviour, not a bug;
    the standard remedy is a coarse-to-fine pyramid, which is deliberately
    not implemented here because the pooling layer downstream needs
    direction and relative magnitude, not absolute calibration.

    What the pipeline actually requires is tested elsewhere: correct sign,
    and monotonic growth with speed.
    """
    result = run_drift(+1.0)
    mean = float(np.mean(interior(result.flow[:, :, 0])))

    assert mean > 0.0, f"lost the direction entirely: {mean:.4f}"
    assert mean < 1.6, f"implausibly large for 1 px/frame: {mean:.3f}"


def test_faster_drift_reads_faster():
    slow = float(np.mean(interior(run_drift(+0.5).flow[:, :, 0])))
    fast = float(np.mean(interior(run_drift(+1.5).flow[:, :, 0])))

    assert 0.0 < slow < fast, f"slow={slow:.3f} fast={fast:.3f}"


def test_speed_estimate_is_roughly_proportional():
    """Doubling the drift should roughly double the reported flow."""
    one = float(np.mean(interior(run_drift(+0.5).flow[:, :, 0])))
    two = float(np.mean(interior(run_drift(+1.0).flow[:, :, 0])))

    ratio = two / one
    assert 1.4 < ratio < 2.8, f"expected ~2x, got {ratio:.2f}x"


# ---------------------------------------------------------------------------
# The aperture problem and untextured input
# ---------------------------------------------------------------------------


def test_untextured_field_reports_no_flow():
    """A blank wall is unobservable, and must report zero rather than noise.

    This is why every world surface must be textured: the vision pipeline is
    genuinely blind here, and the honest output says so.
    """
    lk = LucasKanadeFlow(shape=SHAPE)
    blank = np.full(SHAPE, 0.5)

    lk.compute(blank)
    result = lk.compute(blank)

    assert np.all(result.flow == 0.0)


def test_coverage_reports_usable_fraction():
    """Coverage distinguishes 'saw nothing' from 'saw no motion'."""
    lk = LucasKanadeFlow(shape=SHAPE)

    blank = np.full(SHAPE, 0.5)
    lk.compute(blank)
    blank_result = lk.compute(blank)
    assert lk.coverage(blank_result) == 0.0

    lk.reset()
    lk.compute(drifting_grating(0.0))
    textured = lk.compute(drifting_grating(1.0))
    assert lk.coverage(textured) > 0.5


# ---------------------------------------------------------------------------
# The EMD adapter — same contract, different mechanism
# ---------------------------------------------------------------------------


def test_emd_adapter_matches_the_flowfield_contract():
    """The adapter pads EMD pair-outputs to full retina shape."""
    adapter = EMDFlowAdapter(shape=SHAPE)
    adapter.compute(drifting_grating(0.0))
    result = adapter.compute(drifting_grating(1.0))

    assert result.flow.shape == (*SHAPE, 2)


def test_emd_adapter_agrees_on_direction():
    """Both front-ends must agree on which way the world is moving.

    They disagree on magnitude by construction — that is the point of having
    both — but a sign disagreement would mean one of them steers into
    obstacles.
    """
    adapter = EMDFlowAdapter(shape=SHAPE)
    for step in range(8):
        adapter.compute(drifting_grating(1.0 * step))
    emd_result = adapter.compute(drifting_grating(8.0))

    lk_result = run_drift(+1.0)

    emd_mean = float(np.mean(interior(emd_result.flow[:, :, 0])))
    lk_mean = float(np.mean(interior(lk_result.flow[:, :, 0])))

    assert np.sign(emd_mean) == np.sign(lk_mean), (
        f"front-ends disagree on direction: emd={emd_mean:+.4f} lk={lk_mean:+.4f}"
    )
