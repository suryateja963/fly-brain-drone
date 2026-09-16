"""Synthetic validation of the EMD layer — MANDATORY before Phase 3.

Feed the correlators drifting sinusoid gratings of known direction and speed.
Needs no simulator; run with `pytest`.

If the flow looks noisy but directionally correct, that is fine and expected.
If it looks clean but points the wrong way, stop and fix it.

Phase 2.
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

from emd import ReichardtArray  # noqa: E402

SHAPE = (20, 40)
WAVELENGTH = 8.0
STEPS = 40


def drifting_grating(shape, phase, wavelength=WAVELENGTH, vertical=False):
    """Render one frame of a sinusoid grating at the given phase.

    Args:
        shape: (rows, cols).
        phase: phase offset in pixels; advance it between frames to drift.
        wavelength: grating period in pixels.
        vertical: vary along rows instead of columns, so the pattern drifts
            vertically rather than horizontally.

    Returns:
        float array of `shape`, values in [0, 1].
    """
    # SIGN: the phase is SUBTRACTED, not added. A feature sits where
    # coord - phase is constant, so coord = k + phase — it moves toward +x as
    # phase grows, which is what "positive speed drifts rightward" must mean.
    #
    # Adding the phase drifts the pattern LEFTWARD, and that bug made a
    # correct correlator look inverted: every "rightward" case was really
    # feeding leftward motion. A two-pixel trace (a feature stepping from the
    # left pixel to the right) gave +0.400, proving the correlator right and
    # the generator wrong. Test harnesses get validated too.
    rows, cols = shape
    if vertical:
        coord = np.arange(rows).reshape(rows, 1)
        pattern = np.sin(2.0 * np.pi * (coord - phase) / wavelength)
        return np.tile(0.5 + 0.5 * pattern, (1, cols))

    coord = np.arange(cols).reshape(1, cols)
    pattern = np.sin(2.0 * np.pi * (coord - phase) / wavelength)
    return np.tile(0.5 + 0.5 * pattern, (rows, 1))


def run_grating(speed, vertical=False, steps=STEPS, tau=0.6):
    """Drift a grating past the correlators and return mean responses.

    A positive `speed` advances the pattern toward +x (or +y when vertical).

    Returns:
        (mean_horizontal, mean_vertical) over the second half of the run,
        once the delay filter has settled.
    """
    emd = ReichardtArray(SHAPE, tau=tau)
    h_samples = []
    v_samples = []

    for step in range(steps):
        frame = drifting_grating(SHAPE, phase=speed * step, vertical=vertical)
        horizontal, vertical_response = emd.update(frame)
        if step >= steps // 2:
            h_samples.append(horizontal.mean())
            v_samples.append(vertical_response.mean())

    return float(np.mean(h_samples)), float(np.mean(v_samples))


def test_rightward_grating_gives_positive_response():
    """A rightward-drifting grating must give a positive mean response.

    See the sign convention derivation at the top of emd.py: a feature
    reaching the left detector first makes the first product dominate.
    """
    horizontal, _ = run_grating(speed=+1.0)
    assert horizontal > 0.0, f"expected positive, got {horizontal:+.6f}"


def test_leftward_grating_flips_sign():
    """The same grating drifting left must flip the sign."""
    rightward, _ = run_grating(speed=+1.0)
    leftward, _ = run_grating(speed=-1.0)

    assert leftward < 0.0, f"expected negative, got {leftward:+.6f}"
    # Symmetric drift should give symmetric magnitude.
    assert np.isclose(rightward, -leftward, rtol=0.2), (
        f"asymmetric: {rightward:+.6f} vs {leftward:+.6f}"
    )


def test_static_image_gives_near_zero():
    """A static image must produce near-zero output."""
    horizontal, vertical = run_grating(speed=0.0)
    assert abs(horizontal) < 1e-9, f"horizontal not still: {horizontal:+.6e}"
    assert abs(vertical) < 1e-9, f"vertical not still: {vertical:+.6e}"


def test_vertical_grating_drives_vertical_channel():
    """Vertical drift must appear in the vertical channel, not the horizontal."""
    horizontal, vertical = run_grating(speed=+1.0, vertical=True)

    assert vertical > 0.0, f"expected positive vertical, got {vertical:+.6f}"
    assert abs(horizontal) < abs(vertical) * 0.01, (
        f"leaked into horizontal: h={horizontal:+.6e} v={vertical:+.6e}"
    )


def test_upward_grating_flips_vertical_sign():
    """Upward drift must flip the vertical channel's sign."""
    _, downward = run_grating(speed=+1.0, vertical=True)
    _, upward = run_grating(speed=-1.0, vertical=True)

    assert upward < 0.0, f"expected negative, got {upward:+.6f}"
    assert np.isclose(downward, -upward, rtol=0.2)


def test_first_update_reports_no_motion():
    """A correlator with no history must report zeros, not noise.

    Comparing a frame against itself would fabricate a motion signal from a
    single observation.
    """
    emd = ReichardtArray(SHAPE)
    horizontal, vertical = emd.update(drifting_grating(SHAPE, phase=0.0))

    assert np.all(horizontal == 0.0)
    assert np.all(vertical == 0.0)


def test_response_grows_with_speed():
    """Faster drift must give a stronger response, over a sane speed range.

    Reichardt detectors are tuned to a preferred temporal frequency and fall
    off past it, so this only holds below that peak — which is why the range
    tested here is deliberately narrow.
    """
    slow, _ = run_grating(speed=+0.25)
    fast, _ = run_grating(speed=+0.75)

    assert 0.0 < slow < fast, f"slow={slow:+.6f} fast={fast:+.6f}"


@pytest.mark.parametrize("tau", [0.3, 0.6, 0.85])
def test_sign_is_stable_across_tau(tau):
    """The sign convention must not depend on the delay constant."""
    horizontal, _ = run_grating(speed=+1.0, tau=tau)
    assert horizontal > 0.0, f"tau={tau} gave {horizontal:+.6f}"
