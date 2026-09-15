"""Synthetic validation of the EMD layer — MANDATORY before Phase 3.

Feed the correlators drifting sinusoid gratings of known direction and speed.
Needs no simulator; run with `pytest`.

If the flow looks noisy but directionally correct, that is fine and expected.
If it looks clean but points the wrong way, stop and fix it.

Phase 2.
"""

import pytest


def drifting_grating(shape, phase, wavelength=8.0, vertical=False):
    """Render one frame of a sinusoid grating at the given phase.

    Args:
        shape: (rows, cols).
        phase: phase offset in pixels; advance it between frames to drift.
        wavelength: grating period in pixels.
        vertical: drift vertically instead of horizontally.

    Returns:
        float array of `shape`, values in [0, 1].
    """
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2 not implemented")
def test_rightward_grating_gives_positive_response():
    """A rightward-drifting grating must give a positive mean response.

    See the sign convention in emd.py.
    """


@pytest.mark.skip(reason="Phase 2 not implemented")
def test_leftward_grating_flips_sign():
    """The same grating drifting left must flip the sign."""


@pytest.mark.skip(reason="Phase 2 not implemented")
def test_static_image_gives_near_zero():
    """A static image must produce near-zero output."""


@pytest.mark.skip(reason="Phase 2 not implemented")
def test_vertical_grating_drives_vertical_channel():
    """Vertical drift must appear in the vertical channel, not the horizontal."""
