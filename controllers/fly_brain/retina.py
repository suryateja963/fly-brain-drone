"""Retina — ommatidia.

Coarse, blurry sampling. The agent deliberately sees far less than the viewer:
optic flow is more tractable on blurred input, because fine texture creates
aliasing in the correlators downstream.

Phase 2.
"""

RETINA_SHAPE = (20, 40)  # (rows, cols) — height, width


def sample(frame, shape=RETINA_SHAPE, blur_sigma=1.0):
    """Reduce a full-res RGB camera frame to the fly's view.

    frame -> greyscale -> gaussian blur -> resize -> float [0, 1]

    Blur *before* resizing, so the blur acts as the anti-aliasing filter that
    the downsample needs.

    Args:
        frame: HxWx3 uint8 array from the Webots camera.
        shape: (rows, cols) of the output.
        blur_sigma: gaussian sigma applied at full resolution.

    Returns:
        float array of `shape`, values in [0, 1].
    """
    raise NotImplementedError("Phase 2")
