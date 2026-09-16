"""Retina — ommatidia.

Coarse, blurry sampling. The agent deliberately sees far less than the viewer:
optic flow is more tractable on blurred input, because fine texture creates
aliasing in the correlators downstream.

Phase 2.
"""

import cv2
import numpy as np

RETINA_SHAPE = (20, 40)  # (rows, cols) — height, width
DEFAULT_BLUR_SIGMA = 1.0


def sample(frame, shape=RETINA_SHAPE, blur_sigma=DEFAULT_BLUR_SIGMA):
    """Reduce a full-res camera frame to the fly's view.

    frame -> greyscale -> gaussian blur -> resize -> float [0, 1]

    Blur happens BEFORE the resize, deliberately: at full resolution it acts
    as the anti-aliasing filter the downsample needs. Blurring afterwards
    would smooth an already-aliased image, which cannot recover the motion
    signal the correlators depend on.

    Args:
        frame: HxWx3 (or HxWx4) uint8 array. Webots cameras hand back BGRA.
        shape: (rows, cols) of the output.
        blur_sigma: gaussian sigma applied at full resolution. 0 disables.

    Returns:
        float64 array of `shape`, values in [0, 1].
    """
    array = np.asarray(frame)

    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 4:
            grey = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif channels == 3:
            grey = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"unsupported channel count: {channels}")
    elif array.ndim == 2:
        grey = array
    else:
        raise ValueError(f"expected a 2D or 3D array, got {array.ndim}D")

    if blur_sigma > 0:
        grey = cv2.GaussianBlur(grey, ksize=(0, 0), sigmaX=blur_sigma)

    rows, cols = shape
    # INTER_AREA averages over the source region, which is the right choice
    # for downsampling: it is itself a low-pass filter.
    small = cv2.resize(grey, (cols, rows), interpolation=cv2.INTER_AREA)

    if small.dtype == np.uint8:
        return small.astype(np.float64) / 255.0
    return np.clip(small.astype(np.float64), 0.0, 1.0)


def from_webots(camera, shape=RETINA_SHAPE, blur_sigma=DEFAULT_BLUR_SIGMA):
    """Sample straight from a Webots camera device.

    getImage() returns raw BGRA bytes, height * width * 4. Reshaping here
    keeps the byte layout knowledge in one place.

    Returns:
        float array of `shape`, or None if no frame is available yet.
    """
    raw = camera.getImage()
    if raw is None:
        return None

    height = camera.getHeight()
    width = camera.getWidth()
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return sample(frame, shape=shape, blur_sigma=blur_sigma)
