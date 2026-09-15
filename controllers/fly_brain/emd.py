"""EMD layer — medulla.

Reichardt correlators producing a local optic flow field.

THIS IS THE RISK. Reichardt detectors are easy to implement in a way that
looks plausible and points the wrong direction. Validate against synthetic
drifting gratings (tests/test_emd_grating.py) before connecting anything to a
motor.

Phase 2.
"""

# SIGN CONVENTION: positive horizontal response means RIGHTWARD motion.
# Positive vertical response means DOWNWARD motion.
# Write it down, because you will forget.

DEFAULT_TAU = 0.6


class ReichardtArray:
    """A grid of correlators over the retina, one per adjacent pixel pair.

    For each adjacent pair (A, B):

        response = (A_delayed * B_now) - (B_delayed * A_now)

    The delay is a low-pass filter over past frames, not simply the previous
    frame:

        delayed = tau * delayed_prev + (1 - tau) * current
    """

    def __init__(self, shape, tau=DEFAULT_TAU):
        """Args:
        shape: (rows, cols) of the retina feeding this array.
        tau: time constant of the delay filter, in [0, 1). Higher is slower.
        """
        raise NotImplementedError("Phase 2")

    def update(self, retina):
        """Advance one timestep and return the local flow field.

        Args:
            retina: float array from `retina.sample`.

        Returns:
            (horizontal, vertical) float arrays of local flow responses.
            On the first call, before any delayed state exists, both are zero.
        """
        raise NotImplementedError("Phase 2")
