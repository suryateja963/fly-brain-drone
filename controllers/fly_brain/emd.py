"""EMD layer — medulla.

Reichardt correlators producing a local optic flow field.

THIS IS THE RISK. Reichardt detectors are easy to implement in a way that
looks plausible and points the wrong direction. Validate against synthetic
drifting gratings (tests/test_emd_grating.py) before connecting anything to a
motor.

Phase 2.
"""

import numpy as np

# ============================================================================
# SIGN CONVENTION — write it down, because you will forget.
#
#   horizontal > 0  means RIGHTWARD motion (image features moving +x)
#   vertical   > 0  means DOWNWARD  motion (image features moving +y, which
#                   is DOWN in image coordinates: row 0 is the top)
#
# Derivation, so this can be re-checked rather than re-guessed:
#
# For an adjacent pair A (left) and B (right), the correlator is
#
#     response = (A_delayed * B_now) - (B_delayed * A_now)
#
# Take a bright feature drifting RIGHT. It illuminates A first, then B one
# step later. A_delayed is therefore large at the same moment B_now is large,
# so the first product is large. Meanwhile B_delayed is still small (the
# feature had not reached B yet) while A_now has already dimmed, so the second
# product is small. Result: positive for rightward motion.
#
# The vertical case is identical with A = upper row, B = lower row, giving
# positive for downward motion.
#
# MEASURED, and it matters: the delay state must be advanced BEFORE the
# response is computed, not after. Updating it afterwards leaves `delayed`
# holding an EMA that already incorporates the current frame, so it leads
# rather than lags, and every response comes out inverted — rightward drift
# read -0.098 across every tau tested. The grating tests caught this before
# anything reached a motor, which is exactly what they are for.
# ============================================================================

DEFAULT_TAU = 0.6


class ReichardtArray:
    """A grid of correlators over the retina, one per adjacent pixel pair.

    The delay is a low-pass filter over past frames, not simply the previous
    frame:

        delayed = tau * delayed_prev + (1 - tau) * current

    A single previous frame gives a delay whose length is tied to the frame
    rate. The exponential moving average decouples the two, so tau can be
    tuned against the speeds the drone actually flies.
    """

    def __init__(self, shape, tau=DEFAULT_TAU):
        """Args:
        shape: (rows, cols) of the retina feeding this array.
        tau: time constant of the delay filter, in [0, 1). Higher is slower.
        """
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"tau must be in [0, 1), got {tau}")

        self.shape = tuple(shape)
        self.tau = float(tau)
        self.delayed = None

    def reset(self):
        """Forget the delay state, as if no frame had been seen."""
        self.delayed = None

    def update(self, retina):
        """Advance one timestep and return the local flow field.

        Args:
            retina: float array from `retina.sample`.

        Returns:
            (horizontal, vertical) float arrays of local flow responses.
            Horizontal has shape (rows, cols - 1), vertical (rows - 1, cols):
            one correlator per adjacent PAIR, so each is one narrower than the
            retina along the axis it measures.

            On the first call, before any delayed state exists, both are zero:
            a correlator with no history cannot report motion, and returning
            zeros is more honest than comparing a frame against itself.
        """
        current = np.asarray(retina, dtype=np.float64)
        if current.shape != self.shape:
            raise ValueError(
                f"expected retina of shape {self.shape}, got {current.shape}"
            )

        if self.delayed is None:
            self.delayed = current.copy()
            return (
                np.zeros((self.shape[0], self.shape[1] - 1)),
                np.zeros((self.shape[0] - 1, self.shape[1])),
            )

        # Snapshot the delay line BEFORE advancing it: this is the lagged
        # signal the correlator needs. Advancing first (or computing against
        # an already-advanced state) inverts every response.
        delayed = self.delayed
        self.delayed = self.tau * delayed + (1.0 - self.tau) * current

        # Horizontal pairs: A is the left pixel, B the right.
        a_delayed = delayed[:, :-1]
        b_delayed = delayed[:, 1:]
        a_now = current[:, :-1]
        b_now = current[:, 1:]
        horizontal = (a_delayed * b_now) - (b_delayed * a_now)

        # Vertical pairs: A is the upper row, B the lower.
        a_delayed = delayed[:-1, :]
        b_delayed = delayed[1:, :]
        a_now = current[:-1, :]
        b_now = current[1:, :]
        vertical = (a_delayed * b_now) - (b_delayed * a_now)

        return horizontal, vertical
