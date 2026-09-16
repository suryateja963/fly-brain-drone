"""Lucas-Kanade optic flow — the engineering front-end.

Runs alongside emd.py, which stays as the biological reference. Both produce
the same FlowField contract, so the pipeline downstream cannot tell them
apart and either can drive the drone.

WHY BOTH EXIST, and why this one is primary:

  Lucas-Kanade solves for the flow vector that best explains the observed
  brightness change over a local window. It is a least-squares estimator: it
  produces a velocity in pixels per frame, with a condition number that says
  how much to trust each estimate.

  Reichardt correlators are what the fly actually has. They produce a
  correlation strength, not a velocity — the response confounds speed with
  contrast and pattern wavelength, and peaks at a preferred temporal
  frequency rather than growing monotonically with speed.

  For flying a drone, LK's calibrated velocities are easier to pool and
  threshold. For claiming the architecture is fly-derived, the EMD is the
  honest one. Keeping both means the claim survives and the drone flies well.

THE APERTURE PROBLEM is why this uses a window rather than a per-pixel
solve: a single pixel's brightness change is consistent with infinitely many
motions (only the component along the brightness gradient is observable).
Assuming flow is constant across a small neighbourhood gives enough
equations to solve for both components — that assumption IS the algorithm.
"""

import cv2
import numpy as np

from contracts import FlowField

DEFAULT_WINDOW = 5
DEFAULT_MIN_EIGENVALUE = 1e-4


class LucasKanadeFlow:
    """Dense Lucas-Kanade optic flow over the coarse retina.

    Dense rather than sparse (feature-tracked) because the pooling layer
    downstream wants flow everywhere, not at corners: expansion is a
    wide-field average, and a sparse field would bias it toward whatever
    happens to be textured.
    """

    __slots__ = ("_window", "_min_eigenvalue", "_previous", "_shape", "_kernel")

    def __init__(
        self,
        shape=(20, 40),
        window: int = DEFAULT_WINDOW,
        min_eigenvalue: float = DEFAULT_MIN_EIGENVALUE,
    ) -> None:
        """Args:
        shape: (rows, cols) of the retina.
        window: side length of the neighbourhood assumed to share one flow
            vector. Must be odd. Larger windows are more robust to noise but
            blur genuine flow discontinuities — at 40 columns wide, 5 is
            already an eighth of the field.
        min_eigenvalue: below this, the local structure tensor is singular
            and the flow is unobservable (a blank wall, or an edge with the
            aperture problem). Those locations report zero and are excluded
            rather than returning a fabricated vector.
        """
        if window % 2 == 0:
            raise ValueError(f"window must be odd, got {window}")
        if window < 3:
            raise ValueError(f"window must be at least 3, got {window}")

        self._shape = tuple(shape)
        self._window = window
        self._min_eigenvalue = min_eigenvalue
        self._previous = None
        # Uniform averaging kernel for the structure-tensor sums.
        self._kernel = np.ones((window, window), dtype=np.float64)

    def reset(self) -> None:
        """Forget the previous frame, as if no frame had been seen."""
        self._previous = None

    def compute(self, retina: np.ndarray, timestamp: float = 0.0) -> FlowField:
        """Estimate dense flow between the previous frame and this one.

        Args:
            retina: float array of `shape`, values in [0, 1].
            timestamp: seconds, carried through to the contract.

        Returns:
            FlowField. On the first call `valid` is False and the flow is
            zero: an estimator with no previous frame has nothing to
            difference against, and saying so beats inventing motion.
        """
        current = np.asarray(retina, dtype=np.float64)
        if current.shape != self._shape:
            raise ValueError(
                f"expected retina of shape {self._shape}, got {current.shape}"
            )

        if self._previous is None:
            self._previous = current.copy()
            return FlowField(
                timestamp=timestamp,
                flow=np.zeros((*self._shape, 2), dtype=np.float32),
                valid=False,
            )

        previous = self._previous
        self._previous = current.copy()

        flow = self._solve(previous, current)
        return FlowField(timestamp=timestamp, flow=flow, valid=True)

    def _solve(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """The least-squares solve, vectorised across the whole field.

        For each location we solve the 2x2 system

            [ SIxx  SIxy ] [ u ]     [ -SIxt ]
            [ SIxy  SIyy ] [ v ]  =  [ -SIyt ]

        where each S is a windowed sum of products of image derivatives.
        That matrix is the structure tensor; its eigenvalues say whether the
        local patch has enough texture in enough directions to pin down both
        components of motion.
        """
        # Spatial derivatives on the average of the two frames: centring the
        # gradient between them is a better linearisation of the brightness
        # constancy equation than taking either frame alone.
        mid = 0.5 * (previous + current)
        ix = cv2.Sobel(mid, cv2.CV_64F, 1, 0, ksize=3)
        iy = cv2.Sobel(mid, cv2.CV_64F, 0, 1, ksize=3)
        it = current - previous

        # Windowed sums = the structure tensor entries, per location.
        sxx = cv2.filter2D(ix * ix, cv2.CV_64F, self._kernel)
        syy = cv2.filter2D(iy * iy, cv2.CV_64F, self._kernel)
        sxy = cv2.filter2D(ix * iy, cv2.CV_64F, self._kernel)
        sxt = cv2.filter2D(ix * it, cv2.CV_64F, self._kernel)
        syt = cv2.filter2D(iy * it, cv2.CV_64F, self._kernel)

        determinant = sxx * syy - sxy * sxy
        trace = sxx + syy
        discriminant = np.sqrt(np.maximum(trace * trace - 4.0 * determinant, 0.0))
        min_eigenvalue = 0.5 * (trace - discriminant)

        # RANK-2: both flow components observable. The patch has texture in
        # two directions, so the full system has a unique solution.
        full_rank = (min_eigenvalue > self._min_eigenvalue) & (
            np.abs(determinant) > 1e-12
        )

        safe_determinant = np.where(full_rank, determinant, 1.0)

        # Cramer's rule on the 2x2 system. The negation carries the sign
        # convention: positive u means image features moving toward +x.
        u_full = -(syy * sxt - sxy * syt) / safe_determinant
        v_full = -(sxx * syt - sxy * sxt) / safe_determinant

        # RANK-1: only one direction has texture — a straight edge, or a
        # grating of parallel bars. The flow component ALONG the edge is
        # genuinely unobservable (the aperture problem), but the component
        # ACROSS it is perfectly well determined, and discarding it would
        # throw away most of what a drone actually sees: horizons, building
        # edges, corridor walls and doorframes are all rank-1 structure.
        #
        # Normal flow is the projection onto the gradient direction:
        #
        #     u_n = -It * Ix / (Ix^2 + Iy^2)
        #     v_n = -It * Iy / (Ix^2 + Iy^2)
        #
        # summed over the window. It underestimates true speed for motion
        # oblique to the gradient, which is the honest cost of the ambiguity.
        gradient_energy = trace
        safe_energy = np.where(gradient_energy > 1e-12, gradient_energy, 1.0)
        u_normal = -sxt / safe_energy
        v_normal = -syt / safe_energy

        rank_one = (~full_rank) & (gradient_energy > self._min_eigenvalue)

        flow = np.zeros((*self._shape, 2), dtype=np.float32)
        flow[:, :, 0] = np.where(full_rank, u_full, np.where(rank_one, u_normal, 0.0))
        flow[:, :, 1] = np.where(full_rank, v_full, np.where(rank_one, v_normal, 0.0))

        return flow

    def coverage(self, flow_field: FlowField) -> float:
        """Fraction of locations that produced a usable estimate.

        Worth surfacing in the demo: on an untextured surface this collapses
        toward zero, which is the honest reason the drone cannot see rather
        than a mysterious failure to avoid a wall.
        """
        magnitude = flow_field.magnitude
        return float(np.count_nonzero(magnitude) / magnitude.size)

    @classmethod
    def from_config(cls, cfg) -> "LucasKanadeFlow":
        return cls(
            shape=(cfg.retina.rows, cfg.retina.cols),
            window=cfg.optical_flow.window,
            min_eigenvalue=cfg.optical_flow.min_eigenvalue,
        )


class EMDFlowAdapter:
    """Wraps the Reichardt correlators to produce the same FlowField contract.

    The EMD naturally outputs one value per adjacent PAIR, so its arrays are
    one narrower than the retina along the axis they measure. This pads them
    back to full size so both front-ends are interchangeable downstream.

    Padding rather than interpolating: the edge column genuinely has no
    correlator, and inventing a value there would be a fabrication in exactly
    the place where the field of view ends.
    """

    __slots__ = ("_emd", "_shape")

    def __init__(self, shape=(20, 40), tau: float = 0.6) -> None:
        from emd import ReichardtArray

        self._shape = tuple(shape)
        self._emd = ReichardtArray(shape, tau=tau)

    def reset(self) -> None:
        self._emd.reset()

    def compute(self, retina: np.ndarray, timestamp: float = 0.0) -> FlowField:
        horizontal, vertical = self._emd.update(retina)

        flow = np.zeros((*self._shape, 2), dtype=np.float32)
        flow[:, :-1, 0] = horizontal
        flow[:-1, :, 1] = vertical

        valid = bool(np.any(horizontal) or np.any(vertical))
        return FlowField(timestamp=timestamp, flow=flow, valid=valid)

    @classmethod
    def from_config(cls, cfg) -> "EMDFlowAdapter":
        return cls(shape=(cfg.retina.rows, cfg.retina.cols), tau=cfg.emd.tau)


def make_flow_frontend(cfg):
    """Build whichever front-end the config selects.

    Args:
        cfg: loaded configuration.

    Returns:
        LucasKanadeFlow or EMDFlowAdapter — both satisfy the same interface,
        `compute(retina, timestamp) -> FlowField`.
    """
    method = cfg.optical_flow.method
    if method == "lucas_kanade":
        return LucasKanadeFlow.from_config(cfg)
    if method == "emd":
        return EMDFlowAdapter.from_config(cfg)
    raise ValueError(
        f"unknown optical_flow.method '{method}' — expected "
        "'lucas_kanade' or 'emd'"
    )
