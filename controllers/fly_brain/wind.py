"""Wind estimation and gust detection.

There is no wind sensor. The estimate comes from the gap between the motion
the controller commanded and the motion that actually happened: if the drone
was told to hold station and drifted east at 2 m/s, the wind is blowing east
at roughly 2 m/s.

This is exactly how an insect does it — no anemometer, just the discrepancy
between intended and achieved movement, which for a fly comes from optic
flow and haltere feedback.

WHY A WINDOW RATHER THAN AN INSTANT: a single-sample estimate is dominated
by control transients. Commanding a turn produces a large instantaneous gap
that is not wind at all. Averaging over a window separates the persistent
component (wind) from the transient one (manoeuvring).
"""

from collections import deque

import numpy as np

from contracts import WindEstimate


class WindEstimator:
    """Estimate wind from commanded-versus-achieved velocity."""

    __slots__ = (
        "_window",
        "_gust_threshold",
        "_recovery_timeout",
        "_samples",
        "_gust_active",
        "_gust_elapsed",
        "_last_estimate",
    )

    def __init__(
        self,
        window: int = 25,
        gust_threshold: float = 2.0,
        recovery_timeout_s: float = 3.0,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")

        self._window = window
        self._gust_threshold = gust_threshold
        self._recovery_timeout = recovery_timeout_s
        self._samples = deque(maxlen=window)
        self._gust_active = False
        self._gust_elapsed = 0.0
        self._last_estimate = np.zeros(2, dtype=np.float64)

    def update(
        self,
        commanded_velocity: np.ndarray,
        actual_velocity: np.ndarray,
        dt: float,
    ) -> WindEstimate:
        """Add a sample and return the current estimate.

        Args:
            commanded_velocity: (2,) m/s the controller intended.
            actual_velocity: (2,) m/s actually achieved, from GPS deltas.
            dt: timestep in seconds.

        Returns:
            WindEstimate with direction (rad), speed (m/s) and a gust flag.
        """
        commanded = np.asarray(commanded_velocity, dtype=np.float64)[:2]
        actual = np.asarray(actual_velocity, dtype=np.float64)[:2]

        # The residual IS the wind, to first order.
        residual = actual - commanded
        self._samples.append(residual)

        stacked = np.array(self._samples, dtype=np.float64)
        mean = stacked.mean(axis=0)

        speed = float(np.linalg.norm(mean))
        direction = float(np.arctan2(mean[1], mean[0]))

        gust = self._detect_gust(mean, dt)
        self._last_estimate = mean

        return WindEstimate(
            direction=direction,
            speed=speed,
            gust_detected=gust,
        )

    def _detect_gust(self, mean: np.ndarray, dt: float) -> bool:
        """A gust is a sudden CHANGE, not a high steady value.

        Comparing against the previous estimate rather than against zero
        means steady wind does not permanently assert a gust — the drone is
        already compensating for that and does not need to keep reacting.
        """
        change = float(np.linalg.norm(mean - self._last_estimate))

        if change > self._gust_threshold:
            self._gust_active = True
            self._gust_elapsed = 0.0
            return True

        if self._gust_active:
            self._gust_elapsed += dt
            if self._gust_elapsed >= self._recovery_timeout:
                self._gust_active = False
            return self._gust_active

        return False

    def compensation(self) -> np.ndarray:
        """Velocity correction to add to a command to cancel the wind.

        The negation of the estimate: if wind pushes east, command west.
        """
        return -self._last_estimate.copy()

    @property
    def estimate_vector(self) -> np.ndarray:
        return self._last_estimate.copy()

    @property
    def is_recovering(self) -> bool:
        """Whether a gust is still within its recovery window."""
        return self._gust_active

    def reset(self) -> None:
        self._samples.clear()
        self._gust_active = False
        self._gust_elapsed = 0.0
        self._last_estimate = np.zeros(2, dtype=np.float64)

    @classmethod
    def from_config(cls, cfg) -> "WindEstimator":
        return cls(
            window=cfg.wind.estimation_window,
            gust_threshold=cfg.wind.gust_threshold,
            recovery_timeout_s=cfg.wind.recovery_timeout_s,
        )
