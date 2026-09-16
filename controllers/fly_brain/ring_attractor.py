"""Central complex — ellipsoid body, protocerebral bridge, fan-shaped body.

A ring attractor holding heading, plus path integration for dead reckoning.

Structure:
  - N units arranged in a ring
  - Local excitation: each unit excites its neighbours (gaussian profile)
  - Global inhibition: all units inhibit all others
  - Result: a single stable bump of activity, whose position is the heading

The bump drifts. That is biologically accurate and acceptable over short
flights; the fly has the same problem and solves it with visual landmarks.

WHY THIS MATTERS BEYOND HEADING: path integration on top of the ring gives
GPS-denied return-to-home. The estimate is built from the drone's own motion,
so it cannot be spoofed or jammed — an attacker who controls the GPS signal
controls nothing here. Insects navigate this way; it is not a fallback bolted
on, it is the primary mechanism with GPS as the optional correction.
"""

import numpy as np

from contracts import HeadingState, PathState


class RingAttractor:
    """A ring of units supporting one stable bump of activity."""

    __slots__ = (
        "_n_units",
        "_excitation_strength",
        "_excitation_width",
        "_inhibition_strength",
        "_leak",
        "_units_per_radian",
        "_min_confidence",
        "_activity",
        "_excitation_kernel",
        "_unit_angles",
    )

    def __init__(
        self,
        n_units: int = 32,
        excitation_strength: float = 1.0,
        excitation_width: float = 2.0,
        inhibition_strength: float = 0.5,
        leak: float = 0.1,
        min_confidence: float = 0.3,
        initial_heading: float = 0.0,
    ) -> None:
        if n_units < 8:
            raise ValueError(f"n_units must be at least 8, got {n_units}")

        self._n_units = n_units
        self._excitation_strength = excitation_strength
        self._excitation_width = excitation_width
        self._inhibition_strength = inhibition_strength
        self._leak = leak
        self._units_per_radian = n_units / (2.0 * np.pi)
        self._min_confidence = min_confidence

        # Circular gaussian excitation kernel, applied by FFT convolution so
        # the wrap-around is exact rather than approximated at the seam.
        offsets = np.arange(n_units)
        offsets = np.minimum(offsets, n_units - offsets)
        self._excitation_kernel = np.exp(
            -0.5 * (offsets / excitation_width) ** 2
        )
        self._excitation_kernel /= self._excitation_kernel.sum()

        self._unit_angles = np.arange(n_units) * (2.0 * np.pi / n_units)

        self._activity = self._make_bump(initial_heading)

    def _make_bump(self, heading_rad: float) -> np.ndarray:
        """A clean gaussian bump centred on the given heading."""
        centre = heading_rad * self._units_per_radian
        offsets = np.arange(self._n_units) - centre
        # Wrap to the shortest way round the ring.
        offsets = (offsets + self._n_units / 2.0) % self._n_units - self._n_units / 2.0
        bump = np.exp(-0.5 * (offsets / self._excitation_width) ** 2)
        return bump / bump.sum()

    def step(self, yaw_rate: float, dt: float) -> HeadingState:
        """Shift the bump by the measured yaw rate, then re-sharpen it.

        Args:
            yaw_rate: rad/s from the gyro.
            dt: timestep in seconds.

        Returns:
            HeadingState with the bump, its peak, the decoded heading and a
            confidence derived from how concentrated the bump is.
        """
        shift_units = yaw_rate * dt * self._units_per_radian
        self._activity = self._shift(self._activity, shift_units)

        # One step of excitation / inhibition to re-sharpen.
        excited = np.real(
            np.fft.ifft(
                np.fft.fft(self._activity) * np.fft.fft(self._excitation_kernel)
            )
        )
        excited *= self._excitation_strength

        inhibition = self._inhibition_strength * self._activity.mean()
        updated = excited - inhibition
        updated -= self._leak * self._activity

        # Rectify: firing rates cannot be negative.
        np.maximum(updated, 0.0, out=updated)

        total = updated.sum()
        if total > 1e-12:
            updated /= total
        else:
            # Bump collapsed. Rebuild at the last known heading rather than
            # returning an all-zero state nothing downstream can use.
            updated = self._make_bump(self.heading)

        self._activity = updated
        return self.state()

    def _shift(self, activity: np.ndarray, shift_units: float) -> np.ndarray:
        """Rotate the bump by a fractional number of units.

        Linear interpolation between the two neighbouring integer shifts:
        rounding to whole units would quantise heading to 11.25 degree steps
        and lose slow rotations entirely.
        """
        whole = int(np.floor(shift_units))
        frac = shift_units - whole

        rolled_low = np.roll(activity, whole)
        rolled_high = np.roll(activity, whole + 1)
        return (1.0 - frac) * rolled_low + frac * rolled_high

    def state(self) -> HeadingState:
        """Decode the current bump into a heading estimate."""
        peak_index = int(np.argmax(self._activity))

        # Population vector decoding, which is robust to a slightly
        # asymmetric bump in a way that taking the peak alone is not.
        x = float(np.sum(self._activity * np.cos(self._unit_angles)))
        y = float(np.sum(self._activity * np.sin(self._unit_angles)))
        heading = float(np.arctan2(y, x))

        # Confidence is the resultant length: 1 for a perfectly concentrated
        # bump, 0 for uniform activity. A split or fading bump shows here.
        confidence = float(np.hypot(x, y))

        return HeadingState(
            bump=self._activity.copy(),
            peak_index=peak_index,
            heading_rad=heading,
            confidence=confidence,
        )

    @property
    def heading(self) -> float:
        """Current bump position, in radians, wrapped to [-pi, pi]."""
        x = float(np.sum(self._activity * np.cos(self._unit_angles)))
        y = float(np.sum(self._activity * np.sin(self._unit_angles)))
        return float(np.arctan2(y, x))

    @property
    def confidence(self) -> float:
        x = float(np.sum(self._activity * np.cos(self._unit_angles)))
        y = float(np.sum(self._activity * np.sin(self._unit_angles)))
        return float(np.hypot(x, y))

    def is_trusted(self) -> bool:
        """Whether the heading estimate should be acted on."""
        return self.confidence >= self._min_confidence

    def reset(self, heading_rad: float = 0.0) -> None:
        """Re-seed the bump, e.g. after a visual landmark fix."""
        self._activity = self._make_bump(heading_rad)

    @classmethod
    def from_config(cls, cfg, initial_heading: float = 0.0) -> "RingAttractor":
        return cls(
            n_units=cfg.ring_attractor.n_units,
            excitation_strength=cfg.ring_attractor.excitation_strength,
            excitation_width=cfg.ring_attractor.excitation_width,
            inhibition_strength=cfg.ring_attractor.inhibition_strength,
            leak=cfg.ring_attractor.leak,
            min_confidence=cfg.ring_attractor.min_confidence,
            initial_heading=initial_heading,
        )


def circular_difference(a: float, b: float) -> float:
    """Signed shortest angular difference a - b, wrapped to [-pi, pi].

    A naive subtraction makes a 359 degree error read as 359 rather than -1,
    and the drone turns the long way round.
    """
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


class PathIntegrator:
    """Dead reckoning: accumulate displacement to keep a vector home.

    GPS, when available AND trusted, nudges the estimate rather than
    replacing it. That blend is deliberate: a spoofed GPS signal can then
    only drag the estimate slowly, and the discrepancy is detectable, whereas
    a hard overwrite would hand an attacker the drone's position outright.
    """

    __slots__ = (
        "_position",
        "_home",
        "_gps_correction_rate",
        "_gps_available",
        "_accumulated_distance",
    )

    def __init__(
        self,
        home: np.ndarray = None,
        gps_correction_rate: float = 0.05,
    ) -> None:
        self._home = (
            np.zeros(2, dtype=np.float64) if home is None
            else np.asarray(home, dtype=np.float64).copy()
        )
        self._position = self._home.copy()
        self._gps_correction_rate = gps_correction_rate
        self._gps_available = True
        self._accumulated_distance = 0.0

    def update(
        self,
        heading_rad: float,
        speed: float,
        dt: float,
        gps_position: np.ndarray = None,
    ) -> PathState:
        """Integrate one step of motion.

        Args:
            heading_rad: current heading, from the ring attractor.
            speed: forward speed in m/s.
            dt: timestep in seconds.
            gps_position: (2,) world position, or None when GPS is
                unavailable, jammed, or distrusted.

        Returns:
            PathState with the estimate and the vector home.
        """
        step = speed * dt
        self._position = self._position + np.array(
            [step * np.cos(heading_rad), step * np.sin(heading_rad)],
            dtype=np.float64,
        )
        self._accumulated_distance += abs(step)

        self._gps_available = gps_position is not None
        if self._gps_available:
            gps = np.asarray(gps_position, dtype=np.float64)
            # Partial correction, not replacement — see the class docstring.
            self._position += self._gps_correction_rate * (gps - self._position)

        home_vector = self._home - self._position
        return PathState(
            position_estimate=self._position.copy(),
            home_vector=home_vector,
            distance_to_home=float(np.linalg.norm(home_vector)),
            gps_available=self._gps_available,
        )

    def state(self) -> PathState:
        home_vector = self._home - self._position
        return PathState(
            position_estimate=self._position.copy(),
            home_vector=home_vector,
            distance_to_home=float(np.linalg.norm(home_vector)),
            gps_available=self._gps_available,
        )

    def heading_home(self) -> float:
        """Bearing from the current estimate back to home, in radians."""
        home_vector = self._home - self._position
        return float(np.arctan2(home_vector[1], home_vector[0]))

    def set_home(self, home: np.ndarray) -> None:
        self._home = np.asarray(home, dtype=np.float64).copy()

    @property
    def accumulated_distance(self) -> float:
        """Total path length flown — used by the battery model."""
        return self._accumulated_distance

    @classmethod
    def from_config(cls, cfg, home: np.ndarray = None) -> "PathIntegrator":
        return cls(
            home=home,
            gps_correction_rate=cfg.path_integration.gps_correction_rate,
        )
