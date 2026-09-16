"""Vibration monitoring — detecting a cracked propeller before it fails.

A balanced rotor produces vibration at its rotation frequency and harmonics,
at low amplitude. A chipped or cracked blade is mass-imbalanced, and that
imbalance drives a large peak at exactly the rotor frequency — growing as
the crack propagates.

So the detector is not "is there vibration" (there always is) but "is there
ANOMALOUS energy near the rotor frequency". That distinction is why this
module does a spectral analysis rather than thresholding raw accelerometer
magnitude: a hard manoeuvre produces large broadband acceleration and would
trip a magnitude threshold constantly, while a cracked blade produces a
modest but spectrally concentrated signal.

Detection is deliberately conservative. A false positive aborts a mission;
a false negative loses the aircraft. Sustained confirmation over several
windows is required before the anomaly is declared.
"""

from collections import deque

import numpy as np

from contracts import VibrationState


class VibrationMonitor:
    """FFT-based propeller health monitoring from IMU accelerometer data."""

    __slots__ = (
        "_window_size",
        "_anomaly_amplitude",
        "_rotor_freq",
        "_freq_tolerance",
        "_reduced_speed_scale",
        "_sample_rate",
        "_samples",
        "_anomaly_count",
        "_confirm_windows",
        "_latched",
        "_forced",
    )

    def __init__(
        self,
        window_size: int = 128,
        anomaly_amplitude: float = 0.35,
        rotor_freq_hz: float = 95.0,
        freq_tolerance_hz: float = 12.0,
        reduced_speed_scale: float = 0.5,
        sample_rate_hz: float = 125.0,
        confirm_windows: int = 3,
    ) -> None:
        if window_size < 16:
            raise ValueError(f"window too small for FFT: {window_size}")

        self._window_size = window_size
        self._anomaly_amplitude = anomaly_amplitude
        self._rotor_freq = rotor_freq_hz
        self._freq_tolerance = freq_tolerance_hz
        self._reduced_speed_scale = reduced_speed_scale
        self._sample_rate = sample_rate_hz
        self._samples = deque(maxlen=window_size)
        self._anomaly_count = 0
        self._confirm_windows = confirm_windows
        self._latched = False
        self._forced = False

    def update(self, acceleration: np.ndarray) -> VibrationState:
        """Add an accelerometer sample and analyse when the window fills.

        Args:
            acceleration: (3,) m/s^2 from the IMU.

        Returns:
            VibrationState with the dominant frequency, its amplitude, and
            whether an anomaly is confirmed.
        """
        accel = np.asarray(acceleration, dtype=np.float64)
        # Vertical axis carries propeller imbalance most strongly, since the
        # rotor disc is horizontal.
        self._samples.append(float(accel[2]) if accel.size >= 3 else float(accel[0]))

        if len(self._samples) < self._window_size:
            return VibrationState(
                dominant_freq=0.0,
                amplitude=0.0,
                anomaly_detected=self._latched or self._forced,
            )

        signal = np.array(self._samples, dtype=np.float64)

        # Remove the DC component: gravity dominates the vertical axis and
        # would otherwise swamp every real vibration peak.
        signal = signal - signal.mean()

        # Hann window to stop spectral leakage smearing a sharp rotor peak
        # across neighbouring bins.
        windowed = signal * np.hanning(len(signal))

        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self._sample_rate)

        # Normalise so amplitude is comparable across window lengths.
        spectrum = spectrum * (2.0 / len(windowed))

        if spectrum.size > 1:
            peak_index = int(np.argmax(spectrum[1:]) + 1)
        else:
            peak_index = 0

        dominant_freq = float(freqs[peak_index])
        amplitude = float(spectrum[peak_index])

        near_rotor = (
            abs(dominant_freq - self._rotor_freq) <= self._freq_tolerance
        )
        exceeds = amplitude > self._anomaly_amplitude

        if near_rotor and exceeds:
            self._anomaly_count += 1
        else:
            self._anomaly_count = 0

        if self._anomaly_count >= self._confirm_windows:
            # Latched: a propeller does not heal, and a detector that
            # un-declares would send the drone back out on a cracked blade.
            self._latched = True

        return VibrationState(
            dominant_freq=dominant_freq,
            amplitude=amplitude,
            anomaly_detected=self._latched or self._forced,
        )

    def speed_scale(self, state: VibrationState) -> float:
        """Speed multiplier to apply when an anomaly is present.

        Reducing speed lowers rotor RPM, which reduces the stress driving
        crack propagation and buys time to reach base.
        """
        if state.anomaly_detected:
            return self._reduced_speed_scale
        return 1.0

    def should_return_to_base(self, state: VibrationState) -> bool:
        return state.anomaly_detected

    def force_anomaly(self, active: bool = True) -> None:
        """Assert an anomaly directly. For failure injection in the demo."""
        self._forced = active

    def reset(self) -> None:
        self._samples.clear()
        self._anomaly_count = 0
        self._latched = False
        self._forced = False

    @property
    def is_latched(self) -> bool:
        return self._latched

    @classmethod
    def from_config(cls, cfg) -> "VibrationMonitor":
        return cls(
            window_size=cfg.vibration.sample_window,
            anomaly_amplitude=cfg.vibration.anomaly_amplitude,
            rotor_freq_hz=cfg.vibration.rotor_freq_hz,
            freq_tolerance_hz=cfg.vibration.freq_tolerance_hz,
            reduced_speed_scale=cfg.vibration.reduced_speed_scale,
            sample_rate_hz=cfg.simulation.control_rate_hz,
        )
