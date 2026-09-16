"""Motor failure detection and three-rotor rebalancing.

A quadrotor that loses one rotor cannot hold yaw. This is geometry, not a
tuning problem: yaw comes from the torque imbalance between the two
counter-rotating diagonal pairs, and with one rotor gone that balance cannot
be restored at any throttle setting.

What it CAN do is give up yaw and keep everything else. The airframe spins
continuously about its vertical axis while the remaining three rotors
modulate to hold position and altitude. It is not elegant and the camera is
useless, but it descends under control instead of falling.

That trade is what `degraded_mode` records: yaw authority surrendered in
exchange for a survivable landing.
"""

import numpy as np

from contracts import MotorHealth

FRONT_LEFT = 0
FRONT_RIGHT = 1
REAR_LEFT = 2
REAR_RIGHT = 3

# Diagonal pairs share a rotation direction; losing one means its diagonal
# partner has no counter-torque.
_DIAGONAL_PARTNER = {
    FRONT_LEFT: REAR_RIGHT,
    REAR_RIGHT: FRONT_LEFT,
    FRONT_RIGHT: REAR_LEFT,
    REAR_LEFT: FRONT_RIGHT,
}


class MotorFailureHandler:
    """Detect a dead rotor and rebalance the survivors."""

    __slots__ = (
        "_detection_threshold",
        "_rebalance_gain",
        "_descent_rate",
        "_status",
        "_consecutive",
        "_confirm_samples",
        "_forced_failure",
    )

    def __init__(
        self,
        detection_threshold: float = 0.3,
        rebalance_gain: float = 1.4,
        controlled_descent_rate: float = 0.5,
        confirm_samples: int = 8,
    ) -> None:
        self._detection_threshold = detection_threshold
        self._rebalance_gain = rebalance_gain
        self._descent_rate = controlled_descent_rate
        self._status = np.ones(4, dtype=bool)
        self._consecutive = np.zeros(4, dtype=int)
        # A single bad sample is noise; a rotor is only declared dead after
        # sustained disagreement. Declaring it early would trigger a
        # degraded landing over a transient.
        self._confirm_samples = confirm_samples
        self._forced_failure = None

    def detect(
        self, commanded: np.ndarray, achieved: np.ndarray
    ) -> MotorHealth:
        """Compare commanded against achieved rotor speeds.

        Args:
            commanded: (4,) commanded motor velocities.
            achieved: (4,) measured velocities from the ESC feedback.

        Returns:
            MotorHealth with per-rotor status and the mode flags.
        """
        commanded = np.asarray(commanded, dtype=np.float64)
        achieved = np.asarray(achieved, dtype=np.float64)

        # Ratio of achieved to commanded; a dead rotor reads near zero.
        safe_commanded = np.where(np.abs(commanded) > 1e-6, commanded, 1.0)
        ratio = np.abs(achieved / safe_commanded)

        # Only judge rotors actually being asked to do something.
        active = np.abs(commanded) > 1e-6
        failing = active & (ratio < self._detection_threshold)

        self._consecutive = np.where(failing, self._consecutive + 1, 0)
        newly_dead = self._consecutive >= self._confirm_samples
        self._status = self._status & ~newly_dead

        if self._forced_failure is not None:
            self._status[self._forced_failure] = False

        healthy = int(np.count_nonzero(self._status))
        degraded = healthy < 4

        return MotorHealth(
            rotor_status=self._status.copy(),
            rebalanced=degraded and healthy >= 3,
            degraded_mode=degraded,
        )

    def rebalance(
        self, motors: np.ndarray, health: MotorHealth
    ) -> np.ndarray:
        """Redistribute thrust across the surviving rotors.

        Args:
            motors: (4,) commands from the output bottleneck.
            health: current rotor status.

        Returns:
            (4,) adjusted commands. A dead rotor is commanded to zero, and
            its diagonal partner is reduced to match — keeping that partner
            at full thrust would produce a violent roll, because the torque
            it generates has nothing opposing it.
        """
        adjusted = np.asarray(motors, dtype=np.float64).copy()
        status = health.rotor_status

        if np.all(status):
            return adjusted

        dead = np.flatnonzero(~status)
        if dead.size == 0:
            return adjusted

        reduced_partners = []
        for index in dead:
            adjusted[index] = 0.0
            partner = _DIAGONAL_PARTNER[int(index)]
            if status[partner]:
                # Match the partner down. Full asymmetric thrust flips the
                # airframe; a matched pair keeps it broadly level while it
                # spins.
                adjusted[partner] *= 0.5
                reduced_partners.append(partner)

        alive = np.flatnonzero(status)
        if alive.size == 0:
            return np.zeros(4, dtype=np.float64)

        # Redistribute the lost lift, but NOT onto the reduced partners:
        # they were deliberately brought down to match a dead rotor, and
        # feeding lift back into them undoes that in the same call. MEASURED:
        # without this exclusion the partner returned to 84 from a nominal
        # 70, worse than doing nothing.
        recipients = np.array(
            [i for i in alive if i not in reduced_partners], dtype=int
        )
        if recipients.size == 0:
            recipients = alive

        lost = float(np.sum(motors) - np.sum(adjusted))
        if lost > 0.0:
            adjusted[recipients] += (
                self._rebalance_gain * lost / recipients.size
            )

        return adjusted

    def descent_target(self, current_altitude: float, dt: float) -> float:
        """Altitude target for a controlled descent, one step down."""
        return max(0.0, current_altitude - self._descent_rate * dt)

    def can_hold_position(self, health: MotorHealth) -> bool:
        """Whether position holding is still possible.

        Three rotors can hold position while spinning. Two cannot do
        anything but fall.
        """
        return int(np.count_nonzero(health.rotor_status)) >= 3

    def force_failure(self, rotor_index: int) -> None:
        """Kill a rotor deliberately. For failure injection in the demo."""
        if not 0 <= rotor_index < 4:
            raise ValueError(f"rotor index out of range: {rotor_index}")
        self._forced_failure = rotor_index
        self._status[rotor_index] = False

    def reset(self) -> None:
        self._status = np.ones(4, dtype=bool)
        self._consecutive = np.zeros(4, dtype=int)
        self._forced_failure = None

    @classmethod
    def from_config(cls, cfg) -> "MotorFailureHandler":
        return cls(
            detection_threshold=cfg.motor_failure.detection_threshold,
            rebalance_gain=cfg.motor_failure.rebalance_gain,
            controlled_descent_rate=cfg.motor_failure.controlled_descent_rate,
        )
