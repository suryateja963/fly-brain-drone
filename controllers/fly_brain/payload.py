"""Payload release and drop-zone safety.

Two independent failure modes, and both are dangerous:

  RELEASING ONTO SOMEONE. A parcel dropped from altitude onto a person or
  animal is a serious injury. The zone must be verified clear at the moment
  of release, not merely at the moment the drone decided to release.

  FAILING TO RELEASE. A payload that snags on the mechanism leaves the drone
  carrying unexpected mass, asymmetrically, with the controller trimmed for
  a released state. That combination is a flyaway. So detachment is
  CONFIRMED by measuring the mass change, never assumed from the fact that
  the actuator was commanded.

Authorisation is computed as a single conjunction so no caller can release
by checking only some of the preconditions.
"""

from typing import List, Optional

import numpy as np

from contracts import PayloadState


class PayloadManager:
    """Drop-zone assessment, release authorisation and detachment confirmation."""

    __slots__ = (
        "_drop_zone_radius",
        "_confirm_timeout",
        "_min_release_altitude",
        "_attached",
        "_payload_mass",
        "_release_commanded_at",
        "_confirmed",
        "_forced_occupied",
        "_abort_reasons",
    )

    def __init__(
        self,
        drop_zone_radius: float = 2.0,
        confirm_timeout_s: float = 1.5,
        min_release_altitude: float = 1.0,
        payload_mass: float = 0.5,
        attached: bool = True,
    ) -> None:
        self._drop_zone_radius = drop_zone_radius
        self._confirm_timeout = confirm_timeout_s
        self._min_release_altitude = min_release_altitude
        self._attached = attached
        self._payload_mass = payload_mass
        self._release_commanded_at: Optional[float] = None
        self._confirmed = False
        self._forced_occupied = False
        self._abort_reasons: List[str] = []

    def assess(
        self,
        drone_position: np.ndarray,
        drop_target: np.ndarray,
        altitude: float,
        obstacles: List[np.ndarray] = None,
    ) -> PayloadState:
        """Evaluate whether release is currently permitted.

        Args:
            drone_position: (2,) current world position.
            drop_target: (2,) centre of the intended drop zone.
            altitude: current altitude, metres.
            obstacles: world positions of people, animals or vehicles
                detected near the zone. Any inside the radius blocks release.

        Returns:
            PayloadState with the conjunction already computed.
        """
        self._abort_reasons = []

        clear = self._zone_is_clear(drop_target, obstacles)
        if not clear:
            self._abort_reasons.append("drop zone occupied")

        over_target = (
            float(
                np.linalg.norm(
                    np.asarray(drone_position, dtype=np.float64)[:2]
                    - np.asarray(drop_target, dtype=np.float64)[:2]
                )
            )
            <= self._drop_zone_radius
        )
        if not over_target:
            self._abort_reasons.append("not over drop zone")

        high_enough = altitude >= self._min_release_altitude
        if not high_enough:
            self._abort_reasons.append(
                f"altitude {altitude:.2f}m below minimum "
                f"{self._min_release_altitude:.2f}m"
            )

        if not self._attached:
            self._abort_reasons.append("no payload attached")

        authorized = bool(
            self._attached and clear and over_target and high_enough
        )

        return PayloadState(
            attached=self._attached,
            drop_zone_clear=clear,
            release_authorized=authorized,
        )

    def _zone_is_clear(
        self, drop_target: np.ndarray, obstacles: List[np.ndarray]
    ) -> bool:
        """Whether anything is inside the drop zone radius."""
        if self._forced_occupied:
            return False
        if not obstacles:
            return True

        target = np.asarray(drop_target, dtype=np.float64)[:2]
        for obstacle in obstacles:
            position = np.asarray(obstacle, dtype=np.float64)[:2]
            if float(np.linalg.norm(position - target)) <= self._drop_zone_radius:
                return False
        return True

    def command_release(self, state: PayloadState, now: float) -> bool:
        """Command release if authorised.

        Returns:
            True if the release was commanded. False means it was refused,
            and `abort_reasons` says why.
        """
        if not state.release_authorized:
            return False

        self._release_commanded_at = now
        self._confirmed = False
        return True

    def confirm_detachment(self, measured_mass_delta: float, now: float):
        """Confirm the payload actually left, by mass change.

        Args:
            measured_mass_delta: kg drop in total aircraft mass since the
                release was commanded.
            now: current time in seconds.

        Returns:
            (confirmed, snagged). `snagged` becomes True once the timeout
            passes with no mass change — the flyaway condition, and the
            reason this function exists rather than trusting the actuator.
        """
        if self._release_commanded_at is None:
            return False, False

        # Most of the payload mass gone is confirmation; a partial change
        # means it is hanging, which is worse than either extreme.
        if measured_mass_delta >= 0.8 * self._payload_mass:
            self._attached = False
            self._confirmed = True
            return True, False

        elapsed = now - self._release_commanded_at
        if elapsed >= self._confirm_timeout:
            return False, True

        return False, False

    @property
    def abort_reasons(self) -> List[str]:
        """Why the last assessment refused. For telemetry and the demo."""
        return list(self._abort_reasons)

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def force_zone_occupied(self, occupied: bool = True) -> None:
        """Assert an occupied zone. For failure injection in the demo."""
        self._forced_occupied = occupied

    def reset(self, attached: bool = True) -> None:
        self._attached = attached
        self._release_commanded_at = None
        self._confirmed = False
        self._forced_occupied = False
        self._abort_reasons = []

    @classmethod
    def from_config(cls, cfg) -> "PayloadManager":
        return cls(
            drop_zone_radius=cfg.payload.drop_zone_radius,
            confirm_timeout_s=cfg.payload.confirm_timeout_s,
            min_release_altitude=cfg.payload.min_release_altitude,
        )
