"""Battery model and return-to-home feasibility.

THE QUESTION THAT MATTERS IS NOT "HOW MUCH CHARGE IS LEFT".

It is "can the drone still get home from where it is now". Those differ
enormously: 20% charge is ample at 10 metres out and fatal at 500. A fixed
percentage threshold is the wrong abstraction, and a drone using one will
either turn back far too early on short flights or run out on long ones.

So the reserve is computed live, against the actual distance home from the
path integrator, plus a safety factor. When the margin closes, the drone
turns back. When it closes entirely, it pings its position as an emergency
because it is no longer able to reach home at all — which is the client's
stated requirement, and the honest thing for an aircraft to do.
"""

import numpy as np

from contracts import BatteryState


class BatteryModel:
    """Charge tracking and the derived go/no-go decision."""

    __slots__ = (
        "_capacity_wh",
        "_nominal_voltage",
        "_idle_watts",
        "_thrust_watt_per_unit",
        "_reserve_fraction",
        "_emergency_pct",
        "_cruise_speed",
        "_remaining_wh",
        "_elapsed",
        "_emergency_latched",
    )

    def __init__(
        self,
        capacity_wh: float = 60.0,
        nominal_voltage: float = 15.2,
        idle_watts: float = 15.0,
        thrust_watt_per_unit: float = 0.9,
        reserve_fraction: float = 0.15,
        emergency_pct: float = 10.0,
        cruise_speed_ms: float = 3.0,
        initial_charge_pct: float = 100.0,
    ) -> None:
        if capacity_wh <= 0.0:
            raise ValueError(f"capacity must be positive, got {capacity_wh}")
        if not 0.0 <= initial_charge_pct <= 100.0:
            raise ValueError(f"initial charge out of range: {initial_charge_pct}")

        self._capacity_wh = capacity_wh
        self._nominal_voltage = nominal_voltage
        self._idle_watts = idle_watts
        self._thrust_watt_per_unit = thrust_watt_per_unit
        self._reserve_fraction = reserve_fraction
        self._emergency_pct = emergency_pct
        self._cruise_speed = cruise_speed_ms

        self._remaining_wh = capacity_wh * (initial_charge_pct / 100.0)
        self._elapsed = 0.0
        # Emergency latches: a drone that flickers in and out of emergency
        # as charge hovers at the threshold would oscillate between mission
        # and return, which is worse than committing to one.
        self._emergency_latched = False

    def update(
        self, total_thrust: float, distance_to_home: float, dt: float
    ) -> BatteryState:
        """Draw charge for one timestep and evaluate feasibility.

        Args:
            total_thrust: summed motor command, the main power term.
            distance_to_home: metres, from the path integrator. Works
                without GPS, which is the point.
            dt: timestep in seconds.

        Returns:
            BatteryState including the live return-home verdict.
        """
        watts = self._idle_watts + self._thrust_watt_per_unit * max(
            0.0, total_thrust
        )
        self._remaining_wh -= watts * (dt / 3600.0)
        self._remaining_wh = max(0.0, self._remaining_wh)
        self._elapsed += dt

        charge_pct = 100.0 * self._remaining_wh / self._capacity_wh

        needed = self.energy_to_return(distance_to_home, watts)
        reserve = needed * (1.0 + self._reserve_fraction)
        can_return = self._remaining_wh >= reserve

        if not can_return or charge_pct <= self._emergency_pct:
            self._emergency_latched = True

        return BatteryState(
            charge_pct=float(charge_pct),
            voltage=self._voltage_for(charge_pct),
            can_return_home=bool(can_return and not self._emergency_latched),
            emergency=self._emergency_latched,
        )

    def energy_to_return(self, distance_m: float, current_watts: float) -> float:
        """Watt-hours needed to fly `distance_m` home at cruise speed."""
        if self._cruise_speed <= 0.0:
            return float("inf")
        seconds = distance_m / self._cruise_speed
        return current_watts * (seconds / 3600.0)

    def _voltage_for(self, charge_pct: float) -> float:
        """Approximate terminal voltage.

        Lithium packs hold voltage through most of their range then fall off
        steeply below ~20%, so a linear model would badly misreport exactly
        the region where the reading matters most.
        """
        fraction = charge_pct / 100.0
        if fraction > 0.2:
            sag = 0.12 * (1.0 - fraction)
        else:
            sag = 0.12 * 0.8 + 0.9 * (0.2 - fraction)
        return float(self._nominal_voltage * (1.0 - sag))

    def max_range(self, current_watts: float) -> float:
        """Metres still flyable before the reserve is consumed."""
        if current_watts <= 0.0:
            return float("inf")
        hours = self._remaining_wh / current_watts
        usable = hours * (1.0 - self._reserve_fraction)
        return float(usable * 3600.0 * self._cruise_speed)

    @property
    def charge_pct(self) -> float:
        return float(100.0 * self._remaining_wh / self._capacity_wh)

    @property
    def remaining_wh(self) -> float:
        return float(self._remaining_wh)

    @property
    def consumed_wh(self) -> float:
        return float(self._capacity_wh - self._remaining_wh)

    def reset(self, charge_pct: float = 100.0) -> None:
        self._remaining_wh = self._capacity_wh * (charge_pct / 100.0)
        self._elapsed = 0.0
        self._emergency_latched = False

    def force_level(self, charge_pct: float) -> None:
        """Set charge directly. For failure injection during a demo."""
        if not 0.0 <= charge_pct <= 100.0:
            raise ValueError(f"charge out of range: {charge_pct}")
        self._remaining_wh = self._capacity_wh * (charge_pct / 100.0)

    @classmethod
    def from_config(cls, cfg) -> "BatteryModel":
        return cls(
            capacity_wh=cfg.battery.capacity_wh,
            nominal_voltage=cfg.battery.nominal_voltage,
            idle_watts=cfg.battery.idle_watts,
            thrust_watt_per_unit=cfg.battery.thrust_watt_per_unit,
            reserve_fraction=cfg.battery.reserve_fraction,
            emergency_pct=cfg.battery.emergency_pct,
            cruise_speed_ms=cfg.battery.cruise_speed_ms,
        )
