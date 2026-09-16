"""Height zones — per-region altitude targets along a route.

Separate from geofence.py despite sharing polygon geometry, because the two
answer different questions and fail differently:

  geofence  — may the drone be here at all? A hard refusal.
  height    — how high should it fly here? A target, and a local ceiling.

Folding them together would mean a bug in altitude planning could weaken a
no-fly refusal. Keeping them apart means the safety-critical module stays
small enough to audit.

A zone's max_altitude is a local limit, always subordinate to the global cap
enforced in the output bottleneck. A zone can lower the ceiling; it can
never raise it.
"""

from typing import List, Optional

import numpy as np

from geofence import Polygon


class HeightZone:
    """A region with its own altitude target and local ceiling."""

    __slots__ = ("name", "polygon", "target_altitude", "max_altitude")

    def __init__(
        self,
        name: str,
        vertices: List[List[float]],
        target_altitude: float,
        max_altitude: float = None,
    ) -> None:
        if target_altitude <= 0.0:
            raise ValueError(
                f"zone '{name}': target altitude must be positive, "
                f"got {target_altitude}"
            )

        resolved_max = target_altitude if max_altitude is None else max_altitude
        if resolved_max < target_altitude:
            raise ValueError(
                f"zone '{name}': max altitude {resolved_max} is below the "
                f"target {target_altitude}"
            )

        self.name = name
        self.polygon = Polygon(name, vertices)
        self.target_altitude = target_altitude
        self.max_altitude = resolved_max

    def contains(self, position: np.ndarray) -> bool:
        return self.polygon.contains(np.asarray(position, dtype=np.float64)[:2])


class HeightZoneManager:
    """Resolves which altitude applies at a given position.

    Overlapping zones resolve to the MOST RESTRICTIVE: the lowest ceiling
    wins. Two zones overlapping usually means two separate reasons to stay
    low, and honouring only one of them would violate the other.
    """

    __slots__ = ("_zones", "_default_altitude", "_global_ceiling")

    def __init__(
        self,
        zones: List[HeightZone] = None,
        default_altitude: float = 1.5,
        global_ceiling: float = 10.0,
    ) -> None:
        self._zones = list(zones) if zones else []
        self._default_altitude = default_altitude
        self._global_ceiling = global_ceiling

    def target_for(self, position: np.ndarray) -> float:
        """The altitude the drone should hold at this position.

        Returns the default outside every zone; inside one or more, the
        lowest target among them.
        """
        p = np.asarray(position, dtype=np.float64)[:2]

        applicable = [z for z in self._zones if z.contains(p)]
        if not applicable:
            return min(self._default_altitude, self._global_ceiling)

        lowest = min(z.target_altitude for z in applicable)
        return min(lowest, self._global_ceiling)

    def ceiling_for(self, position: np.ndarray) -> float:
        """The maximum legal altitude at this position.

        Never above the global cap: a zone may lower the ceiling, never
        raise it. That ordering is what keeps the bottleneck's cap absolute.
        """
        p = np.asarray(position, dtype=np.float64)[:2]

        applicable = [z for z in self._zones if z.contains(p)]
        if not applicable:
            return self._global_ceiling

        lowest = min(z.max_altitude for z in applicable)
        return min(lowest, self._global_ceiling)

    def active_zone(self, position: np.ndarray) -> Optional[HeightZone]:
        """The most restrictive zone at this position, or None.

        For telemetry and the demo panel: naming the zone makes the drone's
        altitude change legible rather than mysterious.
        """
        p = np.asarray(position, dtype=np.float64)[:2]
        applicable = [z for z in self._zones if z.contains(p)]
        if not applicable:
            return None
        return min(applicable, key=lambda z: z.max_altitude)

    def violates(self, position: np.ndarray, altitude: float) -> bool:
        """Whether the drone is above the ceiling applying here."""
        return altitude > self.ceiling_for(position)

    @property
    def zones(self) -> List[HeightZone]:
        return list(self._zones)

    @classmethod
    def from_config(cls, cfg) -> "HeightZoneManager":
        zones = []
        entries = cfg.get("height_zones", []) or []
        for entry in entries:
            zones.append(
                HeightZone(
                    name=entry["name"],
                    vertices=entry["polygon"],
                    target_altitude=entry["target_altitude"],
                    max_altitude=entry.get("max_altitude"),
                )
            )
        return cls(
            zones=zones,
            default_altitude=cfg.flight.target_altitude,
            global_ceiling=cfg.altitude_cap.ceiling_m,
        )
