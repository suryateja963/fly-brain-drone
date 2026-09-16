"""Geofencing — no-fly zones the drone refuses to enter.

Polygon regions in world coordinates: airports, military bases, stadiums.
The drone checks its own position against them and, when close, produces a
vector pointing back to legal airspace.

WHY A DIRECTION RATHER THAN A BOOLEAN: a bare "you are outside" flag leaves
the controller nothing to act on but stopping, and a drone that stops dead
at a boundary in wind will drift across it. Returning the escape vector lets
the controller steer out, which is what actually keeps it compliant.

The margin is standoff distance. The drone turns back before touching the
boundary, because a control system with latency that aims exactly at a line
will cross it.
"""

from typing import List, Optional, Tuple

import numpy as np

from contracts import GeofenceStatus


class Polygon:
    """A closed polygon in world coordinates, with point and distance queries."""

    __slots__ = ("name", "_vertices", "_edges")

    def __init__(self, name: str, vertices: List[List[float]]) -> None:
        if len(vertices) < 3:
            raise ValueError(
                f"polygon '{name}' needs at least 3 vertices, got {len(vertices)}"
            )

        self.name = name
        self._vertices = np.asarray(vertices, dtype=np.float64)
        if self._vertices.shape[1] != 2:
            raise ValueError(f"polygon '{name}' vertices must be 2-D points")

        # Precompute edges as (start, end) pairs; the polygon is fixed for
        # the flight, and the control loop runs at 125Hz.
        rolled = np.roll(self._vertices, -1, axis=0)
        self._edges = (self._vertices, rolled)

    def contains(self, point: np.ndarray) -> bool:
        """Ray casting: count edge crossings on a ray to +x.

        Odd crossings means inside. Chosen over winding number because it
        handles the concave polygons real no-fly zones tend to be, and
        needs no assumption about vertex ordering.
        """
        x, y = float(point[0]), float(point[1])
        starts, ends = self._edges

        # An edge crosses the horizontal ray if its endpoints straddle y.
        straddles = (starts[:, 1] > y) != (ends[:, 1] > y)
        if not np.any(straddles):
            return False

        s = starts[straddles]
        e = ends[straddles]

        # x of the intersection with the horizontal line through the point.
        t = (y - s[:, 1]) / (e[:, 1] - s[:, 1])
        intersect_x = s[:, 0] + t * (e[:, 0] - s[:, 0])

        return bool(np.count_nonzero(intersect_x > x) % 2 == 1)

    def distance_to_boundary(self, point: np.ndarray) -> float:
        """Shortest distance from the point to any edge (unsigned)."""
        starts, ends = self._edges
        p = np.asarray(point, dtype=np.float64)

        segment = ends - starts
        length_squared = np.einsum("ij,ij->i", segment, segment)
        # Degenerate edges would divide by zero; treat them as their start.
        length_squared = np.where(length_squared < 1e-12, 1.0, length_squared)

        t = np.einsum("ij,ij->i", p - starts, segment) / length_squared
        t = np.clip(t, 0.0, 1.0)

        closest = starts + t[:, None] * segment
        distances = np.linalg.norm(p - closest, axis=1)
        return float(np.min(distances))

    def escape_vector(self, point: np.ndarray) -> np.ndarray:
        """Unit vector from the point to the nearest boundary point.

        For a point inside the zone this points outward — the way out.
        """
        starts, ends = self._edges
        p = np.asarray(point, dtype=np.float64)

        segment = ends - starts
        length_squared = np.einsum("ij,ij->i", segment, segment)
        length_squared = np.where(length_squared < 1e-12, 1.0, length_squared)

        t = np.clip(
            np.einsum("ij,ij->i", p - starts, segment) / length_squared, 0.0, 1.0
        )
        closest = starts + t[:, None] * segment
        distances = np.linalg.norm(p - closest, axis=1)

        nearest = closest[int(np.argmin(distances))]
        direction = nearest - p
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            # Exactly on the boundary; any outward direction will do.
            return np.array([1.0, 0.0], dtype=np.float64)
        return direction / norm

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()


class Geofence:
    """A set of no-fly polygons, checked every control step."""

    __slots__ = ("_zones", "_margin", "_enabled")

    def __init__(
        self,
        zones: List[Polygon] = None,
        margin: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self._zones = list(zones) if zones else []
        self._margin = margin
        self._enabled = enabled

    def check(self, position: np.ndarray) -> GeofenceStatus:
        """Evaluate a position against every zone.

        Args:
            position: (2,) world coordinates.

        Returns:
            GeofenceStatus. `inside` is True when the position is legal —
            i.e. NOT inside any no-fly zone and not within the margin of one.
        """
        if not self._enabled or not self._zones:
            return GeofenceStatus(
                inside=True, nearest_boundary_dist=float("inf"), violation_direction=None
            )

        p = np.asarray(position, dtype=np.float64)[:2]

        nearest_distance = float("inf")
        violating_zone: Optional[Polygon] = None
        inside_zone = False

        for zone in self._zones:
            distance = zone.distance_to_boundary(p)
            contained = zone.contains(p)

            if contained:
                # Inside a no-fly zone: the worst case, and the escape vector
                # must come from this zone regardless of what else is nearer.
                inside_zone = True
                violating_zone = zone
                nearest_distance = -distance
                break

            if distance < nearest_distance:
                nearest_distance = distance
                if distance < self._margin:
                    violating_zone = zone

        legal = not inside_zone and nearest_distance >= self._margin

        direction = None
        if violating_zone is not None:
            direction = violating_zone.escape_vector(p)
            if not inside_zone:
                # Outside but within the margin: push AWAY from the zone,
                # whereas escape_vector points toward its boundary.
                direction = -direction

        return GeofenceStatus(
            inside=legal,
            nearest_boundary_dist=nearest_distance,
            violation_direction=direction,
        )

    def is_legal(self, position: np.ndarray) -> bool:
        """Convenience predicate for route checks."""
        return self.check(position).inside

    def path_crosses_zone(
        self, start: np.ndarray, end: np.ndarray, samples: int = 24
    ) -> bool:
        """Whether a straight path between two points enters any zone.

        Sampled rather than solved analytically: the polygons are small
        relative to flight distances, and sampling handles concave shapes
        without special cases. `samples` trades precision for cost.
        """
        if not self._enabled or not self._zones:
            return False

        a = np.asarray(start, dtype=np.float64)[:2]
        b = np.asarray(end, dtype=np.float64)[:2]

        for i in range(samples + 1):
            t = i / samples
            point = a + t * (b - a)
            if not self.check(point).inside:
                return True
        return False

    @property
    def zones(self) -> List[Polygon]:
        return list(self._zones)

    @classmethod
    def from_config(cls, cfg) -> "Geofence":
        zones = []
        zone_list = cfg.geofence.get("zones", []) or []
        for entry in zone_list:
            zones.append(Polygon(entry["name"], entry["polygon"]))
        return cls(
            zones=zones,
            margin=cfg.geofence.margin,
            enabled=cfg.geofence.enabled,
        )
