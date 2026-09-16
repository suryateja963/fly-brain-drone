"""Telemetry — flight recording, incidents, and the landing report.

Two jobs:

  THE WAVE REPORT. A summary emitted at touchdown: duration, distance, peak
  altitude, energy used, and everything that went wrong. This is the client's
  stated requirement and the artefact an operator actually reads.

  THE INCIDENT LOG. Every failure, refusal and limit engagement, timestamped.
  A drone that quietly avoided a no-fly zone, rebalanced after losing a rotor
  and refused a payload release has done three noteworthy things, and a
  report saying only "flight completed" hides all of them.

Incidents are DEDUPLICATED by kind. A geofence refusal that persists for 400
control steps is one incident, not 400 — a log that floods is a log nobody
reads. The count is retained so the difference between a brush and a
sustained problem is still visible.
"""

import time
from typing import Dict, List, Optional

import numpy as np

from contracts import TelemetryReport


class IncidentLog:
    """Timestamped incidents, deduplicated by kind."""

    __slots__ = ("_incidents", "_counts", "_first_seen", "_max_size")

    def __init__(self, max_size: int = 64) -> None:
        self._incidents: List[Dict] = []
        self._counts: Dict[str, int] = {}
        self._first_seen: Dict[str, float] = {}
        self._max_size = max_size

    def record(self, kind: str, detail: str, timestamp: float) -> bool:
        """Log an incident.

        Args:
            kind: category, e.g. "geofence_refusal". Deduplication key.
            detail: human-readable specifics.
            timestamp: seconds since flight start.

        Returns:
            True if this was the first of its kind, i.e. genuinely new.
        """
        is_new = kind not in self._counts

        self._counts[kind] = self._counts.get(kind, 0) + 1

        if is_new:
            self._first_seen[kind] = timestamp
            if len(self._incidents) < self._max_size:
                self._incidents.append(
                    {
                        "kind": kind,
                        "detail": detail,
                        "timestamp": timestamp,
                    }
                )

        return is_new

    def summary(self) -> List[str]:
        """Human-readable lines for the landing report."""
        lines = []
        for incident in self._incidents:
            kind = incident["kind"]
            count = self._counts[kind]
            occurrences = "" if count == 1 else f" (x{count})"
            lines.append(
                f"[{incident['timestamp']:6.1f}s] {kind}: "
                f"{incident['detail']}{occurrences}"
            )
        return lines

    def count_of(self, kind: str) -> int:
        return self._counts.get(kind, 0)

    def has(self, kind: str) -> bool:
        return kind in self._counts

    @property
    def kinds(self) -> List[str]:
        return list(self._counts.keys())

    def clear(self) -> None:
        self._incidents.clear()
        self._counts.clear()
        self._first_seen.clear()


class TelemetryRecorder:
    """Accumulates flight statistics and produces the landing report."""

    __slots__ = (
        "_start_time",
        "_elapsed",
        "_distance",
        "_max_altitude",
        "_start_charge",
        "_last_charge",
        "_last_position",
        "_incidents",
        "_samples",
        "_airborne",
        "_landing_position",
    )

    def __init__(self, incident_log_size: int = 64) -> None:
        self._start_time: Optional[float] = None
        self._elapsed = 0.0
        self._distance = 0.0
        self._max_altitude = 0.0
        self._start_charge: Optional[float] = None
        self._last_charge = 0.0
        self._last_position: Optional[np.ndarray] = None
        self._incidents = IncidentLog(max_size=incident_log_size)
        self._samples = 0
        self._airborne = False
        self._landing_position = np.zeros(2, dtype=np.float64)

    def update(
        self,
        position: np.ndarray,
        altitude: float,
        charge_pct: float,
        dt: float,
    ) -> None:
        """Record one control step.

        Distance accumulates only while airborne, so a drone sitting on the
        pad with GPS noise does not log kilometres of travel.
        """
        p = np.asarray(position, dtype=np.float64)[:2]

        if self._start_time is None:
            self._start_time = time.monotonic()
            self._start_charge = charge_pct
            self._last_position = p.copy()

        self._elapsed += dt
        self._samples += 1
        self._last_charge = charge_pct

        # Airborne latches on the way up so the whole flight counts, but the
        # landing position is captured when it comes back down.
        if altitude > 0.15:
            self._airborne = True
        elif self._airborne:
            self._landing_position = p.copy()

        if altitude > self._max_altitude:
            self._max_altitude = altitude

        if self._airborne and self._last_position is not None:
            self._distance += float(np.linalg.norm(p - self._last_position))

        self._last_position = p.copy()

    def record_incident(self, kind: str, detail: str) -> bool:
        """Log an incident at the current flight time."""
        return self._incidents.record(kind, detail, self._elapsed)

    def report(self, landing_position: np.ndarray = None) -> TelemetryReport:
        """Build the wave report emitted at touchdown."""
        position = (
            self._landing_position
            if landing_position is None
            else np.asarray(landing_position, dtype=np.float64)[:2]
        )

        battery_used = 0.0
        if self._start_charge is not None:
            battery_used = max(0.0, self._start_charge - self._last_charge)

        return TelemetryReport(
            flight_duration=float(self._elapsed),
            distance_traveled=float(self._distance),
            max_altitude=float(self._max_altitude),
            battery_used=float(battery_used),
            incidents=self._incidents.summary(),
            landing_position=position.copy(),
        )

    def format_report(self, report: TelemetryReport = None) -> str:
        """The report as text, for the console and the demo panel."""
        r = report if report is not None else self.report()

        lines = [
            "=" * 58,
            "FLIGHT REPORT",
            "=" * 58,
            f"  Duration          {r.flight_duration:8.1f} s",
            f"  Distance flown    {r.distance_traveled:8.2f} m",
            f"  Max altitude      {r.max_altitude:8.2f} m",
            f"  Battery used      {r.battery_used:8.1f} %",
            f"  Landed at         ({r.landing_position[0]:+.2f}, "
            f"{r.landing_position[1]:+.2f})",
        ]

        if r.incidents:
            lines.append("")
            lines.append(f"  INCIDENTS ({len(r.incidents)})")
            for incident in r.incidents:
                lines.append(f"    {incident}")
        else:
            lines.append("")
            lines.append("  No incidents.")

        lines.append("=" * 58)
        return "\n".join(lines)

    @property
    def incidents(self) -> IncidentLog:
        return self._incidents

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def is_airborne(self) -> bool:
        return self._airborne

    def reset(self) -> None:
        self._start_time = None
        self._elapsed = 0.0
        self._distance = 0.0
        self._max_altitude = 0.0
        self._start_charge = None
        self._last_charge = 0.0
        self._last_position = None
        self._incidents.clear()
        self._samples = 0
        self._airborne = False
        self._landing_position = np.zeros(2, dtype=np.float64)

    @classmethod
    def from_config(cls, cfg) -> "TelemetryRecorder":
        return cls(incident_log_size=cfg.telemetry.incident_log_size)
