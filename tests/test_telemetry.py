"""Telemetry tests — the wave report and incident log."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "controllers",
        "fly_brain",
    ),
)

from telemetry import IncidentLog, TelemetryRecorder  # noqa: E402

DT = 0.008


# ===========================================================================
# Incident log
# ===========================================================================


def test_first_incident_is_new():
    log = IncidentLog()
    assert log.record("geofence", "entered zone A", 1.0)


def test_repeat_incident_is_not_new():
    """A refusal persisting for 400 steps is one incident, not 400."""
    log = IncidentLog()
    log.record("geofence", "entered zone A", 1.0)

    assert not log.record("geofence", "entered zone A", 1.1)


def test_repeat_count_is_retained():
    """Deduplication must not hide the difference between a brush and a
    sustained problem."""
    log = IncidentLog()
    for i in range(400):
        log.record("geofence", "entered zone A", float(i))

    assert log.count_of("geofence") == 400
    assert len(log.summary()) == 1
    assert "x400" in log.summary()[0]


def test_single_occurrence_has_no_count_suffix():
    log = IncidentLog()
    log.record("motor_failure", "rotor 2 dead", 5.0)

    assert "x1" not in log.summary()[0]


def test_different_kinds_logged_separately():
    log = IncidentLog()
    log.record("geofence", "zone A", 1.0)
    log.record("battery", "low charge", 2.0)
    log.record("motor_failure", "rotor 1", 3.0)

    assert len(log.summary()) == 3
    assert set(log.kinds) == {"geofence", "battery", "motor_failure"}


def test_first_timestamp_is_kept():
    """When the problem STARTED is the useful fact, not when it last recurred."""
    log = IncidentLog()
    log.record("vibration", "anomaly", 10.0)
    log.record("vibration", "anomaly", 90.0)

    assert "10.0" in log.summary()[0]


def test_log_respects_max_size():
    log = IncidentLog(max_size=3)
    for i in range(10):
        log.record(f"kind_{i}", "detail", float(i))

    assert len(log.summary()) == 3


def test_has_and_count_for_unseen_kind():
    log = IncidentLog()
    assert not log.has("nothing")
    assert log.count_of("nothing") == 0


def test_clear_empties_the_log():
    log = IncidentLog()
    log.record("geofence", "zone A", 1.0)
    log.clear()

    assert log.summary() == []
    assert not log.has("geofence")


# ===========================================================================
# Flight recording
# ===========================================================================


def test_duration_accumulates():
    recorder = TelemetryRecorder()
    for _ in range(125):
        recorder.update(np.array([0.0, 0.0]), altitude=1.0, charge_pct=100.0, dt=DT)

    assert recorder.elapsed == pytest.approx(1.0, abs=1e-6)


def test_max_altitude_tracked():
    recorder = TelemetryRecorder()
    for altitude in (0.5, 3.2, 1.8, 7.4, 2.0):
        recorder.update(
            np.array([0.0, 0.0]), altitude=altitude, charge_pct=100.0, dt=DT
        )

    assert recorder.report().max_altitude == pytest.approx(7.4)


def test_distance_accumulates_while_airborne():
    recorder = TelemetryRecorder()
    for i in range(11):
        recorder.update(
            np.array([float(i), 0.0]), altitude=2.0, charge_pct=100.0, dt=DT
        )

    assert recorder.distance == pytest.approx(10.0, abs=0.01)


def test_ground_noise_does_not_accumulate_distance():
    """A drone on the pad with GPS jitter must not log kilometres."""
    recorder = TelemetryRecorder()
    for i in range(100):
        jitter = np.array([0.05 * (i % 2), 0.05 * ((i + 1) % 2)])
        recorder.update(jitter, altitude=0.02, charge_pct=100.0, dt=DT)

    assert recorder.distance == 0.0
    assert not recorder.is_airborne


def test_battery_used_is_the_difference():
    recorder = TelemetryRecorder()
    recorder.update(np.array([0.0, 0.0]), altitude=1.0, charge_pct=100.0, dt=DT)
    recorder.update(np.array([0.0, 0.0]), altitude=1.0, charge_pct=62.0, dt=DT)

    assert recorder.report().battery_used == pytest.approx(38.0)


def test_landing_position_captured_on_touchdown():
    recorder = TelemetryRecorder()

    recorder.update(np.array([0.0, 0.0]), altitude=2.0, charge_pct=100.0, dt=DT)
    recorder.update(np.array([5.0, 3.0]), altitude=2.0, charge_pct=95.0, dt=DT)
    recorder.update(np.array([5.0, 3.0]), altitude=0.05, charge_pct=94.0, dt=DT)

    report = recorder.report()
    assert report.landing_position[0] == pytest.approx(5.0)
    assert report.landing_position[1] == pytest.approx(3.0)


def test_report_includes_incidents():
    recorder = TelemetryRecorder()
    recorder.update(np.array([0.0, 0.0]), altitude=1.0, charge_pct=100.0, dt=DT)
    recorder.record_incident("geofence", "refused entry to airport_approach")

    report = recorder.report()
    assert len(report.incidents) == 1
    assert "geofence" in report.incidents[0]


def test_formatted_report_is_readable():
    recorder = TelemetryRecorder()
    for _ in range(125):
        recorder.update(
            np.array([1.0, 1.0]), altitude=3.0, charge_pct=80.0, dt=DT
        )
    recorder.record_incident("battery", "return-to-home triggered")

    text = recorder.format_report()

    assert "FLIGHT REPORT" in text
    assert "Duration" in text
    assert "INCIDENTS" in text
    assert "battery" in text


def test_clean_flight_reports_no_incidents():
    recorder = TelemetryRecorder()
    recorder.update(np.array([0.0, 0.0]), altitude=1.0, charge_pct=100.0, dt=DT)

    assert "No incidents." in recorder.format_report()


def test_reset_clears_everything():
    recorder = TelemetryRecorder()
    for _ in range(100):
        recorder.update(
            np.array([5.0, 5.0]), altitude=4.0, charge_pct=70.0, dt=DT
        )
    recorder.record_incident("test", "detail")

    recorder.reset()

    assert recorder.elapsed == 0.0
    assert recorder.distance == 0.0
    assert recorder.report().max_altitude == 0.0
    assert recorder.report().incidents == []
