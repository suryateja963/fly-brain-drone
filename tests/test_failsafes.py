"""Failsafe tests: battery, motor failure, vibration, payload.

Each failure is injected and the correct response asserted. These are the
client's safety requirements, and a test that merely constructs the object
proves nothing — every test here drives the module into its failure state.
"""

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

from battery import BatteryModel  # noqa: E402
from motor_failure import MotorFailureHandler  # noqa: E402
from payload import PayloadManager  # noqa: E402
from vibration import VibrationMonitor  # noqa: E402

DT = 0.008


# ===========================================================================
# Battery — the question is "can it get home", not "how much is left"
# ===========================================================================


def test_charge_depletes_under_load():
    battery = BatteryModel()
    start = battery.charge_pct

    for _ in range(1000):
        battery.update(total_thrust=280.0, distance_to_home=5.0, dt=DT)

    assert battery.charge_pct < start


def test_higher_thrust_drains_faster():
    light = BatteryModel()
    heavy = BatteryModel()

    for _ in range(1000):
        light.update(total_thrust=100.0, distance_to_home=5.0, dt=DT)
        heavy.update(total_thrust=400.0, distance_to_home=5.0, dt=DT)

    assert heavy.charge_pct < light.charge_pct


def test_same_charge_different_verdict_by_distance():
    """THE CENTRAL TEST.

    Identical charge, different distance home, opposite verdicts. This is
    why a fixed percentage threshold is the wrong abstraction: 20% is ample
    at 10 metres and fatal at 500.
    """
    near = BatteryModel(initial_charge_pct=20.0)
    far = BatteryModel(initial_charge_pct=20.0)

    near_state = near.update(total_thrust=280.0, distance_to_home=10.0, dt=DT)
    far_state = far.update(total_thrust=280.0, distance_to_home=5000.0, dt=DT)

    assert near_state.can_return_home
    assert not far_state.can_return_home


def test_emergency_when_return_impossible():
    """Unable to reach home means emergency, which is the ping condition."""
    battery = BatteryModel(initial_charge_pct=5.0)
    state = battery.update(total_thrust=280.0, distance_to_home=8000.0, dt=DT)

    assert state.emergency
    assert not state.can_return_home


def test_emergency_latches():
    """Emergency must not flicker as charge hovers at a threshold.

    A drone oscillating between 'mission' and 'return' is worse than one
    committed to either.
    """
    battery = BatteryModel(initial_charge_pct=9.0)
    battery.update(total_thrust=280.0, distance_to_home=10.0, dt=DT)

    battery.force_level(100.0)
    state = battery.update(total_thrust=280.0, distance_to_home=1.0, dt=DT)

    assert state.emergency, "emergency un-latched after recovery"


def test_voltage_falls_off_steeply_when_low():
    """Lithium packs hold voltage then collapse; linear would mislead."""
    battery = BatteryModel()

    battery.force_level(80.0)
    high = battery.update(total_thrust=0.0, distance_to_home=1.0, dt=DT).voltage

    battery.force_level(50.0)
    mid = battery.update(total_thrust=0.0, distance_to_home=1.0, dt=DT).voltage

    battery.force_level(5.0)
    low = battery.update(total_thrust=0.0, distance_to_home=1.0, dt=DT).voltage

    assert high > mid > low
    assert (mid - low) > (high - mid), "voltage curve is not steepening"


def test_charge_never_goes_negative():
    battery = BatteryModel(initial_charge_pct=1.0)
    for _ in range(50000):
        battery.update(total_thrust=500.0, distance_to_home=5.0, dt=DT)

    assert battery.charge_pct >= 0.0


def test_max_range_shrinks_as_charge_falls():
    battery = BatteryModel()
    full = battery.max_range(current_watts=200.0)

    battery.force_level(25.0)
    quarter = battery.max_range(current_watts=200.0)

    assert quarter < full


def test_rejects_invalid_initial_charge():
    with pytest.raises(ValueError):
        BatteryModel(initial_charge_pct=150.0)


# ===========================================================================
# Motor failure — three rotors, no yaw, controlled descent
# ===========================================================================


def test_healthy_rotors_report_no_failure():
    handler = MotorFailureHandler()
    commanded = np.array([70.0, 70.0, 70.0, 70.0])

    health = handler.detect(commanded, commanded)

    assert np.all(health.rotor_status)
    assert not health.degraded_mode


def test_single_bad_sample_does_not_declare_failure():
    """A transient must not trigger a degraded landing."""
    handler = MotorFailureHandler(confirm_samples=8)
    commanded = np.array([70.0, 70.0, 70.0, 70.0])
    achieved = np.array([70.0, 0.0, 70.0, 70.0])

    health = handler.detect(commanded, achieved)

    assert np.all(health.rotor_status), "declared dead on one sample"


def test_sustained_failure_is_detected():
    handler = MotorFailureHandler(confirm_samples=8)
    commanded = np.array([70.0, 70.0, 70.0, 70.0])
    achieved = np.array([70.0, 0.0, 70.0, 70.0])

    for _ in range(10):
        health = handler.detect(commanded, achieved)

    assert not health.rotor_status[1]
    assert health.degraded_mode
    assert health.rebalanced


def test_rebalance_zeroes_dead_rotor():
    handler = MotorFailureHandler()
    handler.force_failure(1)
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    adjusted = handler.rebalance(np.full(4, 70.0), health)

    assert adjusted[1] == 0.0


def test_rebalance_reduces_diagonal_partner():
    """Full thrust opposite a dead rotor flips the airframe.

    Rotor 1 (front right) pairs diagonally with rotor 2 (rear left).
    """
    handler = MotorFailureHandler()
    handler.force_failure(1)
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    adjusted = handler.rebalance(np.full(4, 70.0), health)

    assert adjusted[2] < 70.0, "diagonal partner not reduced"


def test_rebalance_redistributes_lost_lift():
    """Surviving rotors take up the slack, so it descends rather than drops."""
    handler = MotorFailureHandler()
    handler.force_failure(0)
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    adjusted = handler.rebalance(np.full(4, 70.0), health)
    alive = adjusted[[1, 3]]

    assert np.any(alive > 70.0), "no lift redistributed to survivors"


def test_three_rotors_can_hold_position():
    handler = MotorFailureHandler()
    handler.force_failure(2)
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    assert handler.can_hold_position(health)


def test_two_rotors_cannot_hold_position():
    handler = MotorFailureHandler()
    handler.force_failure(0)
    handler.force_failure(1)
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    assert not handler.can_hold_position(health)


def test_descent_target_decreases():
    handler = MotorFailureHandler(controlled_descent_rate=0.5)
    target = handler.descent_target(current_altitude=5.0, dt=1.0)

    assert target == pytest.approx(4.5)


def test_descent_target_never_negative():
    handler = MotorFailureHandler(controlled_descent_rate=0.5)
    assert handler.descent_target(current_altitude=0.1, dt=10.0) == 0.0


def test_reset_restores_all_rotors():
    handler = MotorFailureHandler()
    handler.force_failure(3)
    handler.reset()
    health = handler.detect(np.full(4, 70.0), np.full(4, 70.0))

    assert np.all(health.rotor_status)


def test_rejects_bad_rotor_index():
    handler = MotorFailureHandler()
    with pytest.raises(ValueError):
        handler.force_failure(9)


# ===========================================================================
# Vibration — spectral, not magnitude
# ===========================================================================


def sine_samples(monitor, freq, amplitude, count, rate=125.0):
    """Feed a pure tone at a given frequency."""
    state = None
    for i in range(count):
        value = amplitude * np.sin(2.0 * np.pi * freq * i / rate)
        state = monitor.update(np.array([0.0, 0.0, value]))
    return state


def test_no_state_until_window_fills():
    monitor = VibrationMonitor(window_size=128)
    state = monitor.update(np.array([0.0, 0.0, 1.0]))

    assert state.dominant_freq == 0.0
    assert not state.anomaly_detected


def test_clean_rotor_produces_no_anomaly():
    """Low-amplitude vibration at rotor frequency is normal operation."""
    monitor = VibrationMonitor(
        window_size=128, anomaly_amplitude=0.35, rotor_freq_hz=40.0
    )
    state = sine_samples(monitor, freq=40.0, amplitude=0.05, count=600)

    assert not state.anomaly_detected


def test_cracked_propeller_is_detected():
    """Large amplitude AT rotor frequency is the imbalance signature."""
    monitor = VibrationMonitor(
        window_size=128, anomaly_amplitude=0.35, rotor_freq_hz=40.0
    )
    state = sine_samples(monitor, freq=40.0, amplitude=2.0, count=800)

    assert state.anomaly_detected
    assert state.dominant_freq == pytest.approx(40.0, abs=6.0)


def test_off_frequency_vibration_is_ignored():
    """THE REASON THIS IS SPECTRAL.

    Large vibration far from rotor frequency is a manoeuvre or turbulence,
    not a cracked blade. A magnitude threshold would trip on it constantly.
    """
    monitor = VibrationMonitor(
        window_size=128,
        anomaly_amplitude=0.35,
        rotor_freq_hz=40.0,
        freq_tolerance_hz=5.0,
    )
    state = sine_samples(monitor, freq=8.0, amplitude=3.0, count=800)

    assert not state.anomaly_detected, "tripped on non-rotor vibration"


def test_gravity_does_not_mask_the_signal():
    """DC removal matters: gravity dominates the vertical axis."""
    monitor = VibrationMonitor(
        window_size=128, anomaly_amplitude=0.35, rotor_freq_hz=40.0
    )

    state = None
    for i in range(800):
        value = 9.81 + 2.0 * np.sin(2.0 * np.pi * 40.0 * i / 125.0)
        state = monitor.update(np.array([0.0, 0.0, value]))

    assert state.anomaly_detected, "gravity swamped the vibration peak"


def test_anomaly_latches():
    """A propeller does not heal."""
    monitor = VibrationMonitor(
        window_size=128, anomaly_amplitude=0.35, rotor_freq_hz=40.0
    )
    sine_samples(monitor, freq=40.0, amplitude=2.0, count=800)
    assert monitor.is_latched

    state = sine_samples(monitor, freq=40.0, amplitude=0.001, count=400)
    assert state.anomaly_detected, "un-declared a cracked propeller"


def test_anomaly_reduces_speed():
    monitor = VibrationMonitor(reduced_speed_scale=0.5)
    monitor.force_anomaly(True)
    state = monitor.update(np.array([0.0, 0.0, 0.0]))

    assert monitor.speed_scale(state) == 0.5
    assert monitor.should_return_to_base(state)


def test_healthy_speed_scale_is_unity():
    monitor = VibrationMonitor()
    state = monitor.update(np.array([0.0, 0.0, 0.0]))

    assert monitor.speed_scale(state) == 1.0


def test_rejects_tiny_window():
    with pytest.raises(ValueError):
        VibrationMonitor(window_size=8)


# ===========================================================================
# Payload — refuse onto people, confirm detachment
# ===========================================================================


def test_release_authorized_when_all_conditions_met():
    payload = PayloadManager(drop_zone_radius=2.0, min_release_altitude=1.0)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )

    assert state.release_authorized
    assert state.drop_zone_clear


def test_person_in_zone_blocks_release():
    """THE SAFETY TEST. A parcel dropped on someone is a serious injury."""
    payload = PayloadManager(drop_zone_radius=2.0)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[np.array([1.0, 0.5])],
    )

    assert not state.drop_zone_clear
    assert not state.release_authorized
    assert "drop zone occupied" in payload.abort_reasons


def test_obstacle_outside_radius_does_not_block():
    payload = PayloadManager(drop_zone_radius=2.0)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[np.array([50.0, 50.0])],
    )

    assert state.release_authorized


def test_release_refused_when_not_over_target():
    payload = PayloadManager(drop_zone_radius=2.0)
    state = payload.assess(
        drone_position=np.array([30.0, 30.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )

    assert not state.release_authorized
    assert "not over drop zone" in payload.abort_reasons


def test_release_refused_when_too_low():
    payload = PayloadManager(min_release_altitude=1.0)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=0.2,
        obstacles=[],
    )

    assert not state.release_authorized
    assert any("below minimum" in r for r in payload.abort_reasons)


def test_command_release_refused_without_authorisation():
    payload = PayloadManager()
    state = payload.assess(
        drone_position=np.array([99.0, 99.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )

    assert not payload.command_release(state, now=0.0)


def test_detachment_confirmed_by_mass_change():
    """Confirmed by measurement, never assumed from the actuator command."""
    payload = PayloadManager(payload_mass=0.5)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )
    payload.command_release(state, now=0.0)

    confirmed, snagged = payload.confirm_detachment(
        measured_mass_delta=0.5, now=0.2
    )

    assert confirmed
    assert not snagged
    assert not payload.attached


def test_snagged_payload_detected_after_timeout():
    """THE FLYAWAY TEST.

    No mass change after the timeout means the payload is hanging. The
    controller is trimmed for a released state and the mass is asymmetric.
    """
    payload = PayloadManager(payload_mass=0.5, confirm_timeout_s=1.5)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )
    payload.command_release(state, now=0.0)

    confirmed, snagged = payload.confirm_detachment(
        measured_mass_delta=0.0, now=2.0
    )

    assert not confirmed
    assert snagged
    assert payload.attached, "reported detached while still carrying"


def test_partial_mass_change_is_not_confirmation():
    """A partially hanging payload is worse than either extreme."""
    payload = PayloadManager(payload_mass=0.5, confirm_timeout_s=1.5)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )
    payload.command_release(state, now=0.0)

    confirmed, _ = payload.confirm_detachment(measured_mass_delta=0.1, now=0.5)

    assert not confirmed


def test_forced_occupancy_blocks_release():
    """Failure injection for the demo."""
    payload = PayloadManager()
    payload.force_zone_occupied(True)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )

    assert not state.release_authorized


def test_detached_payload_cannot_be_released_again():
    payload = PayloadManager(attached=False)
    state = payload.assess(
        drone_position=np.array([0.0, 0.0]),
        drop_target=np.array([0.0, 0.0]),
        altitude=3.0,
        obstacles=[],
    )

    assert not state.release_authorized
    assert "no payload attached" in payload.abort_reasons
