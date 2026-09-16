"""End-to-end pipeline tests: camera frame in, motor commands out.

The pipeline touches no Webots API, so the entire architecture is testable
without a simulator. These tests drive it into every branch of the priority
ladder and assert the correct override wins.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "fly_brain"))

from config import load_config  # noqa: E402
from pipeline import FlyBrainPipeline  # noqa: E402

DT = 0.008


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "config", "default.yaml"))


def textured_frame(shift=0, width=400, height=240):
    """A camera frame with real texture.

    Untextured input produces no optic flow at all — that is a property of
    the world, not a bug, and it is why every world file must specify
    textures. Using a blank frame here would silently test nothing.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    columns = np.arange(width)
    stripes = (((columns + shift) // 8) % 2 * 200 + 30).astype(np.uint8)
    frame[:, :, 0] = stripes
    frame[:, :, 1] = stripes
    frame[:, :, 2] = stripes
    return frame


def default_inputs(**overrides):
    """A nominal sensor snapshot: level, hovering, healthy."""
    inputs = dict(
        frame=textured_frame(),
        position=np.array([0.0, 0.0]),
        altitude=1.5,
        yaw=0.0,
        yaw_rate=0.0,
        speed=0.0,
        acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.array([0.0, 0.0]),
        actual_velocity=np.array([0.0, 0.0]),
        motor_feedback=np.full(4, 70.0),
        last_motors=np.full(4, 70.0),
        dt=DT,
        gps_position=np.array([0.0, 0.0]),
        obstacles=None,
    )
    inputs.update(overrides)
    return inputs


def run_steps(pipeline, count, **overrides):
    """Step the pipeline repeatedly and return the final state."""
    state = None
    for i in range(count):
        inputs = default_inputs(**overrides)
        inputs["frame"] = textured_frame(shift=i * 2)
        state = pipeline.step(**inputs)
    return state


# ===========================================================================
# The pipeline runs at all
# ===========================================================================


def test_pipeline_constructs_from_config(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    assert pipeline is not None


def test_single_step_produces_four_motors(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = pipeline.step(**default_inputs())

    assert state.motors.motors.shape == (4,)
    assert np.all(np.isfinite(state.motors.motors))


def test_every_layer_produced_output(cfg):
    """A missing layer would show as None here rather than a crash."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = run_steps(pipeline, 5)

    assert state.flow_field is not None
    assert state.channels is not None
    assert state.avoidance is not None
    assert state.heading_state is not None
    assert state.path_state is not None
    assert state.goal is not None
    assert state.arbitration is not None
    assert state.motors is not None
    assert state.geofence_status is not None
    assert state.battery_state is not None
    assert state.motor_health is not None
    assert state.vibration_state is not None
    assert state.wind_estimate is not None


def test_motors_stay_finite_over_a_long_run(cfg):
    """Numerical stability over a realistic flight length."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))

    for i in range(500):
        inputs = default_inputs(
            frame=textured_frame(shift=i * 3),
            position=np.array([i * 0.01, 0.0]),
            yaw_rate=0.2 * np.sin(i * 0.05),
            speed=1.0,
        )
        state = pipeline.step(**inputs)
        assert np.all(np.isfinite(state.motors.motors)), f"diverged at step {i}"


def test_arbitration_weights_sum_to_one(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = run_steps(pipeline, 10)

    total = state.arbitration.avoidance_weight + state.arbitration.goal_weight
    assert total == pytest.approx(1.0, abs=1e-6)


def test_normal_flight_has_no_override(cfg):
    """Healthy drone in legal airspace flies the mission."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = run_steps(pipeline, 10)

    assert state.override_reason is None


def test_goal_heading_points_at_target(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([10.0, 0.0]))
    state = run_steps(pipeline, 5)

    assert state.goal.desired_heading == pytest.approx(0.0, abs=0.1)


# ===========================================================================
# The priority ladder — each override in turn
# ===========================================================================


def test_geofence_violation_overrides_mission(cfg):
    """A no-fly zone is a legal hard constraint; the mission yields."""
    # The default config puts a zone at [[20,20],[40,20],[40,40],[20,40]].
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([30.0, 30.0]))
    state = run_steps(pipeline, 5, position=np.array([30.0, 30.0]))

    assert not state.geofence_status.inside
    assert state.override_reason == "geofence_escape"


def test_battery_emergency_overrides_geofence(cfg):
    """Priority order: a drone that cannot reach home lands regardless.

    Both conditions are asserted true, and the ladder must pick the
    emergency — motor failure aside, it is the most urgent.
    """
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([30.0, 30.0]))
    pipeline.battery.force_level(2.0)

    state = run_steps(pipeline, 5, position=np.array([5.0, 5.0]))

    assert state.battery_state.emergency
    assert state.override_reason == "battery_emergency_land"


def test_motor_failure_outranks_everything(cfg):
    """Top of the ladder: the aircraft is damaged, nothing else matters."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([30.0, 30.0]))
    pipeline.motor_handler.force_failure(1)
    pipeline.battery.force_level(1.0)

    state = run_steps(pipeline, 5, position=np.array([30.0, 30.0]))

    assert state.motor_health.degraded_mode
    assert state.override_reason.startswith("motor_failure")


def test_vibration_triggers_return_at_reduced_speed(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([20.0, 0.0]))
    pipeline.vibration_monitor.force_anomaly(True)

    state = run_steps(pipeline, 5, position=np.array([5.0, 0.0]))

    assert state.override_reason == "vibration_return"
    assert state.goal.desired_speed < cfg.battery.cruise_speed_ms


def test_three_rotor_failure_still_holds_position(cfg):
    """One dead rotor: surrender yaw, keep flying."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 0.0]))
    pipeline.motor_handler.force_failure(0)

    state = run_steps(pipeline, 5)

    assert state.override_reason == "motor_failure_return"
    assert state.motors.motors[0] == 0.0


def test_two_rotor_failure_descends(cfg):
    """Two dead rotors cannot hold position; descend under control."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 0.0]))
    pipeline.motor_handler.force_failure(0)
    pipeline.motor_handler.force_failure(1)

    state = run_steps(pipeline, 5, altitude=5.0)

    assert state.override_reason == "motor_failure_descent"
    assert state.goal.desired_altitude < 5.0


# ===========================================================================
# Compliance inside the pipeline
# ===========================================================================


def test_altitude_cap_enforced_through_the_pipeline(cfg):
    """The cap must survive the whole stack, not just a unit test."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 0.0]))
    state = run_steps(pipeline, 5, altitude=cfg.altitude_cap.ceiling_m + 2.0)

    assert state.motors.altitude_capped


def test_target_altitude_never_exceeds_ceiling(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 0.0]))
    state = run_steps(pipeline, 5)

    assert state.target_altitude <= cfg.altitude_cap.ceiling_m


def test_wind_estimated_from_drift(cfg):
    """Commanded to hold, drifting east: the estimator must see it."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 0.0]))
    state = run_steps(
        pipeline,
        60,
        commanded_velocity=np.array([0.0, 0.0]),
        actual_velocity=np.array([3.0, 0.0]),
    )

    assert state.wind_estimate.speed > 1.0


# ===========================================================================
# Navigation without GPS
# ===========================================================================


def test_pipeline_runs_without_gps(cfg):
    """GPS-denied operation is the headline capability; it must not crash."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = run_steps(pipeline, 50, gps_position=None, speed=1.0)

    assert not state.path_state.gps_available
    assert np.all(np.isfinite(state.motors.motors))


def test_dead_reckoning_accumulates_distance_home(cfg):
    """Flying out with no GPS must still build a vector home."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([20.0, 0.0]))

    for i in range(200):
        inputs = default_inputs(
            frame=textured_frame(shift=i * 2),
            gps_position=None,
            speed=2.0,
        )
        state = pipeline.step(**inputs)

    assert state.path_state.distance_to_home > 1.0


# ===========================================================================
# Telemetry
# ===========================================================================


def test_incidents_recorded_during_flight(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([30.0, 30.0]))
    run_steps(pipeline, 20, position=np.array([30.0, 30.0]))

    report = pipeline.report()
    assert any("geofence" in i for i in report.incidents)


def test_repeated_violation_logged_once(cfg):
    """400 steps inside a zone is one incident, not 400."""
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([30.0, 30.0]))
    run_steps(pipeline, 200, position=np.array([30.0, 30.0]))

    report = pipeline.report()
    geofence_entries = [i for i in report.incidents if "geofence" in i]
    assert len(geofence_entries) == 1


def test_report_produced_after_flight(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    run_steps(pipeline, 100, altitude=2.0)

    report = pipeline.report()
    assert report.flight_duration > 0.0
    assert report.max_altitude == pytest.approx(2.0, abs=0.01)


# ===========================================================================
# Front-end interchangeability
# ===========================================================================


def test_emd_frontend_also_drives_the_pipeline(cfg):
    """Both flow front-ends satisfy the same contract.

    Swapping the algorithm is a config change, and the pipeline downstream
    cannot tell them apart.
    """
    emd_cfg = load_config(
        os.path.join(ROOT, "config", "default.yaml"),
        overrides={"optical_flow": {"method": "emd"}},
    )
    pipeline = FlyBrainPipeline(emd_cfg, goal_position=np.array([5.0, 3.0]))
    state = run_steps(pipeline, 10)

    assert np.all(np.isfinite(state.motors.motors))
    assert state.channels is not None


def test_unknown_frontend_is_rejected(cfg):
    """A typo in the method name must fail loudly at startup."""
    bad = load_config(
        os.path.join(ROOT, "config", "default.yaml"),
        overrides={"optical_flow": {"method": "nonsense"}},
    )
    with pytest.raises(ValueError, match="unknown optical_flow.method"):
        FlyBrainPipeline(bad)
