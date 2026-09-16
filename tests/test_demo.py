"""Demo layer tests: five-panel rendering and failure injection.

The demo is the deliverable the client actually watches, so a panel that
silently renders nothing is a delivery failure. These tests assert pixels
change, not merely that functions return.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controllers", "fly_brain"))
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from config import load_config  # noqa: E402
from five_panel import FivePanelView  # noqa: E402
from inject_failures import FailureInjector, write_flags  # noqa: E402
from pipeline import FlyBrainPipeline  # noqa: E402

DT = 0.008


@pytest.fixture
def cfg():
    return load_config(os.path.join(ROOT, "config", "default.yaml"))


def textured_frame(shift=0, width=400, height=240):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    columns = np.arange(width)
    stripes = (((columns + shift) // 8) % 2 * 200 + 30).astype(np.uint8)
    frame[:, :, 0] = stripes
    frame[:, :, 1] = stripes
    frame[:, :, 2] = stripes
    return frame


def flown_state(pipeline, steps=10):
    """Run the pipeline far enough to have real values in every field."""
    state = None
    for i in range(steps):
        state = pipeline.step(
            frame=textured_frame(shift=i * 3),
            position=np.array([i * 0.1, 0.0]),
            altitude=1.5,
            yaw=0.2,
            yaw_rate=0.1,
            speed=1.0,
            acceleration=np.array([0.0, 0.0, 9.81]),
            commanded_velocity=np.array([1.0, 0.0]),
            actual_velocity=np.array([1.0, 0.0]),
            motor_feedback=np.full(4, 70.0),
            last_motors=np.full(4, 70.0),
            dt=DT,
            gps_position=np.array([i * 0.1, 0.0]),
        )
    return state


# ===========================================================================
# Five-panel rendering
# ===========================================================================


def test_renders_at_requested_size():
    view = FivePanelView(width=1280, height=720)
    canvas = view.render(state=None)

    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8


def test_renders_with_no_state_at_all():
    """A frame before the first pipeline step must not crash the recorder."""
    view = FivePanelView()
    canvas = view.render(state=None)

    assert canvas is not None
    # Panel chrome should still be drawn, so the frame is not uniform.
    assert len(np.unique(canvas.reshape(-1, 3), axis=0)) > 1


def test_renders_a_real_pipeline_state(cfg):
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    canvas = view.render(state=state, elapsed=1.5)

    assert canvas.shape == (720, 1280, 3)
    assert np.any(canvas > 0)


def test_agent_eye_panel_draws_the_retina(cfg):
    """The retina panel must show the retina, not stay empty.

    Compared against the same frame rendered without a retina: if the panel
    is drawing, those two differ.
    """
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    retina = np.random.RandomState(0).rand(20, 40)

    without = view.render(state=state)
    with_retina = view.render(state=state, retina=retina)

    assert not np.array_equal(without, with_retina), "retina panel drew nothing"


def test_chase_panel_draws_the_frame(cfg):
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    without = view.render(state=state)
    with_chase = view.render(state=state, chase_frame=textured_frame())

    assert not np.array_equal(without, with_chase), "chase panel drew nothing"


def test_flow_panel_responds_to_flow(cfg):
    """Different flow must produce a different picture.

    A heatmap that looks the same regardless of input is showing nothing.
    """
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    quiet = view.render(state=state)

    state.flow_field.flow[:, :, 0] = 2.0
    state.flow_field.flow[:, :, 1] = -1.0
    busy = view.render(state=state)

    assert not np.array_equal(quiet, busy), "flow panel ignored the flow field"


def test_channel_bars_respond_to_channel_values(cfg):
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    state.channels.left_expansion = 0.0
    state.channels.right_expansion = 0.0
    neutral = view.render(state=state)

    state.channels.left_expansion = 0.8
    state.channels.right_expansion = -0.6
    deflected = view.render(state=state)

    assert not np.array_equal(neutral, deflected), "bars ignored the channels"


def test_ring_panel_responds_to_heading(cfg):
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    state.heading_state.heading_rad = 0.0
    north = view.render(state=state)

    state.heading_state.heading_rad = np.pi
    south = view.render(state=state)

    assert not np.array_equal(north, south), "ring ignored the heading"


def test_override_banner_appears(cfg):
    """The override banner is the most important thing on screen when the
    mission stops. It must be visibly different."""
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    state.override_reason = None
    normal = view.render(state=state)

    state.override_reason = "battery_emergency_land"
    emergency = view.render(state=state)

    assert not np.array_equal(normal, emergency), "banner did not render"


def test_handles_invalid_flow_field(cfg):
    """The first frame has no flow history; the panel must cope."""
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline, steps=1)

    canvas = view.render(state=state)
    assert canvas.shape == (720, 1280, 3)


def test_handles_zero_flow_without_dividing_by_zero(cfg):
    view = FivePanelView()
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    state = flown_state(pipeline)

    state.flow_field.flow[:] = 0.0
    canvas = view.render(state=state)

    assert np.all(np.isfinite(canvas.astype(float)))


# ===========================================================================
# Failure injection — in process
# ===========================================================================


def test_kill_motor_takes_effect(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline)

    injector.kill_motor(2)
    state = flown_state(pipeline)

    assert not state.motor_health.rotor_status[2]
    assert state.motor_health.degraded_mode


def test_set_battery_takes_effect(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline)

    injector.set_battery(4.0)
    state = flown_state(pipeline)

    assert state.battery_state.charge_pct < 10.0
    assert state.battery_state.emergency


def test_trigger_vibration_takes_effect(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline)

    injector.trigger_vibration()
    state = flown_state(pipeline)

    assert state.vibration_state.anomaly_detected


def test_occupy_drop_zone_blocks_release(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([0.0, 0.0]))
    injector = FailureInjector(pipeline)
    injector.occupy_drop_zone()

    state = pipeline.step(
        frame=textured_frame(),
        position=np.array([0.0, 0.0]),
        altitude=3.0,
        yaw=0.0,
        yaw_rate=0.0,
        speed=0.0,
        acceleration=np.array([0.0, 0.0, 9.81]),
        commanded_velocity=np.zeros(2),
        actual_velocity=np.zeros(2),
        motor_feedback=np.full(4, 70.0),
        last_motors=np.full(4, 70.0),
        dt=DT,
        obstacles=[],
    )

    assert not state.payload_state.release_authorized


def test_clear_all_resets_every_failure(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline)

    injector.kill_motor(0)
    injector.trigger_vibration()
    injector.set_battery(5.0)
    injector.clear_all()

    state = flown_state(pipeline)

    assert np.all(state.motor_health.rotor_status)
    assert not state.vibration_state.anomaly_detected
    assert state.battery_state.charge_pct > 90.0


def test_applied_records_what_was_injected(cfg):
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline)

    injector.kill_motor(1)
    injector.set_battery(30.0)

    assert "motor_1" in injector.applied
    assert injector.applied["battery"] == 30.0


# ===========================================================================
# Failure injection — file based, for live demos
# ===========================================================================


def test_poll_applies_flags_from_file(cfg, tmp_path):
    flag_path = str(tmp_path / "flags.json")
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline, flag_path=flag_path)

    write_flags({"motor": 3}, flag_path)

    assert injector.poll()
    state = flown_state(pipeline)
    assert not state.motor_health.rotor_status[3]


def test_poll_without_a_file_is_harmless(cfg, tmp_path):
    """The common case every control step: no file, no work."""
    injector = FailureInjector(
        FlyBrainPipeline(cfg), flag_path=str(tmp_path / "absent.json")
    )

    assert not injector.poll()


def test_poll_only_applies_once_per_write(cfg, tmp_path):
    """Re-applying the same flag every step would spam the incident log."""
    flag_path = str(tmp_path / "flags.json")
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline, flag_path=flag_path)

    write_flags({"vibration": True}, flag_path)

    assert injector.poll()
    assert not injector.poll(), "re-applied an unchanged flag file"


def test_poll_survives_a_corrupt_file(cfg, tmp_path):
    """A half-written file must not crash the flight.

    The writer renames atomically, but a truncated file from any other
    source must still be survivable: the flight matters more than the flag.
    """
    flag_path = str(tmp_path / "flags.json")
    with open(flag_path, "w", encoding="utf-8") as handle:
        handle.write('{"motor": ')

    injector = FailureInjector(FlyBrainPipeline(cfg), flag_path=flag_path)

    assert not injector.poll()


def test_clear_flag_resets(cfg, tmp_path):
    flag_path = str(tmp_path / "flags.json")
    pipeline = FlyBrainPipeline(cfg, goal_position=np.array([5.0, 3.0]))
    injector = FailureInjector(pipeline, flag_path=flag_path)

    injector.kill_motor(0)
    write_flags({"clear": True}, flag_path)
    injector.poll()

    state = flown_state(pipeline)
    assert np.all(state.motor_health.rotor_status)


def test_write_flags_produces_valid_json(tmp_path):
    flag_path = str(tmp_path / "flags.json")
    write_flags({"motor": 2, "battery": 15.0}, flag_path)

    with open(flag_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded["motor"] == 2
    assert loaded["battery"] == 15.0
