"""Data contracts crossing module boundaries.

Every structure that passes between layers is defined here and nowhere else.
No module invents its own dict or tuple for inter-layer communication: the
contract is the interface, and keeping it in one file means a change to a
layer's output is visible as a change to this file.

All dataclasses use __slots__ (the control loop runs at 125Hz, and slotted
attribute access avoids a per-instance dict) and carry full annotations.

The signal path these contracts describe:

    CameraFrame -> FlowField -> PooledChannels -> AvoidanceCommand ----+
                                                                       |
                        HeadingState + PathState -> GoalCommand -------+
                                                                       |
                                                    ArbitrationOutput <-+
                                                             |
                                                      MotorCommands
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Vision front-end
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CameraFrame:
    """A frame from the agent's deliberately coarse eye.

    The 20x40 shape is the thesis, not a limitation: fly vision is genuinely
    this low-resolution, and optic flow is more tractable on blurred input
    because fine texture creates aliasing in the correlators downstream.
    """

    timestamp: float
    image: np.ndarray  # (20, 40, 3) uint8

    def validate(self) -> None:
        if self.image.shape != (20, 40, 3):
            raise ValueError(f"expected (20, 40, 3), got {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {self.image.dtype}")


@dataclass(slots=True)
class FlowField:
    """Dense local optic flow, one 2-vector per retinal location.

    `valid` is False on the first frame of a sequence: a motion detector with
    no history cannot report motion, and saying so is more honest than
    comparing a frame against itself.
    """

    timestamp: float
    flow: np.ndarray  # (20, 40, 2) float32 — [..., 0] = x, [..., 1] = y
    valid: bool

    def validate(self) -> None:
        if self.flow.shape != (20, 40, 2):
            raise ValueError(f"expected (20, 40, 2), got {self.flow.shape}")

    @property
    def magnitude(self) -> np.ndarray:
        """Per-location flow magnitude, shape (20, 40)."""
        return np.linalg.norm(self.flow, axis=2)


# ---------------------------------------------------------------------------
# Wide-field pooling — lobula plate tangential cells
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PooledChannels:
    """Thousands of local detectors collapsed into four numbers.

    `raw_rotation_before_subtraction` is retained deliberately: rotation and
    expansion are confounded, because yawing produces uniform horizontal flow
    across the whole field. Keeping the pre-subtraction value makes the
    correction auditable rather than invisible — if expansion spikes during a
    yaw in open space, this field says whether the subtraction ran.
    """

    left_expansion: float
    right_expansion: float
    vertical_expansion: float
    rotation: float
    raw_rotation_before_subtraction: float

    @property
    def total_expansion(self) -> float:
        """Summed lateral expansion — the threat signal driving braking."""
        return self.left_expansion + self.right_expansion

    @property
    def expansion_difference(self) -> float:
        """Left minus right. Positive means the obstacle is on the left."""
        return self.left_expansion - self.right_expansion


# ---------------------------------------------------------------------------
# Avoidance — the innate reflex pathway
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AvoidanceCommand:
    """Output of the hardwired avoidance reflex.

    `urgency` in [0, 1] is what the arbitration layer blends on. It is a
    sigmoid of total expansion, never a boolean: a hard switch produces a
    dodge-return limit cycle.
    """

    turn_rate: float
    brake_force: float
    urgency: float

    def validate(self) -> None:
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError(f"urgency must be in [0, 1], got {self.urgency}")


# ---------------------------------------------------------------------------
# Central complex — heading and path integration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeadingState:
    """Ring attractor state: a single bump of activity encoding heading.

    `confidence` falls when the bump broadens or splits, which is the
    system's own signal that its heading estimate has degraded.
    """

    bump: np.ndarray  # (32,) float64
    peak_index: int
    heading_rad: float
    confidence: float

    def validate(self) -> None:
        if self.bump.ndim != 1:
            raise ValueError(f"bump must be 1-D, got {self.bump.ndim}-D")


@dataclass(slots=True)
class PathState:
    """Dead-reckoned position and the vector home.

    This is what makes GPS-denied return-to-home possible: the estimate is
    integrated from the drone's own motion, so it cannot be spoofed or jammed.
    Insects do exactly this.
    """

    position_estimate: np.ndarray  # (2,) float64
    home_vector: np.ndarray  # (2,) float64
    distance_to_home: float
    gps_available: bool


@dataclass(slots=True)
class GoalCommand:
    """Where the goal-seeking pathway wants to go.

    This is the single seam a higher-level planner (or a future language
    model, per the project spec) ever touches. Keeping it narrow is what
    keeps the architecture honest.
    """

    desired_heading: float
    desired_speed: float
    desired_altitude: float


# ---------------------------------------------------------------------------
# Arbitration — the two pathways compete
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ArbitrationOutput:
    """Sigmoid blend of avoidance and goal-seeking.

    The two weights are retained so the demo can show the blend happening
    live: watching avoidance_weight rise as an obstacle looms is the clearest
    visual explanation of the whole architecture.
    """

    final_heading: float
    final_speed: float
    final_altitude: float
    avoidance_weight: float
    goal_weight: float

    def validate(self) -> None:
        total = self.avoidance_weight + self.goal_weight
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1, got {total}")


# ---------------------------------------------------------------------------
# Output bottleneck — descending neurons
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotorCommands:
    """The four values that reach the rotors. Nothing else touches them.

    `altitude_capped` records that the safety ceiling engaged. It is enforced
    here, at the final stage, rather than in the planner — so no planning
    change can ever bypass it.
    """

    motors: np.ndarray  # (4,) float64
    clamped: bool
    altitude_capped: bool

    def validate(self) -> None:
        if self.motors.shape != (4,):
            raise ValueError(f"expected 4 motors, got {self.motors.shape}")


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GeofenceStatus:
    """Position relative to no-fly polygons.

    `violation_direction` is the unit vector pointing back to legal airspace,
    or None when inside. Giving the direction rather than a bare boolean lets
    the controller steer out rather than merely stop.
    """

    inside: bool
    nearest_boundary_dist: float
    violation_direction: Optional[np.ndarray] = None


@dataclass(slots=True)
class WindEstimate:
    """Estimated wind, derived from the gap between commanded and actual motion."""

    direction: float
    speed: float
    gust_detected: bool


# ---------------------------------------------------------------------------
# Failsafes and energy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatteryState:
    """Charge and the derived question that actually matters.

    `can_return_home` is computed against the live distance-to-home, not a
    fixed threshold: 20% charge is plenty at 10m and fatal at 500m.
    """

    charge_pct: float
    voltage: float
    can_return_home: bool
    emergency: bool


@dataclass(slots=True)
class MotorHealth:
    """Per-rotor health and whether the mix has compensated.

    A quadrotor with one dead rotor cannot hold yaw, but it can hold position
    by spinning and rebalancing the remaining three — `degraded_mode` says
    that trade has been made.
    """

    rotor_status: np.ndarray  # (4,) bool — True means healthy
    rebalanced: bool
    degraded_mode: bool


@dataclass(slots=True)
class VibrationState:
    """IMU spectral state. A cracked propeller shows as a peak at rotor frequency."""

    dominant_freq: float
    amplitude: float
    anomaly_detected: bool


@dataclass(slots=True)
class PayloadState:
    """Payload and drop-zone authorisation.

    `release_authorized` is the conjunction of every precondition, computed
    once so no caller can release by checking only some of them.
    """

    attached: bool
    drop_zone_clear: bool
    release_authorized: bool


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryReport:
    """Emitted at landing — the client's 'wave report'."""

    flight_duration: float
    distance_traveled: float
    max_altitude: float
    battery_used: float
    incidents: List[str] = field(default_factory=list)
    landing_position: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
