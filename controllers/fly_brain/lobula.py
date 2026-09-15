"""Wide-field pooling — lobula plate tangential cells.

The conceptual heart of the design: thousands of local detectors collapse into
four numbers.

| Channel          | Computation                              |
|------------------|------------------------------------------|
| left_expansion   | Mean outward horizontal flow, left half  |
| right_expansion  | Mean outward horizontal flow, right half |
| vertical_flow    | Mean vertical flow, whole field          |
| rotation         | Mean uniform horizontal flow (yaw signal)|

Phase 3.
"""

from collections import namedtuple

WideField = namedtuple(
    "WideField", ["left_expansion", "right_expansion", "vertical_flow", "rotation"]
)


def pool(horizontal, vertical, yaw_rate):
    """Collapse a local flow field into the four wide-field channels.

    Rotation and expansion are confounded: yawing produces uniform horizontal
    flow across the whole field, which contaminates the expansion estimate.
    Subtract the rotational component before computing expansion. The fly does
    something equivalent with haltere feedback — we have an IMU, so cancel it
    with the measured yaw rate.

    Args:
        horizontal: local horizontal flow from the EMD array.
        vertical: local vertical flow from the EMD array.
        yaw_rate: measured yaw rate, rad/s, from the IMU.

    Returns:
        WideField.
    """
    raise NotImplementedError("Phase 3")
