"""Wide-field pooling — lobula plate tangential cells.

The conceptual heart of the design: thousands of local detectors collapse
into four numbers. In the fly these are a handful of giant interneurons, each
integrating the whole visual field with a particular spatial weighting.

THE CRITICAL OPERATION IS ROTATION SUBTRACTION, AND IT HAPPENS FIRST.

Rotation and expansion are confounded. Yawing produces uniform horizontal
flow across the entire field, which reads as expansion on the side you are
turning toward — a phantom obstacle that appears whenever the drone turns in
open space. The fly resolves this using haltere feedback (its gyroscopes);
we have an IMU, so we use the measured yaw rate.

Subtract rotation BEFORE computing expansion, never after. Computing
expansion from contaminated flow and then correcting the scalar cannot work:
the contamination is asymmetric across the field, so it does not factor out
of the pooled value.
"""

import numpy as np

from contracts import FlowField, PooledChannels


class WideFieldPooling:
    """Collapse a dense flow field into four channels.

    | Channel            | Computation                                  |
    |--------------------|----------------------------------------------|
    | left_expansion     | Mean outward horizontal flow, left half      |
    | right_expansion    | Mean outward horizontal flow, right half     |
    | vertical_expansion | Mean vertical flow magnitude, whole field    |
    | rotation           | Mean uniform horizontal flow (yaw signal)    |
    """

    __slots__ = (
        "_split_fraction",
        "_horizon_row_start",
        "_yaw_rate_to_flow",
        "_left_slice",
        "_right_slice",
        "_rows",
        "_cols",
    )

    def __init__(
        self,
        rows: int = 20,
        cols: int = 40,
        split_fraction: float = 0.5,
        horizon_row_start: int = 4,
        yaw_rate_to_flow: float = 1.0,
    ) -> None:
        """Args:
        rows, cols: retina dimensions.
        split_fraction: fraction of width treated as the left half.
        horizon_row_start: rows above this are excluded from lateral
            expansion. The top of the frame is usually sky, which
            contributes noise rather than obstacle signal.
        yaw_rate_to_flow: gain converting measured yaw rate (rad/s) into
            the flow units the rotation channel uses.
        """
        if not 0.0 < split_fraction < 1.0:
            raise ValueError(f"split_fraction must be in (0,1), got {split_fraction}")
        if not 0 <= horizon_row_start < rows:
            raise ValueError(f"horizon_row_start out of range: {horizon_row_start}")

        self._rows = rows
        self._cols = cols
        self._split_fraction = split_fraction
        self._horizon_row_start = horizon_row_start
        self._yaw_rate_to_flow = yaw_rate_to_flow

        split = int(cols * split_fraction)
        self._left_slice = slice(0, split)
        self._right_slice = slice(split, cols)

    def pool(self, flow_field: FlowField, yaw_rate: float) -> PooledChannels:
        """Collapse the flow field, cancelling rotation first.

        Args:
            flow_field: dense local flow from the EMD layer.
            yaw_rate: measured yaw rate in rad/s, from the IMU/gyro.

        Returns:
            PooledChannels, including the pre-subtraction rotation value so
            the correction is auditable.
        """
        if not flow_field.valid:
            return PooledChannels(0.0, 0.0, 0.0, 0.0, 0.0)

        horizontal = flow_field.flow[:, :, 0]
        vertical = flow_field.flow[:, :, 1]

        # The rotation channel is the mean horizontal flow across the whole
        # field: a pure yaw moves every point the same way, so the mean is
        # the rotational component and structure averages out.
        raw_rotation = float(np.mean(horizontal))

        # Predicted rotational flow from the IMU. This is the haltere
        # equivalent — an independent estimate of self-motion that does not
        # come from vision, so it is not fooled by what vision is looking at.
        predicted = yaw_rate * self._yaw_rate_to_flow

        # SUBTRACT FIRST. Everything downstream sees derotated flow.
        derotated = horizontal - predicted
        residual_rotation = raw_rotation - predicted

        band = slice(self._horizon_row_start, self._rows)
        left = derotated[band, self._left_slice]
        right = derotated[band, self._right_slice]

        # Outward flow means leftward on the left half and rightward on the
        # right half, so the left sign is inverted to make both channels
        # positive-is-closer.
        left_expansion = float(np.mean(-left))
        right_expansion = float(np.mean(right))

        vertical_expansion = float(np.mean(np.abs(vertical[band, :])))

        return PooledChannels(
            left_expansion=left_expansion,
            right_expansion=right_expansion,
            vertical_expansion=vertical_expansion,
            rotation=residual_rotation,
            raw_rotation_before_subtraction=raw_rotation,
        )

    @classmethod
    def from_config(cls, cfg) -> "WideFieldPooling":
        """Build from the loaded configuration."""
        return cls(
            rows=cfg.retina.rows,
            cols=cfg.retina.cols,
            split_fraction=cfg.pooling.split_fraction,
            horizon_row_start=cfg.pooling.horizon_row_start,
            yaw_rate_to_flow=cfg.pooling.yaw_rate_to_flow,
        )
