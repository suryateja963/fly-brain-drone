"""Standalone flow viewer — the Phase 2 exit test, and the money shot.

A live side-by-side window: camera view on the left, optic flow field as
arrows or a heatmap on the right. Fly forward and you should see outward
expansion; yaw left and you should see uniform rightward flow.

The same view, upscaled, is the best footage in the demo video: photoreal 3D
on one side, a 40x20 motion field on the other, both driving the same flight.

Phase 2.
"""


def render(frame, horizontal, vertical, scale=8):
    """Compose the side-by-side visualisation.

    Args:
        frame: full-res camera frame.
        horizontal: local horizontal flow.
        vertical: local vertical flow.
        scale: upscale factor for the flow panel, for legibility.

    Returns:
        A single BGR image ready for cv2.imshow.
    """
    raise NotImplementedError("Phase 2")


def main():
    """Open a window and stream the visualisation until the user quits."""
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    main()
