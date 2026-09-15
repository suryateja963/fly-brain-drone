"""Main Webots controller loop.

Signal path, camera to motors, every timestep:

    retina -> emd -> lobula -> (avoidance | central_complex) -> arbitration
           -> output -> motors

Phase 0 is the toolchain check: print the camera frame shape and the current
yaw every timestep. Do not write a single line of neural code until that
passes.

Phase 1 is waypoint flight on ground-truth position, no vision. Do not skip
it — if flight is unstable later, you need to know whether the bug is in the
fly brain or the flight control.
"""

TARGET_ALTITUDE = 1.0
GOAL_HEADING = 0.0


def main():
    """Set up devices, then run the control loop until the sim stops.

    Phase 0: enable the camera and IMU, print frame shape and yaw.
    Phase 1: add position, bearing to a hardcoded target, proportional
             control on yaw and pitch, hold altitude.
    Phase 2+: replace the Phase 1 steering with the fly brain, one module at a
             time, keeping each phase's exit test passing.
    """
    raise NotImplementedError("Phase 0")


if __name__ == "__main__":
    main()
