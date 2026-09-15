"""Main Webots controller loop.

Signal path, camera to motors, every timestep:

    retina -> emd -> lobula -> (avoidance | central_complex) -> arbitration
           -> output -> motors

PHASE 0 (current): the toolchain check. Enable the camera and IMU, and print
the camera frame shape and current yaw every timestep. Nothing else. Do not
write a single line of neural code until this passes.

PHASE 1 (next): waypoint flight on ground-truth position, no vision. Do not
skip it — if flight is unstable later, you need to know whether the bug is in
the fly brain or the flight control.
"""

from controller import Robot

TARGET_ALTITUDE = 1.0
GOAL_HEADING = 0.0

# Print every Nth timestep. At a 32ms step the sim runs ~31Hz, and printing
# every frame floods the console faster than you can read it.
PRINT_EVERY = 16


def main():
    """Phase 0: enable sensors, print frame shape and yaw every timestep."""
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # The Mavic 2 Pro carries an InertialUnit for orientation and a Gyro for
    # rates. Phase 0 needs orientation only; the gyro's yaw rate matters from
    # Phase 3, where it cancels rotation out of the expansion estimate.
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    print(f"[phase0] timestep={timestep}ms", flush=True)
    print(f"[phase0] camera {camera.getWidth()}x{camera.getHeight()}", flush=True)

    frame_count = 0
    while robot.step(timestep) != -1:
        frame_count += 1

        # getImage returns raw BGRA bytes, height * width * 4. Phase 2's retina
        # will reshape this into an array; here we only prove it arrives.
        image = camera.getImage()
        if image is None:
            continue

        height = camera.getHeight()
        width = camera.getWidth()
        shape = (height, width, 4)

        # InertialUnit returns roll, pitch, yaw in radians.
        roll, pitch, yaw = imu.getRollPitchYaw()
        yaw_rate = gyro.getValues()[2]

        if frame_count % PRINT_EVERY == 0:
            print(
                f"[phase0] frame={frame_count:6d} "
                f"shape={shape} bytes={len(image)} "
                f"yaw={yaw:+.3f}rad yaw_rate={yaw_rate:+.3f}rad/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
