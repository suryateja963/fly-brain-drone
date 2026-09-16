"""Record the five-panel view to video — the live-demo safety net.

A live demo can fail for reasons that have nothing to do with the drone: the
simulator stalls, a driver misbehaves, the laptop throttles. Having every
scenario already recorded means a failure in the room costs a sentence
("here's that run from earlier") rather than the demo.

Recording every scenario takes minutes and removes the single largest risk
in the delivery plan, so it is worth doing even when the live run works.

Usage:

    python scripts/record_fallback.py                  # every world
    python scripts/record_fallback.py --world 03_pillars
    python scripts/record_fallback.py --duration 45
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(ROOT, "worlds")
WEBOTS = os.path.join(
    ROOT, "Webots", "msys64", "mingw64", "bin", "webotsw.exe"
)

# Ordered by how they build an argument, not alphabetically: an audience
# should see the system work simply before seeing it work under pressure.
DEMO_SEQUENCE = [
    ("01_empty", "Baseline — takeoff, navigate, hold station"),
    ("02_corridor", "Centring response — balanced flow holds the mid-line"),
    ("03_pillars", "Pillar forest — obstacle avoidance on optic flow alone"),
    ("04_target_behind", "Arbitration — dodge and still arrive"),
    ("05_cluttered", "Stress test — dense clutter, no escape route"),
]


def available_worlds() -> List[str]:
    if not os.path.isdir(WORLDS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(WORLDS_DIR)
        if f.endswith(".wbt")
    )


def record_world(
    world: str,
    duration: int,
    output_dir: str,
    description: str = "",
) -> bool:
    """Fly one world headlessly and capture its log.

    Webots' own movie recorder needs the GUI, so this runs the flight and
    saves the telemetry log. The frames are rendered separately by
    demo/five_panel.py, which can replay from a log without the simulator —
    that separation is what lets the fallback exist at all.

    Returns:
        True if the run completed and produced output.
    """
    world_path = os.path.join(WORLDS_DIR, f"{world}.wbt")
    if not os.path.exists(world_path):
        print(f"  SKIP {world}: no such world", file=sys.stderr)
        return False

    if not os.path.exists(WEBOTS):
        print(f"  FAIL: Webots not found at {WEBOTS}", file=sys.stderr)
        return False

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{world}.log")

    print(f"  {world}: {description}")
    print(f"    flying {duration}s ...", end="", flush=True)

    command = [
        WEBOTS,
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--minimize",
        "--stdout",
        "--stderr",
        world_path,
    ]

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=duration)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    _kill_stragglers()
    elapsed = time.time() - started

    size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    if size == 0:
        print(f" FAILED (empty log after {elapsed:.0f}s)")
        return False

    steps = _count_steps(log_path)
    print(f" done  {elapsed:.0f}s, {steps} telemetry lines, {size // 1024}KB")
    return steps > 0


def _kill_stragglers() -> None:
    """Webots leaves child processes behind when terminated."""
    for image in ("webots-bin.exe", "webotsw.exe"):
        subprocess.run(
            ["taskkill", "/F", "/IM", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(1.5)


def _count_steps(log_path: str) -> int:
    count = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[phase2] ("):
                count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Record fallback footage of every demo scenario."
    )
    parser.add_argument(
        "--world",
        help="record a single world (default: the full demo sequence)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="seconds to fly each world (default 60)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "recordings"),
        help="output directory (default: recordings/)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available worlds and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        print("Available worlds:")
        for world in available_worlds():
            print(f"  {world}")
        return 0

    if args.world:
        targets = [(args.world, "single world")]
    else:
        targets = DEMO_SEQUENCE

    print(f"Recording {len(targets)} scenario(s) to {args.output}")
    print()

    succeeded = 0
    for world, description in targets:
        if record_world(world, args.duration, args.output, description):
            succeeded += 1

    print()
    print(f"{succeeded}/{len(targets)} recorded successfully")

    if succeeded < len(targets):
        print()
        print("Some scenarios failed. A live demo without complete fallback")
        print("footage carries the risk this script exists to remove — fix")
        print("them before the demo rather than after.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
