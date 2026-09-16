"""Failure injection — trigger any failure mid-flight.

Two ways in, because a live demo and an automated test need different things:

  IN-PROCESS. Import FailureInjector, hand it a pipeline, call a method. Used
  by tests and by scripted demo sequences.

  FILE-BASED. Write a JSON flag file; the controller polls it each step. This
  is how a failure gets triggered during a LIVE demo, where the controller is
  already running inside Webots and there is no way to call into it. Chosen
  over a socket because it needs no port, no daemon, and survives the
  controller restarting between worlds.

Usage during a demo:

    python scripts/inject_failures.py motor 1
    python scripts/inject_failures.py battery 8
    python scripts/inject_failures.py vibration
    python scripts/inject_failures.py dropzone
    python scripts/inject_failures.py clear
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Dict, Optional

# Shared location, so the injector and the controller agree without either
# needing to know where the other was launched from.
FLAG_PATH = os.path.join(tempfile.gettempdir(), "fly_brain_failures.json")


class FailureInjector:
    """Applies failures to a running pipeline."""

    __slots__ = ("_pipeline", "_applied", "_flag_path", "_last_mtime")

    def __init__(self, pipeline, flag_path: str = FLAG_PATH) -> None:
        self._pipeline = pipeline
        self._applied: Dict[str, object] = {}
        self._flag_path = flag_path
        self._last_mtime: Optional[float] = None

    # -----------------------------------------------------------------
    # Direct injection
    # -----------------------------------------------------------------

    def kill_motor(self, rotor_index: int) -> None:
        """Kill a rotor. 0=FL, 1=FR, 2=RL, 3=RR."""
        self._pipeline.motor_handler.force_failure(rotor_index)
        self._applied[f"motor_{rotor_index}"] = True

    def set_battery(self, charge_pct: float) -> None:
        """Force the battery to a charge level."""
        self._pipeline.battery.force_level(charge_pct)
        self._applied["battery"] = charge_pct

    def trigger_vibration(self, active: bool = True) -> None:
        """Assert a cracked-propeller anomaly."""
        self._pipeline.vibration_monitor.force_anomaly(active)
        self._applied["vibration"] = active

    def occupy_drop_zone(self, occupied: bool = True) -> None:
        """Put something in the drop zone so release is refused."""
        self._pipeline.payload.force_zone_occupied(occupied)
        self._applied["dropzone"] = occupied

    def clear_all(self) -> None:
        """Reset every injected failure."""
        self._pipeline.motor_handler.reset()
        self._pipeline.vibration_monitor.reset()
        self._pipeline.payload.reset()
        self._pipeline.battery.reset(100.0)
        self._applied.clear()

    @property
    def applied(self) -> Dict[str, object]:
        return dict(self._applied)

    # -----------------------------------------------------------------
    # File-based polling, for live demos
    # -----------------------------------------------------------------

    def poll(self) -> bool:
        """Check the flag file and apply anything new.

        Cheap enough to call every control step: it stats one file and
        returns immediately unless the mtime changed.

        Returns:
            True if a failure was applied this call.
        """
        try:
            mtime = os.path.getmtime(self._flag_path)
        except OSError:
            return False

        if self._last_mtime is not None and mtime <= self._last_mtime:
            return False
        self._last_mtime = mtime

        try:
            with open(self._flag_path, "r", encoding="utf-8") as handle:
                flags = json.load(handle)
        except (OSError, ValueError):
            # A half-written file is normal: the writer may be mid-write.
            # Returning False retries on the next poll rather than crashing
            # the flight.
            return False

        return self._apply(flags)

    def _apply(self, flags: Dict) -> bool:
        applied = False

        if flags.get("clear"):
            self.clear_all()
            return True

        motor = flags.get("motor")
        if motor is not None:
            self.kill_motor(int(motor))
            applied = True

        battery = flags.get("battery")
        if battery is not None:
            self.set_battery(float(battery))
            applied = True

        if flags.get("vibration"):
            self.trigger_vibration(True)
            applied = True

        if flags.get("dropzone"):
            self.occupy_drop_zone(True)
            applied = True

        return applied


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def write_flags(flags: Dict, flag_path: str = FLAG_PATH) -> None:
    """Write the flag file atomically.

    Written to a temporary file and renamed, so a controller polling mid-write
    never sees a truncated JSON document.
    """
    directory = os.path.dirname(flag_path) or "."
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(flags, stream)
        os.replace(temporary, flag_path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject a failure into a running fly-brain drone."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    motor = sub.add_parser("motor", help="kill a rotor")
    motor.add_argument("index", type=int, choices=[0, 1, 2, 3],
                       help="0=front-left 1=front-right 2=rear-left 3=rear-right")

    battery = sub.add_parser("battery", help="force a charge level")
    battery.add_argument("percent", type=float, help="0-100")

    sub.add_parser("vibration", help="assert a cracked propeller")
    sub.add_parser("dropzone", help="occupy the drop zone")
    sub.add_parser("clear", help="reset every injected failure")
    sub.add_parser("status", help="show the current flag file")

    args = parser.parse_args(argv)

    if args.command == "status":
        if not os.path.exists(FLAG_PATH):
            print("no flag file — nothing injected")
            return 0
        with open(FLAG_PATH, "r", encoding="utf-8") as handle:
            print(json.dumps(json.load(handle), indent=2))
        return 0

    flags: Dict[str, object] = {"timestamp": time.time()}

    if args.command == "motor":
        flags["motor"] = args.index
        message = f"killed rotor {args.index}"
    elif args.command == "battery":
        if not 0.0 <= args.percent <= 100.0:
            print("battery percent must be 0-100", file=sys.stderr)
            return 2
        flags["battery"] = args.percent
        message = f"battery forced to {args.percent}%"
    elif args.command == "vibration":
        flags["vibration"] = True
        message = "vibration anomaly asserted"
    elif args.command == "dropzone":
        flags["dropzone"] = True
        message = "drop zone occupied"
    else:
        flags = {"clear": True, "timestamp": time.time()}
        message = "all failures cleared"

    write_flags(flags)
    print(f"{message}  ->  {FLAG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
