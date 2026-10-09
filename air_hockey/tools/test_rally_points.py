"""Camera-free Stage A: manually request READY, LEFT, CENTER, RIGHT, READY.

Offline: python3 air_hockey/tools/test_rally_points.py --dry-run
Live:    python3 air_hockey/tools/test_rally_points.py --plc 192.168.0.64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from air_hockey import core_config as core
from air_hockey.ai import AIDecision
from air_hockey.app.rally import ARRIVAL_TOLERANCE, READY_POSITION
from air_hockey.control import PLCLink, PlcControlAdapter
from air_hockey.control.plc import DEFAULT_PERIOD, TARGET_MAX_AGE


COMMANDS = "e enable | h home | c reset | n next point | q quit (then Enter)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Camera-free Stage A PLC point test: READY -> LEFT -> CENTER -> RIGHT -> READY. "
            "Live motion requires explicit operator commands; there is no automatic enable."
        ),
        epilog=(
            f"Live commands: {COMMANDS}. Home/reset cancel the pending point and disarm; "
            "enable again and request n to retry. Keep the workspace clear and hardware "
            "emergency stop accessible. Example: %(prog)s --plc 192.168.0.64"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plc", metavar="IP", help="PLC address for explicit live field testing")
    mode.add_argument(
        "--dry-run", action="store_true",
        help="print validated table/physical point mappings and exit; no network or motion",
    )
    return parser


def build_points():
    adapter = PlcControlAdapter()
    center_x, ready_y = READY_POSITION
    quarter_width = (core.RINK_RIGHT - core.RINK_LEFT) / 4.0
    return tuple(
        (name, adapter.build_target(AIDecision(x, ready_y, "idle"), timestamp=0.0))
        for name, x in (
            ("READY", center_x), ("LEFT", center_x - quarter_width),
            ("CENTER", center_x), ("RIGHT", center_x + quarter_width), ("READY", center_x),
        )
    )


def format_pair(x, y):
    return f"({x:.2f}, {y:.2f})"


def print_points(link, points):
    print(f"Stage A: arrival tolerance {ARRIVAL_TOLERANCE:g} table units; no camera or tracking.")
    print("#  POINT   TABLE TARGET (units)      PHYSICAL TARGET (mm)")
    for index, (name, target) in enumerate(points, 1):
        physical = link.game_to_physical(target.target_x, target.target_y)
        print(
            f"{index}  {name:<7} {format_pair(target.target_x, target.target_y):<25} "
            f"{format_pair(*physical)}"
        )


class PointTest:
    """One explicit n per point; a revoked request is never replayed."""

    def __init__(self, link, points):
        self.link = link
        self.points = points
        self.next_index = 0
        self.pending = None
        self.requested_at = 0.0
        self.exit_requested = False

    def safety_failure(self, feedback, now):
        if not self.link.connected:
            return "PLC disconnected"
        if not feedback.valid:
            return "no actual PLC feedback"
        age = now - feedback.stamp
        if not math.isfinite(age) or feedback.stamp <= 0 or not 0 <= age <= TARGET_MAX_AGE:
            return "PLC feedback stale or invalid"
        if not all(math.isfinite(value) for value in (feedback.x, feedback.y)):
            return "PLC position invalid"
        failures = (
            (feedback.axis_fault, "axis fault"),
            (feedback.comm_lost, "communication lost"),
            (feedback.comm_lost_latch, "communication-loss latch"),
            (feedback.group_stop, "group stop active"),
            (feedback.kinematics_error != 0, f"kinematics error {feedback.kinematics_error}"),
            (not feedback.plc_echo_ok, "PLC heartbeat not confirmed"),
            (not feedback.axes_ready, "axes not ready"),
            (not (feedback.x_en and feedback.y_en), "axes not enabled"),
        )
        for failed, reason in failures:
            if failed:
                return reason
        if not self.link.armed:
            return "not armed; explicit e required"
        return None

    def clear_pending(self):
        self.pending = None
        self.requested_at = 0.0
        self.link.clear_target()

    def revoke(self, reason):
        self.clear_pending()
        return f"SAFETY: {reason}; pending point cancelled. A new n is required; e if disarmed."

    def command(self, command, *, now=None):
        command = command.strip().lower()
        now = time.monotonic() if now is None else now
        if command == "q":
            self.clear_pending()
            self.exit_requested = True
            return "Exiting; clearing targets and stopping the PLC worker."
        if command in ("e", "h", "c"):
            self.clear_pending()
            action, label = {
                "e": (self.link.enable_axes, "Enable"),
                "h": (self.link.home_axes, "Home"),
                "c": (self.link.axes_reset, "Reset"),
            }[command]
            if not action():
                return f"SAFETY: {label} request rejected; pending point cleared. Check PLC messages."
            return (
                f"{label} queued, not yet confirmed; pending point cleared. "
                "Wait for healthy ARMED feedback (e after home/reset), then request n."
            )
        if command != "n":
            return f"Commands: {COMMANDS}"
        failure = self.safety_failure(self.link.feedback, now)
        if failure:
            return self.revoke(failure)
        if self.pending is not None:
            return f"Point {self.points[self.pending][0]} still pending; wait for actual arrival."
        if self.next_index == len(self.points):
            return "Stage A complete: all five points arrived. Use q to stop."
        self.pending = self.next_index
        self.requested_at = now
        name, target = self.points[self.pending]
        if not self.link.set_target(target.target_x, target.target_y):
            return self.revoke("target publication rejected")
        return f"Requested {self.pending + 1}/{len(self.points)} {name}; awaiting fresh, non-busy arrival."

    def poll(self, *, now=None):
        if self.pending is None:
            return None
        now = time.monotonic() if now is None else now
        feedback = self.link.feedback
        failure = self.safety_failure(feedback, now)
        if failure:
            return self.revoke(failure)
        name, target = self.points[self.pending]
        if (
            feedback.stamp > self.requested_at and not feedback.busy
            and math.hypot(feedback.x - target.target_x, feedback.y - target.target_y) <= ARRIVAL_TOLERANCE
        ):
            elapsed = now - self.requested_at
            self.next_index += 1
            self.clear_pending()
            suffix = "Stage A complete; q to stop." if self.next_index == len(self.points) else "n for the next point."
            return (f"ARRIVED {self.next_index}/{len(self.points)} {name} from actual PLC feedback; "
                    f"host request-to-arrival {elapsed:.3f}s; {suffix}")
        if not self.link.set_target(target.target_x, target.target_y):
            return self.revoke("target publication rejected")
        return None

    def status(self):
        feedback = self.link.feedback
        failure = self.safety_failure(feedback, time.monotonic())
        target_text = "target=none"
        if self.pending is not None:
            name, target = self.points[self.pending]
            target_text = (
                f"target={name} table={format_pair(target.target_x, target.target_y)} "
                f"physical_mm={format_pair(*self.link.game_to_physical(target.target_x, target.target_y))}"
            )
        actual = "actual=unavailable"
        if feedback.valid and all(math.isfinite(value) for value in (feedback.x, feedback.y)):
            actual = (
                f"actual_table={format_pair(feedback.x, feedback.y)} "
                f"actual_physical_mm={format_pair(*self.link.game_to_physical(feedback.x, feedback.y))}"
                if (core.RINK_LEFT <= feedback.x <= core.RINK_RIGHT
                    and core.RINK_TOP <= feedback.y <= core.RINK_CENTER_Y)
                else f"actual_table={format_pair(feedback.x, feedback.y)} (outside mapped PLC zone)"
            )
        safety = f"SAFETY: {failure}" if failure else "ARMED / healthy"
        return f"[{safety}; busy={feedback.busy}] {target_text}; {actual}"


def run_live(link, points):
    session = PointTest(link, points)
    print("LIVE FIELD TEST: keep the workspace clear; hardware emergency stop must be accessible.")
    print(f"No automatic enable or point motion. {COMMANDS}")
    commands = queue.Queue()

    def read_commands():
        try:
            for command in sys.stdin:
                commands.put(command)
        finally:
            commands.put("")

    last_status = 0.0
    try:
        link.start()
        threading.Thread(target=read_commands, name="point-test-console", daemon=True).start()
        while not session.exit_requested:
            for message in link.drain_messages():
                print(f"[PLC] {message}")
            message = session.poll()
            if message:
                print(message)
                last_status = 0.0
            if time.monotonic() - last_status >= 0.5:
                print(session.status(), flush=True)
                last_status = time.monotonic()
            try:
                command = commands.get(timeout=DEFAULT_PERIOD)
            except queue.Empty:
                continue
            if not command:
                break
            print(session.command(command), flush=True)
            last_status = 0.0
    except KeyboardInterrupt:
        print("Interrupted; clearing targets and stopping the PLC worker.")
    finally:
        session.clear_pending()
        if not link.stop():
            print("SAFETY: worker cleanup is still pending; inspect the machine and use hardware emergency stop if needed.")
            while not link.stop(timeout=1.0):
                pass
        for message in link.drain_messages():
            print(f"[PLC] {message}")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and (not args.plc or not args.plc.strip()):
        parser.error("live Stage A requires --plc IP; use --dry-run for offline mapping only")
    link = PLCLink(args.plc.strip() if args.plc else "dry-run")
    points = build_points()
    print_points(link, points)
    if args.dry_run:
        print("DRY RUN: mappings only; no network start, enable, home, reset, or motion calls.")
        return 0
    return run_live(link, points)


if __name__ == "__main__":
    raise SystemExit(main())
