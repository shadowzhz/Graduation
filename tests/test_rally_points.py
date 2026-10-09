"""Stage A operator requests, feedback arrival, and offline safety without hardware."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import math
from unittest.mock import patch

from air_hockey import core_config as core
from air_hockey.app.rally import ARRIVAL_TOLERANCE, READY_POSITION
from air_hockey.control import PLCFeedback, PLCLink
from air_hockey.control.plc import (
    PHYSICAL_X_MIN, PHYSICAL_X_MAX, PHYSICAL_Y_MIN, PHYSICAL_Y_MAX,
    PLCInterface, TARGET_MAX_AGE,
)
from air_hockey.tools import test_rally_points as points_tool


class FakeLink:
    def __init__(self):
        self.connected = True
        self.armed = True
        self.feedback = PLCFeedback(
            stamp=100.0, valid=True, x=READY_POSITION[0], y=READY_POSITION[1],
            x_en=True, y_en=True, axes_ready=True, plc_echo_ok=True,
        )
        self.targets = []
        self.clears = 0
        self.commands = []
        self.accept_target = True

    def set_target(self, x, y):
        self.targets.append((x, y))
        return self.accept_target

    def clear_target(self):
        self.clears += 1

    def enable_axes(self):
        self.commands.append("e")
        return self.connected

    def home_axes(self):
        self.commands.append("h")
        self.armed = False
        return self.connected

    def axes_reset(self):
        self.commands.append("c")
        self.armed = False
        return self.connected


def session():
    link = FakeLink()
    return link, points_tool.PointTest(link, points_tool.build_points())


def test_n_requires_explicit_enable_and_never_queues_for_later():
    link, test = session()
    link.armed = False
    test.command("n", now=100.0)
    assert test.pending is None and not link.targets and not link.commands
    test.poll(now=100.0)
    assert not link.commands
    test.command("e", now=100.0)
    assert link.commands == ["e"]
    link.armed = True
    assert test.poll(now=100.0) is None
    assert not link.targets
    test.command("n", now=100.0)
    assert test.pending == 0 and link.targets == [READY_POSITION]


def test_arrival_requires_post_request_feedback_non_busy_and_tolerance():
    link, test = session()
    test.command("n", now=100.0)
    assert test.poll(now=100.01) is None  # Cached pre-request position cannot finish READY.
    assert test.next_index == 0
    link.feedback = replace(link.feedback, stamp=100.02, busy=True)
    assert test.poll(now=100.02) is None
    link.feedback = replace(
        link.feedback, stamp=100.03, busy=False,
        x=READY_POSITION[0] + ARRIVAL_TOLERANCE + 0.01,
    )
    assert test.poll(now=100.03) is None
    link.feedback = replace(link.feedback, stamp=100.04, x=READY_POSITION[0] + ARRIVAL_TOLERANCE)
    assert test.poll(now=100.04) is not None
    assert test.pending is None and test.next_index == 1
    previous_targets = list(link.targets)
    assert test.poll(now=100.05) is None
    assert link.targets == previous_targets  # Arrival does not automatically start LEFT.
    test.command("n", now=100.05)
    assert test.pending == 1
    assert link.targets[-1] == (
        core.RINK_CENTER_X - (core.RINK_RIGHT - core.RINK_LEFT) / 4.0, READY_POSITION[1],
    )
    test.command("n", now=100.05)
    assert test.pending == 1 and test.next_index == 1


def test_operator_actions_revoke_pending_without_advancing():
    for command in ("e", "h", "c", "q"):
        link, test = session()
        test.command("n", now=100.0)
        before = len(link.targets)
        test.command(command, now=100.01)
        assert test.pending is None and test.next_index == 0 and link.clears > 0
        assert link.commands == ([] if command == "q" else [command])
        assert test.exit_requested == (command == "q")
        link.armed = True
        link.feedback = replace(link.feedback, stamp=100.02)
        assert test.poll(now=100.02) is None
        assert len(link.targets) == before


def test_feedback_failures_cancel_pending_and_recovery_requires_new_n():
    for changes in (
        {"valid": False}, {"stamp": 99.0}, {"stamp": 101.0}, {"x": float("nan")},
        {"x_err": True}, {"comm_lost": True}, {"comm_lost_latch": True},
        {"group_stop": True}, {"kinematics_error": 801}, {"plc_echo_ok": False},
        {"axes_ready": False}, {"y_en": False},
    ):
        link, test = session()
        test.command("n", now=100.0)
        before = len(link.targets)
        good = link.feedback
        link.feedback = replace(good, **changes)
        assert test.poll(now=100.01) is not None
        assert test.pending is None and test.next_index == 0 and link.clears > 0
        assert len(link.targets) == before
        link.feedback = replace(good, stamp=100.02)
        assert test.poll(now=100.02) is None
        assert len(link.targets) == before
        test.command("n", now=100.02)
        assert test.pending == 0


def test_link_failure_revokes_request_without_resuming():
    for failure in ("disarmed", "disconnected", "publication rejected"):
        link, test = session()
        test.command("n", now=100.0)
        if failure == "disarmed":
            link.armed = False
        elif failure == "disconnected":
            link.connected = False
        else:
            link.accept_target = False
        assert test.poll(now=100.01) is not None
        assert test.pending is None and test.next_index == 0 and link.clears > 0
        before = len(link.targets)
        link.armed = link.connected = link.accept_target = True
        link.feedback = replace(link.feedback, stamp=100.02)
        assert test.poll(now=100.02) is None
        assert len(link.targets) == before


def test_fresh_pending_target_is_refreshed_only_until_stale():
    link, test = session()
    link.feedback = replace(link.feedback, x=READY_POSITION[0] + ARRIVAL_TOLERANCE + 1)
    test.command("n", now=100.0)
    test.poll(now=100.1)
    assert len(link.targets) == 2
    assert test.poll(now=100.0 + TARGET_MAX_AGE + 0.01) is not None
    assert len(link.targets) == 2 and test.pending is None


def test_all_five_points_need_separate_operator_requests():
    link, test = session()
    now = 100.0
    for index, (_name, target) in enumerate(test.points):
        test.command("n", now=now)
        assert test.pending == index
        now += 0.01
        link.feedback = replace(
            link.feedback, stamp=now, x=target.target_x, y=target.target_y,
        )
        assert test.poll(now=now) is not None
        assert test.pending is None and test.next_index == index + 1
    test.command("n", now=now)
    assert test.pending is None and test.next_index == len(test.points)
    assert len(link.targets) == 5 and not link.commands


def test_point_sequence_uses_shared_ready_and_reachable_physical_mapping():
    points = points_tool.build_points()
    assert [name for name, _target in points] == ["READY", "LEFT", "CENTER", "RIGHT", "READY"]
    quarter_width = (core.RINK_RIGHT - core.RINK_LEFT) / 4.0
    assert [(target.target_x, target.target_y) for _name, target in points] == [
        READY_POSITION,
        (READY_POSITION[0] - quarter_width, READY_POSITION[1]),
        READY_POSITION,
        (READY_POSITION[0] + quarter_width, READY_POSITION[1]),
        READY_POSITION,
    ]
    link = PLCLink("offline")
    for _name, target in points:
        assert core.RINK_LEFT + core.MALLET_RADIUS <= target.target_x <= core.RINK_RIGHT - core.MALLET_RADIUS
        assert core.RINK_TOP + core.MALLET_RADIUS <= target.target_y <= core.RINK_CENTER_Y - core.MALLET_RADIUS
        px, py = link.game_to_physical(target.target_x, target.target_y)
        assert PHYSICAL_X_MIN <= px <= PHYSICAL_X_MAX
        assert PHYSICAL_Y_MIN <= py <= PHYSICAL_Y_MAX
        assert math.isclose(
            (px - PHYSICAL_X_MIN) / (PHYSICAL_X_MAX - PHYSICAL_X_MIN),
            (target.target_x - core.RINK_LEFT) / (core.RINK_RIGHT - core.RINK_LEFT),
        )
        assert math.isclose(
            (py - PHYSICAL_Y_MIN) / (PHYSICAL_Y_MAX - PHYSICAL_Y_MIN),
            (target.target_y - core.RINK_TOP) / (core.RINK_CENTER_Y - core.RINK_TOP),
        )


def test_dry_run_uses_real_validated_mapping_without_network_or_commands():
    forbidden = (
        "start", "stop", "set_target", "clear_target", "enable_axes", "home_axes", "axes_reset",
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("dry-run invoked live PLC API")

    output = io.StringIO()
    with redirect_stdout(output), patch.multiple(PLCLink, **{
        name: unexpected_call for name in forbidden
    }), patch.object(PLCInterface, "connect", unexpected_call), patch.object(
        PLCInterface, "send_linear_move", unexpected_call,
    ):
        assert points_tool.main(["--dry-run"]) == 0
    text = output.getvalue()
    for name, _target in points_tool.build_points():
        assert name in text


def test_missing_live_address_explains_offline_alternative():
    error = io.StringIO()
    with redirect_stderr(error):
        try:
            points_tool.main([])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("live mode accepted a missing PLC address")
    assert "--plc" in error.getvalue() and "--dry-run" in error.getvalue()


def test_console_eof_and_quit_stop_without_automatic_enable_or_motion():
    class ConsoleLink(FakeLink):
        def __init__(self):
            super().__init__()
            self.feedback = PLCFeedback()
            self.armed = False
            self.started = self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True
            return True

        def drain_messages(self):
            return []

    for input_text, expected_commands in (("", []), ("n\nh\nc\ne\nq\n", ["h", "c", "e"])):
        link = ConsoleLink()
        with patch.object(points_tool.sys, "stdin", io.StringIO(input_text)), redirect_stdout(io.StringIO()):
            assert points_tool.run_live(link, points_tool.build_points()) == 0
        assert link.started and link.stopped and link.clears > 0
        assert link.commands == expected_commands and not link.targets
