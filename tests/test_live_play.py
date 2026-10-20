"""Pixel-only, 10 FPS live control against the virtual game's real contact rules."""

import importlib.util
import math
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

from air_hockey import core_config as core
from air_hockey.app.rally import READY_POSITION, RallyController, RallyState
from air_hockey.app.vision_runtime import VisionRuntime
from air_hockey.camera.types import Frame
from air_hockey.control import PlcControlAdapter
from air_hockey.control import plc as protocol
from air_hockey.control.plc import PLCFeedback, PLCLink
from air_hockey.physics import StoneMotion, goal_scorer
from air_hockey.app.live import submit_rally_target


def _contact_game(stone):
    directory = Path(__file__).resolve().parents[1] / "sim_gui"
    spec = importlib.util.spec_from_file_location("_live_contact_game", directory / "game.py")
    module = importlib.util.module_from_spec(spec)
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    game = module.AirHockeyGame.__new__(module.AirHockeyGame)
    game.stone = stone
    return game


class _LiveTable:
    def __init__(self, stone, mallet=READY_POSITION, plc=None):
        self.runtime = VisionRuntime(
            table_roi=(0, 0, 1280, 720),
            table_calibration_file=None,
            disable_undistort=True,
        )
        self.plc = plc
        if plc is not None:
            self.runtime.ai.recovery_bounds = plc.reachable_bounds
        self.controller = RallyController(reachable_bounds=None if plc is None else plc.reachable_bounds)
        self.adapter = PlcControlAdapter()
        self.game = _contact_game(stone)
        self.mallet = mallet
        self.sequence = 0
        self.impacts = 0
        self.states = set()
        self.minimum_y = stone.y
        hsv = np.array([[[5, 157, 186]]], dtype=np.uint8)
        self.red = tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def observe(self):
        self.sequence += 1
        timestamp = 10.0 + (self.sequence - 1) * 0.1
        stone = self.game.stone
        geometry = self.runtime.camera_geometry
        pixel_x, pixel_y = geometry.table_to_raw(stone.x, stone.y)
        edge_x, _ = geometry.table_to_raw(stone.x + core.STONE_RADIUS, stone.y)
        _, edge_y = geometry.table_to_raw(stone.x, stone.y + core.STONE_RADIUS)
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        cv2.ellipse(
            image, (round(pixel_x), round(pixel_y)),
            (round(edge_x - pixel_x), round(edge_y - pixel_y)),
            0, 0, 360, self.red, -1,
        )
        feedback = PLCFeedback(
            stamp=timestamp, valid=True, x=self.mallet[0], y=self.mallet[1],
            x_en=True, y_en=True, axes_ready=True, plc_echo_ok=True,
        )
        with patch("air_hockey.app.vision_runtime.time.perf_counter", return_value=timestamp):
            self.runtime.set_ai_feedback(feedback)
            result = self.runtime.process_frame(Frame(image, timestamp=timestamp, sequence=self.sequence))
        result.rally = self.controller.update(result, now=timestamp, feedback=feedback)
        request = submit_rally_target(result, self.adapter, None)
        assert result.detection is not None, (self.sequence, stone)
        assert result.track_confirmed, (self.sequence, result.track)
        self.states.add(result.rally.state)
        if request is None:
            assert result.rally.target is None, (self.sequence, result.rally)
            return result, None
        limits = self.adapter._limits
        for x, y in (result.rally.target, (request.target_x, request.target_y)):
            assert math.isfinite(x) and math.isfinite(y)
            assert limits[0] <= x <= limits[1] and limits[2] <= y <= limits[3], (x, y)
        return result, request

    def advance(self, request):
        target = self.mallet if request is None else (request.target_x, request.target_y)
        if self.plc is not None and request is not None:
            target = self.plc.physical_to_game(*self.plc.game_to_physical(*target))
        steps = math.ceil(0.1 / core.MAX_PHYSICS_STEP)
        dt = 0.1 / steps
        stone = self.game.stone
        for _ in range(steps):
            previous_mallet = self.mallet
            x, y, vx, vy = self.game._move_towards(
                *self.mallet, *target,
                core.DIFFICULTIES["普通"].ai_speed, dt,
            )
            self.mallet = (x, y)
            assert math.hypot(x - previous_mallet[0], y - previous_mallet[1]) <= (
                core.DIFFICULTIES["普通"].ai_speed * dt + 1e-9
            )
            previous_stone = (stone.x, stone.y)
            stone.advance_position(dt)
            stone.resolve_walls()
            stone.resolve_goal_posts()
            previous_velocity = stone.collision_velocity()
            impact_vx = vx * core.AI_IMPACT_SPEED_SCALE
            impact_vy = vy * core.AI_IMPACT_SPEED_SCALE
            self.game._swept_collide_with_mallet(
                previous_stone, previous_mallet, x, y, impact_vx, impact_vy,
            )
            self.game._collide_with_mallet(x, y, impact_vx, impact_vy)
            self.impacts += stone.collision_velocity() != previous_velocity
            stone.resolve_walls()
            stone.resolve_goal_posts()
            assert goal_scorer(stone) != "player", (self.sequence, stone, self.mallet)
            stone.advance_velocity(dt)
            self.minimum_y = min(self.minimum_y, stone.y)

    def return_puck(self, first_observation=None, *, held_request=None):
        previous_impacts = self.impacts
        returned = False
        observation = first_observation
        if held_request is not None:
            self.advance(held_request)
        for _ in range(80):
            result, request = observation if observation is not None else self.observe()
            observation = None
            stone = self.game.stone
            returned |= stone.y > core.RINK_CENTER_Y + core.STONE_RADIUS and stone.vy > 0
            if (returned and result.curling_state.y > core.RINK_CENTER_Y
                    and request is not None and request.target_y == READY_POSITION[1]):
                assert self.impacts > previous_impacts, (self.sequence, stone)
                assert result.curling_state.y > core.RINK_CENTER_Y
                return result, request
            self.advance(request)
        raise AssertionError(
            ("Puck was not contacted, returned, and released to READY within 8 seconds",
             self.game.stone, self.mallet, self.impacts - previous_impacts, result.rally)
        )


def test_live_incoming_shots_contact_and_return_at_ten_fps():
    for x, vx in ((300.0, 0.0), (430.0, -70.0)):
        table = _LiveTable(StoneMotion(x=x, y=600.0, vx=vx, vy=-400.0))
        table.return_puck()
        assert RallyState.DEFENDING in table.states


def test_live_stationary_machine_half_pucks_are_struck_on_first_detection():
    for x, y in ((300.0, 340.0), (360.0, 370.0), (300.0, core.RINK_CENTER_Y),
                 (core.RINK_LEFT + core.STONE_RADIUS, 250.0),
                 (core.RINK_RIGHT - core.STONE_RADIUS, 250.0)):
        table = _LiveTable(StoneMotion(x=x, y=y))
        observation = table.observe()
        assert observation[0].rally.state == RallyState.ATTACKING
        assert observation[0].rally.target != READY_POSITION
        table.return_puck(observation)


def test_live_behind_mallet_pucks_are_retrieved_without_own_goal():
    for y, mallet_y in ((180.0, 250.0), (200.0, 320.0), (100.0, 250.0)):
        table = _LiveTable(StoneMotion(x=300.0, y=y), (300.0, mallet_y))
        observation = table.observe()
        assert observation[0].rally.state == RallyState.ATTACKING
        assert observation[0].rally.target != READY_POSITION
        table.return_puck(observation)
        assert table.minimum_y >= y - 1.0, (table.minimum_y, y)


def test_live_rear_pucks_return_with_measured_mechanical_travel():
    for x, y, mallet_x in ((300, 140, 300), (480, 180, 480), (70, 180, 150)):
        table = _LiveTable(StoneMotion(x=x, y=y), (mallet_x, 250), PLCLink("offline"))
        table.return_puck()
        assert table.minimum_y >= y - 1.0


def test_live_moving_rear_puck_is_bypassed_and_returned():
    table = _LiveTable(StoneMotion(x=300, y=180, vx=80, vy=-20), (300, 250), PLCLink("offline"))
    table.return_puck()


def test_live_rear_contact_uses_runtime_default_command_margin():
    original_margin = protocol.MOTION_MARGIN_MM
    try:
        protocol.set_motion_margin(40.0)
        table = _LiveTable(StoneMotion(x=300, y=140), (300, 250), PLCLink("offline"))
        for _ in range(30):
            _, request = table.observe()
            assert request is not None
            table.advance(request)
            if table.impacts:
                assert table.game.stone.vy > 0
                assert table.minimum_y >= 139.0
                break
        else:
            raise AssertionError(("Reachable rear puck was never contacted", table.mallet))
    finally:
        protocol.set_motion_margin(original_margin)


def test_live_unreachable_goal_side_puck_is_not_pushed_into_own_goal():
    link = PLCLink("offline")
    y = link.reachable_bounds[2] - core.STONE_RADIUS
    table = _LiveTable(StoneMotion(x=300, y=y), (300, 250), link)
    for _ in range(30):
        _, request = table.observe()
        assert request is None
        table.advance(request)
    assert table.impacts == 0
    assert table.minimum_y == y
    assert math.dist(table.mallet, (300, y)) > core.MALLET_RADIUS + core.STONE_RADIUS


def test_live_repeated_human_returns_need_no_rearm_or_ready_arrival():
    table = _LiveTable(StoneMotion(x=360.0, y=560.0, vy=-350.0))
    observation = table.return_puck()
    for _ in range(2):
        # The opponent changes only physical truth; vision must discover the new direction.
        table.game.stone.set_immediate_velocity(0.0, -350.0)
        observation = table.return_puck(held_request=observation[1])
    assert table.impacts >= 3
