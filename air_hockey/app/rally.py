"""单回合防守控制；只决定目标，不授权 PLC 运动。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .. import core_config as core
from ..control.plc import TARGET_MAX_AGE
from .vision_runtime import AI_HOME_Y

if TYPE_CHECKING:
    from .vision_runtime import VisionResult

READY_POSITION = (core.RINK_CENTER_X, AI_HOME_Y)
ARRIVAL_TOLERANCE = core.STONE_RADIUS
MIN_DIRECTION_SPEED = 25.0  # 与现有 AI 的 goal-threat 判断一致。
RETURN_HITS = 2
RETURN_WINDOW = 2 * core.MAX_FRAME_GAP
RALLY_TIMEOUT = 5.0


class RallyState(Enum):
    WAITING = "WAITING"
    DEFENDING = "DEFENDING"
    RETURNED = "RETURNED"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class RallyDecision:
    state: RallyState
    target: tuple[float, float] | None
    reason: str
    direction: str
    mallet_position: tuple[float, float]
    mallet_source: str
    intercept: tuple[float, float] | None = None
    time_to_intercept: float | None = None
    time_to_mallet: float | None = None
    reachable: bool = False
    transitions: tuple[str, ...] = ()


class RallyController:
    def __init__(self, *, mallet_speed=core.DIFFICULTIES["普通"].ai_speed, hardware=False):
        if not math.isfinite(mallet_speed) or mallet_speed <= 0:
            raise ValueError("mallet_speed must be finite and positive (table units/s)")
        self.mallet_speed = mallet_speed
        self.hardware = hardware
        self.state = RallyState.WAITING
        self.counters = dict(incoming_detected=0, defense_entered=0, intercept_attempted=0,
                             successful_return=0, failed_intercept=0, recovered=0)
        self._mallet = READY_POSITION
        self._target = READY_POSITION
        self._last_update = None
        self._identity = None
        self._rearmed = True
        self._entered = 0.0
        self._last_hard = -math.inf
        self._last_observation = None
        self._contact_at = -math.inf
        self._return_hits = 0
        self._return_started = 0.0
        self._return_y = 0.0
        self._attempted = False
        self._intercept_target = None

    def update(self, result: VisionResult, *, now: float, feedback=None) -> RallyDecision:
        transitions = []
        dt = 0.0 if self._last_update is None else max(0.0, min(now - self._last_update, core.MAX_FRAME_GAP))
        self._last_update = now
        feedback_valid = (feedback is not None and feedback.valid
                          and 0 <= now - feedback.stamp <= TARGET_MAX_AGE
                          and all(math.isfinite(v) for v in (feedback.x, feedback.y)))
        if feedback_valid:
            self._mallet = (feedback.x, feedback.y)
        elif not self.hardware and self._target is not None:
            dx, dy = self._target[0] - self._mallet[0], self._target[1] - self._mallet[1]
            distance = math.hypot(dx, dy)
            if distance:
                scale = min(1.0, self.mallet_speed * dt / distance)
                self._mallet = self._mallet[0] + dx * scale, self._mallet[1] + dy * scale
        source = "PLC" if feedback_valid else "software"
        ready = (math.dist(self._mallet, READY_POSITION) <= ARRIVAL_TOLERANCE
                 and (not self.hardware or (feedback_valid and not feedback.busy)))
        stone = result.curling_state
        fresh = (math.isfinite(now) and math.isfinite(result.frame.timestamp)
                 and 0 <= now - result.frame.timestamp <= core.MAX_FRAME_GAP)
        valid = (fresh and result.track_confirmed and result.track is not None and stone is not None
                 and all(math.isfinite(v) for v in (stone.x, stone.y, stone.vx, stone.vy, stone.timestamp))
                 and 0 <= now - stone.timestamp <= core.MAX_FRAME_GAP
                 and core.RINK_LEFT <= stone.x <= core.RINK_RIGHT
                 and core.RINK_TOP <= stone.y <= core.RINK_BOTTOM)
        direction = "invalid"
        if valid:
            direction = ("incoming" if stone.vy < -MIN_DIRECTION_SPEED else
                         "outgoing" if stone.vy > MIN_DIRECTION_SPEED else "still")
        hard = (valid and result.detection is not None
                and result.detection.timestamp > self._last_hard)
        if hard:
            self._last_hard = result.detection.timestamp
        identity = result.track.track_id if result.track is not None else None
        if not result.track_confirmed or (valid and hard and stone.y >= core.RINK_CENTER_Y):
            self._rearmed = True
        if valid and identity != self._identity:
            self._rearmed = True

        def transition(state, reason):
            transitions.append(f"{self.state.value} -> {state.value}: {reason}")
            self.state = state
            self._entered = now

        def finish_failure(reason):
            self.counters["failed_intercept"] += 1
            self._rearmed = False
            transition(RallyState.RECOVERING, reason)

        target = READY_POSITION
        reason = "waiting for confirmed incoming puck"
        intercept = t_stone = t_mallet = None
        reachable = False
        if self.state == RallyState.RETURNED:
            transition(RallyState.RECOVERING, "puck returned; stop chasing")
            reason = "returning to READY"
        elif self.state == RallyState.RECOVERING:
            if ready:
                self.counters["recovered"] += 1
                transition(RallyState.WAITING, "mallet ready")
            reason = ("waiting for next rally" if self.state == RallyState.WAITING else
                      "recovery failure: check feedback/motion" if now - self._entered > RALLY_TIMEOUT else
                      "returning to READY")
        else:
            if self.state == RallyState.WAITING:
                if not valid:
                    reason = ("detection failure / no track" if result.track is None else
                              "tracker not confirmed" if not result.track_confirmed else "invalid/stale puck state")
                elif direction == "still":
                    reason = "low speed / no clear incoming direction"
                elif direction != "incoming":
                    reason = "outgoing puck; do not chase"
                elif not ready:
                    reason = "recovery required: mallet not READY"
                elif not hard or not self._rearmed:
                    reason = "waiting for rearm / real incoming observation"
                elif stone.y <= AI_HOME_Y + core.STONE_RADIUS + core.MALLET_RADIUS:
                    reason = "incoming puck already behind interception line"
                else:
                    self._identity = identity
                    self._rearmed = False
                    self._contact_at = -math.inf
                    self._last_observation = None
                    self._return_hits = 0
                    self._attempted = False
                    self._intercept_target = None
                    self.counters["incoming_detected"] += 1
                    self.counters["defense_entered"] += 1
                    transition(RallyState.DEFENDING, "confirmed incoming puck")
            if self.state == RallyState.DEFENDING:
                if not valid or identity != self._identity:
                    finish_failure("track lost / stale / identity changed")
                    target = None
                    reason = "tracker not confirmed / stale; target cleared"
                else:
                    if hard:
                        position = (stone.x, stone.y)
                        previous = self._last_observation
                        # 实际球槌附近的观测/观测线段，而非球台反弹事件。
                        if previous is not None and now - previous[1] <= core.MAX_FRAME_GAP:
                            near = self._segment_distance(previous[0], position, self._mallet)
                        else:
                            near = math.dist(position, self._mallet)
                        if near <= core.STONE_RADIUS + core.MALLET_RADIUS + core.STONE_RADIUS:
                            self._contact_at = now
                        self._last_observation = (position, now)
                        if direction == "outgoing" and now - self._contact_at <= RETURN_WINDOW:
                            if self._return_hits == 0:
                                self._return_started, self._return_y = now, stone.y
                            self._return_hits += 1
                        else:
                            self._return_hits = 0
                        if (self._return_hits >= RETURN_HITS
                                and now - self._return_started >= core.MAX_FRAME_TIME
                                and stone.y - self._return_y >= core.STONE_RADIUS * 0.5):
                            self.counters["successful_return"] += 1
                            self._rearmed = False
                            transition(RallyState.RETURNED, "sustained observed return near mallet")
                    if self.state == RallyState.RETURNED:
                        reason = "puck returned; READY requested"
                    elif now - self._entered > RALLY_TIMEOUT:
                        finish_failure("return detection failure / rally timeout")
                        reason = "return detection failure / missed interception"
                    elif stone.y < AI_HOME_Y - core.STONE_RADIUS:
                        finish_failure("puck passed defense line")
                        reason = "failed intercept; return to READY"
                    elif direction != "incoming":
                        target = (self._intercept_target if self._intercept_target is not None
                                  and math.dist(self._mallet, self._intercept_target) <= ARRIVAL_TOLERANCE else None)
                        reason = "return pending; hold arrived intercept without chasing"
                    else:
                        intercept, t_stone = self._intercept(result, now)
                        if intercept is None:
                            if (self._intercept_target is not None
                                    and stone.y <= AI_HOME_Y + core.MALLET_RADIUS + core.STONE_RADIUS
                                    and math.dist(self._mallet, self._intercept_target) <= ARRIVAL_TOLERANCE):
                                target = intercept = self._intercept_target
                                t_stone = 0.0
                                t_mallet = math.dist(self._mallet, intercept) / self.mallet_speed
                                reachable = True
                                reason = "holding arrived intercept through contact"
                            else:
                                reason = "predictor/intercept error: no defense-line crossing"
                        else:
                            distance = math.dist(self._mallet, intercept)
                            t_mallet = distance / self.mallet_speed
                            # 已覆盖拦截点时无需再次预留移动延迟，避免接触前撤回。
                            reachable = (distance <= ARRIVAL_TOLERANCE or
                                         t_mallet + core.DIFFICULTIES["普通"].reaction_delay <= t_stone)
                            if reachable:
                                target = intercept
                                self._intercept_target = intercept
                                reason = "reachable interception"
                                if not self._attempted:
                                    self.counters["intercept_attempted"] += 1
                                    self._attempted = True
                            else:
                                reason = "intercept unreachable; hold READY"
        if not fresh:
            target = None
            reason = "stale camera frame; target cleared"
        self._target = target
        return RallyDecision(self.state, target, reason, direction, self._mallet, source,
                             intercept, t_stone, t_mallet, reachable, tuple(transitions))

    @staticmethod
    def _segment_distance(start, end, point):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        t = 0.0 if not length_sq else max(0.0, min(1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
        return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)

    @staticmethod
    def _intercept(result, now):
        prediction = result.prediction
        stone = result.curling_state
        if prediction is None or result.ai_decision is None:
            return None, None
        source = prediction.source_state
        if (source != stone or not prediction.trajectory
                or not all(math.isfinite(v) for pt in prediction.trajectory for v in pt)):
            return None, None
        contact_y = AI_HOME_Y + core.MALLET_RADIUS + core.STONE_RADIUS
        if stone.y <= contact_y:
            return None, None
        length = 0.0
        # 碰撞采样没有时间戳；自由滑动只会减速，用路径/速度给保守到达下界。
        # ponytail: this lower bound ignores deceleration/PLC acceleration; verify --mallet-speed on hardware before play.
        speed_bound = max(math.hypot(stone.vx, stone.vy), core.MIN_WALL_BOUNCE_SPEED)
        for start, end in zip(prediction.trajectory, prediction.trajectory[1:]):
            segment = math.dist(start, end)
            if start[1] >= contact_y >= end[1] and start[1] > end[1]:
                fraction = (contact_y - start[1]) / (end[1] - start[1])
                x = start[0] + fraction * (end[0] - start[0])
                t_stone = max(0.0, (length + fraction * segment) / speed_bound - (now - stone.timestamp))
                left = core.RINK_LEFT + core.MALLET_RADIUS
                right = core.RINK_RIGHT - core.MALLET_RADIUS
                mallet_x = max(left, min(right, x))
                if abs(mallet_x - x) > core.STONE_RADIUS + core.MALLET_RADIUS:
                    return None, None
                return (mallet_x, AI_HOME_Y), t_stone
            length += segment
        return None, None
