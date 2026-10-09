"""冰壶的几何辅助函数、运动和碰撞规则。"""

import math
from dataclasses import dataclass, field

from . import core_config as core


def clamp(value, low, high):
    return max(low, min(high, value))


def apply_friction_velocity(vx: float, vy: float, dt: float) -> tuple[float, float]:
    """按现有自由滑动规则衰减速度，不处理响应加速度、碰撞或速度上限。"""
    speed = math.hypot(vx, vy)
    new_speed = max(0.0, speed - core.STONE_FRICTION_DECELERATION * dt)
    if speed <= 1e-9 or new_speed <= 1e-9:
        return 0.0, 0.0
    scale = new_speed / speed
    return vx * scale, vy * scale


def stone_inside_goal_mouth(x):
    return core.GOAL_LEFT + core.STONE_RADIUS < x < core.GOAL_RIGHT - core.STONE_RADIUS


def resolve_wall_axis(position, velocity, lower, upper):
    """Clamp one wall axis and reflect only an outward normal velocity."""
    if position < lower - core.COLLISION_EPSILON:
        position = lower
        bounced = velocity < -core.COLLISION_EPSILON
    elif position > upper + core.COLLISION_EPSILON:
        position = upper
        bounced = velocity > core.COLLISION_EPSILON
    elif position <= lower + core.COLLISION_EPSILON and velocity < -core.COLLISION_EPSILON:
        position = lower
        bounced = True
    elif position >= upper - core.COLLISION_EPSILON and velocity > core.COLLISION_EPSILON:
        position = upper
        bounced = True
    else:
        bounced = False
    return position, -velocity * core.WALL_RESTITUTION if bounced else velocity, bounced


def circle_post_contact(circle_x, circle_y, post_x, post_y, minimum_distance):
    """圆和门柱的接触检测，没碰到返回 None。"""
    dx = circle_x - post_x
    dy = circle_y - post_y
    distance_sq = dx * dx + dy * dy
    if distance_sq >= minimum_distance * minimum_distance:
        return None
    if distance_sq > core.COLLISION_EPSILON:
        distance = math.sqrt(distance_sq)
        return distance, dx / distance, dy / distance
    # 重合时拿指向场地中心的方向当法线
    nx = core.RINK_CENTER_X - post_x
    ny = core.RINK_CENTER_Y - post_y
    length = math.hypot(nx, ny)
    if length <= core.COLLISION_EPSILON:
        return 0.0, 0.0, 1.0
    return 0.0, nx / length, ny / length


@dataclass
class StoneMotion:
    x: float = field(default_factory=lambda: core.RINK_CENTER_X)
    y: float = field(default_factory=lambda: core.RINK_CENTER_Y)
    vx: float = 0.0
    vy: float = 0.0
    target_vx: float = 0.0
    target_vy: float = 0.0
    response_active: bool = False

    @staticmethod
    def _limited_velocity(vx, vy):
        speed = math.hypot(vx, vy)
        if speed <= core.MAX_STONE_SPEED:
            return vx, vy
        scale = core.MAX_STONE_SPEED / speed
        return vx * scale, vy * scale

    def collision_velocity(self):
        # 碰撞时用目标速度算，避免刚撞完速度还没跟上导致二次穿透
        if self.response_active and math.hypot(self.target_vx, self.target_vy) > 1e-9:
            return self.target_vx, self.target_vy
        return self.vx, self.vy

    def set_target_velocity(self, target_vx, target_vy):
        self.vx, self.vy = self._limited_velocity(self.vx, self.vy)
        self.target_vx, self.target_vy = self._limited_velocity(target_vx, target_vy)
        current_speed = math.hypot(self.vx, self.vy)
        target_speed = math.hypot(self.target_vx, self.target_vy)
        if current_speed > 1e-9 and target_speed > 1e-9:
            self.vx = self.target_vx / target_speed * current_speed
            self.vy = self.target_vy / target_speed * current_speed
        self.response_active = math.hypot(self.target_vx - self.vx, self.target_vy - self.vy) > 1e-9

    def set_immediate_velocity(self, vx, vy):
        self.vx, self.vy = self._limited_velocity(vx, vy)
        self.target_vx = self.vx
        self.target_vy = self.vy
        self.response_active = False

    def set_wall_reflection(self, vx, vy, target_vx, target_vy):
        self.vx, self.vy = self._limited_velocity(vx, vy)
        if self.response_active:
            self.target_vx, self.target_vy = self._limited_velocity(target_vx, target_vy)
            self.response_active = math.hypot(self.target_vx - self.vx, self.target_vy - self.vy) > core.COLLISION_EPSILON
        else:
            self.target_vx = self.vx
            self.target_vy = self.vy

    def advance_velocity(self, dt):
        if self.response_active and dt > 0:
            delta_x = self.target_vx - self.vx
            delta_y = self.target_vy - self.vy
            delta_speed = math.hypot(delta_x, delta_y)
            velocity_step = core.STONE_RESPONSE_ACCELERATION * dt
            if delta_speed <= velocity_step + 1e-9:
                self.vx = self.target_vx
                self.vy = self.target_vy
                self.response_active = False
            else:
                scale = velocity_step / delta_speed
                self.vx += delta_x * scale
                self.vy += delta_y * scale
        self.vx, self.vy = apply_friction_velocity(self.vx, self.vy, dt)
        self.vx, self.vy = self._limited_velocity(self.vx, self.vy)

    def advance_position(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def step(self, dt: float) -> bool:
        """Advance one free-motion step: position, walls/posts, then response and friction.

        Returns whether this step produced a wall or goal-post bounce.
        Goal scoring remains the caller's responsibility.
        """
        try:
            dt = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt must be finite and non-negative") from exc
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        if dt == 0.0:
            return False

        self.advance_position(dt)
        bounced = self.resolve_walls()
        bounced = self.resolve_goal_posts() or bounced
        self.advance_velocity(dt)
        return bounced

    def resolve_walls(self):
        left_limit = core.RINK_LEFT + core.STONE_RADIUS
        right_limit = core.RINK_RIGHT - core.STONE_RADIUS
        top_limit = core.RINK_TOP + core.STONE_RADIUS
        bottom_limit = core.RINK_BOTTOM - core.STONE_RADIUS
        vx, vy = self.vx, self.vy
        target_vx, target_vy = self.target_vx, self.target_vy
        reflected_vx, reflected_vy = vx, vy
        reflected_target_vx, reflected_target_vy = target_vx, target_vy
        self.x, reflected_vx, bounced = resolve_wall_axis(self.x, vx, left_limit, right_limit)
        if bounced:
            _, reflected_target_vx, _ = resolve_wall_axis(self.x, target_vx, left_limit, right_limit)
        # 球门口不封上下边，让球能进洞
        if not stone_inside_goal_mouth(self.x):
            self.y, reflected_vy, y_bounced = resolve_wall_axis(self.y, vy, top_limit, bottom_limit)
            if y_bounced:
                _, reflected_target_vy, _ = resolve_wall_axis(self.y, target_vy, top_limit, bottom_limit)
            bounced = bounced or y_bounced
        # 卡在角落出不来时给一个最小弹出速度
        at_left_or_right = self.x <= left_limit + core.COLLISION_EPSILON or self.x >= right_limit - core.COLLISION_EPSILON
        at_top_or_bottom = self.y <= top_limit + core.COLLISION_EPSILON or self.y >= bottom_limit - core.COLLISION_EPSILON
        current_speed = math.hypot(vx, vy)
        if at_left_or_right and at_top_or_bottom and current_speed <= core.COLLISION_EPSILON:
            direction_x = 1.0 if self.x <= left_limit + core.COLLISION_EPSILON else -1.0
            direction_y = 1.0 if self.y <= top_limit + core.COLLISION_EPSILON else -1.0
            component_speed = core.MIN_WALL_BOUNCE_SPEED / math.sqrt(2.0)
            reflected_vx = direction_x * component_speed
            reflected_vy = direction_y * component_speed
            bounced = True
        elif bounced and current_speed > core.COLLISION_EPSILON and at_left_or_right and at_top_or_bottom:
            reflected_speed = math.hypot(reflected_vx, reflected_vy)
            if reflected_speed < core.MIN_WALL_BOUNCE_SPEED:
                scale = core.MIN_WALL_BOUNCE_SPEED / reflected_speed
                reflected_vx *= scale
                reflected_vy *= scale
        if bounced:
            if current_speed <= core.COLLISION_EPSILON:
                self.set_immediate_velocity(reflected_vx, reflected_vy)
            else:
                self.set_wall_reflection(reflected_vx, reflected_vy, reflected_target_vx, reflected_target_vy)
        return bounced

    def resolve_goal_posts(self):
        bounced = False
        minimum_distance = core.STONE_RADIUS + core.GOAL_POST_RADIUS
        for post_x, post_y in core.GOAL_POSTS:
            contact = circle_post_contact(self.x, self.y, post_x, post_y, minimum_distance)
            if contact is None:
                continue
            _distance, nx, ny = contact
            self.x = post_x + nx * minimum_distance
            self.y = post_y + ny * minimum_distance
            vx, vy = self.collision_velocity()
            normal_speed = vx * nx + vy * ny
            if normal_speed < 0:
                impulse = (1.0 + core.GOAL_POST_RESTITUTION) * normal_speed
                self.set_immediate_velocity(vx - impulse * nx, vy - impulse * ny)
                bounced = True
        return bounced


def goal_scorer(stone):
    """球整体越过门线时返回得分方，否则返回 None。"""
    if not stone_inside_goal_mouth(stone.x):
        return None
    if stone.y + core.STONE_RADIUS < core.RINK_TOP:
        return "player"
    if stone.y - core.STONE_RADIUS > core.RINK_BOTTOM:
        return "ai"
    return None
