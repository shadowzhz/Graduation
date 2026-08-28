"""冰球的几何辅助函数、运动和碰撞规则。"""

import math
from dataclasses import dataclass, field

import air_hockey_config as layout


def clamp(value, low, high):
    return max(low, min(high, value))


def reflected_coordinate(value, low, high):
    """把坐标按镜面反射折叠回 [low, high]，预测冰球反弹位置用。"""
    span = high - low
    if span <= 0:
        return low
    folded = (value - low) % (2 * span)
    if folded > span:
        folded = 2 * span - folded
    return low + folded


def puck_inside_goal_mouth(x):
    return layout.GOAL_LEFT + layout.PUCK_RADIUS < x < layout.GOAL_RIGHT - layout.PUCK_RADIUS


def circle_post_contact(circle_x, circle_y, post_x, post_y, minimum_distance):
    """圆和门柱的接触检测，没碰到返回 None。"""
    dx = circle_x - post_x
    dy = circle_y - post_y
    distance_sq = dx * dx + dy * dy
    if distance_sq >= minimum_distance * minimum_distance:
        return None
    if distance_sq > layout.COLLISION_EPSILON:
        distance = math.sqrt(distance_sq)
        return distance, dx / distance, dy / distance
    # 重合时拿指向场地中心的方向当法线
    nx = layout.RINK_CENTER_X - post_x
    ny = layout.RINK_CENTER_Y - post_y
    length = math.hypot(nx, ny)
    if length <= layout.COLLISION_EPSILON:
        return 0.0, 0.0, 1.0
    return 0.0, nx / length, ny / length


@dataclass
class PuckMotion:
    x: float = field(default_factory=lambda: layout.RINK_CENTER_X)
    y: float = field(default_factory=lambda: layout.RINK_CENTER_Y)
    vx: float = 0.0
    vy: float = 0.0
    target_vx: float = 0.0
    target_vy: float = 0.0
    response_active: bool = False

    @staticmethod
    def _limited_velocity(vx, vy):
        speed = math.hypot(vx, vy)
        if speed <= layout.MAX_PUCK_SPEED:
            return vx, vy
        scale = layout.MAX_PUCK_SPEED / speed
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
            self.response_active = math.hypot(self.target_vx - self.vx, self.target_vy - self.vy) > layout.COLLISION_EPSILON
        else:
            self.target_vx = self.vx
            self.target_vy = self.vy

    def advance_velocity(self, dt):
        if self.response_active and dt > 0:
            delta_x = self.target_vx - self.vx
            delta_y = self.target_vy - self.vy
            delta_speed = math.hypot(delta_x, delta_y)
            velocity_step = layout.PUCK_RESPONSE_ACCELERATION * dt
            if delta_speed <= velocity_step + 1e-9:
                self.vx = self.target_vx
                self.vy = self.target_vy
                self.response_active = False
            else:
                scale = velocity_step / delta_speed
                self.vx += delta_x * scale
                self.vy += delta_y * scale
        speed = math.hypot(self.vx, self.vy)
        new_speed = max(0.0, speed - layout.PUCK_FRICTION_DECELERATION * dt)
        if speed <= 1e-9 or new_speed <= 1e-9:
            self.vx = self.vy = 0.0
        else:
            scale = new_speed / speed
            self.vx *= scale
            self.vy *= scale
        self.vx, self.vy = self._limited_velocity(self.vx, self.vy)

    def resolve_walls(self):
        left_limit = layout.RINK_LEFT + layout.PUCK_RADIUS
        right_limit = layout.RINK_RIGHT - layout.PUCK_RADIUS
        top_limit = layout.RINK_TOP + layout.PUCK_RADIUS
        bottom_limit = layout.RINK_BOTTOM - layout.PUCK_RADIUS
        vx, vy = self.vx, self.vy
        target_vx, target_vy = self.target_vx, self.target_vy
        reflected_vx, reflected_vy = vx, vy
        reflected_target_vx, reflected_target_vy = target_vx, target_vy
        bounced = False
        if self.x < left_limit - layout.COLLISION_EPSILON:
            self.x = left_limit
            if vx < -layout.COLLISION_EPSILON:
                reflected_vx = abs(vx) * layout.WALL_RESTITUTION
                if target_vx < -layout.COLLISION_EPSILON:
                    reflected_target_vx = abs(target_vx) * layout.WALL_RESTITUTION
                bounced = True
        elif self.x > right_limit + layout.COLLISION_EPSILON:
            self.x = right_limit
            if vx > layout.COLLISION_EPSILON:
                reflected_vx = -abs(vx) * layout.WALL_RESTITUTION
                if target_vx > layout.COLLISION_EPSILON:
                    reflected_target_vx = -abs(target_vx) * layout.WALL_RESTITUTION
                bounced = True
        elif self.x <= left_limit + layout.COLLISION_EPSILON and vx < -layout.COLLISION_EPSILON:
            self.x = left_limit
            reflected_vx = abs(vx) * layout.WALL_RESTITUTION
            if target_vx < -layout.COLLISION_EPSILON:
                reflected_target_vx = abs(target_vx) * layout.WALL_RESTITUTION
            bounced = True
        elif self.x >= right_limit - layout.COLLISION_EPSILON and vx > layout.COLLISION_EPSILON:
            self.x = right_limit
            reflected_vx = -abs(vx) * layout.WALL_RESTITUTION
            if target_vx > layout.COLLISION_EPSILON:
                reflected_target_vx = -abs(target_vx) * layout.WALL_RESTITUTION
            bounced = True
        # 球门口不封上下边，让球能进洞
        if not puck_inside_goal_mouth(self.x):
            if self.y < top_limit - layout.COLLISION_EPSILON:
                self.y = top_limit
                if vy < -layout.COLLISION_EPSILON:
                    reflected_vy = abs(vy) * layout.WALL_RESTITUTION
                    if target_vy < -layout.COLLISION_EPSILON:
                        reflected_target_vy = abs(target_vy) * layout.WALL_RESTITUTION
                    bounced = True
            elif self.y > bottom_limit + layout.COLLISION_EPSILON:
                self.y = bottom_limit
                if vy > layout.COLLISION_EPSILON:
                    reflected_vy = -abs(vy) * layout.WALL_RESTITUTION
                    if target_vy > layout.COLLISION_EPSILON:
                        reflected_target_vy = -abs(target_vy) * layout.WALL_RESTITUTION
                    bounced = True
            elif self.y <= top_limit + layout.COLLISION_EPSILON and vy < -layout.COLLISION_EPSILON:
                self.y = top_limit
                reflected_vy = abs(vy) * layout.WALL_RESTITUTION
                if target_vy < -layout.COLLISION_EPSILON:
                    reflected_target_vy = abs(target_vy) * layout.WALL_RESTITUTION
                bounced = True
            elif self.y >= bottom_limit - layout.COLLISION_EPSILON and vy > layout.COLLISION_EPSILON:
                self.y = bottom_limit
                reflected_vy = -abs(vy) * layout.WALL_RESTITUTION
                if target_vy > layout.COLLISION_EPSILON:
                    reflected_target_vy = -abs(target_vy) * layout.WALL_RESTITUTION
                bounced = True
        # 卡在角落出不来时给一个最小弹出速度
        at_left_or_right = self.x <= left_limit + layout.COLLISION_EPSILON or self.x >= right_limit - layout.COLLISION_EPSILON
        at_top_or_bottom = self.y <= top_limit + layout.COLLISION_EPSILON or self.y >= bottom_limit - layout.COLLISION_EPSILON
        current_speed = math.hypot(vx, vy)
        if at_left_or_right and at_top_or_bottom and current_speed <= layout.COLLISION_EPSILON:
            direction_x = 1.0 if self.x <= left_limit + layout.COLLISION_EPSILON else -1.0
            direction_y = 1.0 if self.y <= top_limit + layout.COLLISION_EPSILON else -1.0
            component_speed = layout.MIN_WALL_BOUNCE_SPEED / math.sqrt(2.0)
            reflected_vx = direction_x * component_speed
            reflected_vy = direction_y * component_speed
            bounced = True
        elif bounced and current_speed > layout.COLLISION_EPSILON and at_left_or_right and at_top_or_bottom:
            reflected_speed = math.hypot(reflected_vx, reflected_vy)
            if reflected_speed < layout.MIN_WALL_BOUNCE_SPEED:
                scale = layout.MIN_WALL_BOUNCE_SPEED / reflected_speed
                reflected_vx *= scale
                reflected_vy *= scale
        if bounced:
            if current_speed <= layout.COLLISION_EPSILON:
                self.set_immediate_velocity(reflected_vx, reflected_vy)
            else:
                self.set_wall_reflection(reflected_vx, reflected_vy, reflected_target_vx, reflected_target_vy)
        return bounced

    def resolve_goal_posts(self):
        bounced = False
        minimum_distance = layout.PUCK_RADIUS + layout.GOAL_POST_RADIUS
        for post_x, post_y in layout.GOAL_POSTS:
            contact = circle_post_contact(self.x, self.y, post_x, post_y, minimum_distance)
            if contact is None:
                continue
            _distance, nx, ny = contact
            self.x = post_x + nx * minimum_distance
            self.y = post_y + ny * minimum_distance
            vx, vy = self.collision_velocity()
            normal_speed = vx * nx + vy * ny
            if normal_speed < 0:
                impulse = (1.0 + layout.GOAL_POST_RESTITUTION) * normal_speed
                self.set_immediate_velocity(vx - impulse * nx, vy - impulse * ny)
                bounced = True
        return bounced


def goal_scorer(puck):
    """球整体越过门线时返回得分方，否则返回 None。"""
    if not puck_inside_goal_mouth(puck.x):
        return None
    if puck.y + layout.PUCK_RADIUS < layout.RINK_TOP:
        return "player"
    if puck.y - layout.PUCK_RADIUS > layout.RINK_BOTTOM:
        return "ai"
    return None
