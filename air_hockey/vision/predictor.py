"""冰壶轨迹预测：支持桌面边界碰撞反弹与摩擦阻尼。"""

import math


def reflected_coordinate(value, low, high):
    """把坐标按镜面反射折叠回 [low, high]，用于反弹计算。"""
    span = high - low
    if span <= 0:
        return low
    folded = (value - low) % (2 * span)
    if folded > span:
        folded = 2 * span - folded
    return low + folded


def predict_position(x, y, vx, vy, time_seconds):
    """根据当前位置和速度预测未来某个时间点的位置（常速直线外推）。"""
    future_x = x + vx * time_seconds
    future_y = y + vy * time_seconds
    return future_x, future_y


def predict_trajectory(
    x,
    y,
    vx,
    vy,
    duration=2.0,
    step=0.06,
    bounds=None,
    radius=14.0,
    restitution=0.97,
    friction=72.0,
    max_bounces=5,
):
    """生成一段包含球桌边缘碰撞反弹和冰面摩擦阻尼的连续预测轨迹。

    参数:
        x, y: 冰壶起始位置（球桌坐标系）
        vx, vy: 当前速度分量
        duration: 预测总时长（秒）
        step: 时间步长（秒）
        bounds: 桌面边界 (min_x, max_x, min_y, max_y)，默认从配置加载
        radius: 冰壶半径，用于计算反弹边界
        restitution: 边壁反弹恢复系数 (0~1)
        friction: 冰面摩擦减速度 (px/s^2)
        max_bounces: 最大反弹折返次数
    """
    speed = math.hypot(vx, vy)
    if speed < 1.0 or duration <= 0:
        return [(x, y)]

    if bounds is None:
        try:
            import air_hockey_config as layout
            min_x, max_x = layout.RINK_LEFT, layout.RINK_RIGHT
            min_y, max_y = layout.RINK_TOP, layout.RINK_BOTTOM
            if radius is None:
                radius = layout.STONE_RADIUS
        except Exception:
            min_x, max_x, min_y, max_y = 36.0, 564.0, 46.0, 714.0
    else:
        min_x, max_x, min_y, max_y = bounds

    wall_left = min_x + radius
    wall_right = max_x - radius
    wall_top = min_y + radius
    wall_bottom = max_y - radius

    trajectory = [(x, y)]
    curr_x, curr_y = x, y
    curr_vx, curr_vy = vx, vy
    current_time = 0.0
    bounces = 0

    while current_time < duration:
        current_time += step
        curr_speed = math.hypot(curr_vx, curr_vy)
        if curr_speed <= 5.0:
            break

        if friction > 0:
            new_speed = max(0.0, curr_speed - friction * step)
            scale = new_speed / curr_speed
            curr_vx *= scale
            curr_vy *= scale

        next_x = curr_x + curr_vx * step
        next_y = curr_y + curr_vy * step

        # 左右桌边反弹
        if next_x < wall_left:
            next_x = reflected_coordinate(next_x, wall_left, wall_right)
            curr_vx = abs(curr_vx) * restitution
            bounces += 1
        elif next_x > wall_right:
            next_x = reflected_coordinate(next_x, wall_left, wall_right)
            curr_vx = -abs(curr_vx) * restitution
            bounces += 1

        # 上下桌边反弹
        if next_y < wall_top:
            next_y = reflected_coordinate(next_y, wall_top, wall_bottom)
            curr_vy = abs(curr_vy) * restitution
            bounces += 1
        elif next_y > wall_bottom:
            next_y = reflected_coordinate(next_y, wall_top, wall_bottom)
            curr_vy = -abs(curr_vy) * restitution
            bounces += 1

        curr_x, curr_y = next_x, next_y
        trajectory.append((curr_x, curr_y))

        if bounces >= max_bounces:
            break

    return trajectory
