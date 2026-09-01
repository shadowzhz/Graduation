"""冰壶轨迹简单预测。"""


def predict_position(x, y, vx, vy, time_seconds):
    """根据当前位置和速度预测未来某个时间点的位置。"""
    future_x = x + vx * time_seconds
    future_y = y + vy * time_seconds
    return future_x, future_y


def predict_trajectory(x, y, vx, vy, duration=5.0, step=0.1):
    """生成一段匀速运动的预测轨迹。"""
    trajectory = []
    current_time = 0.0
    while current_time <= duration:
        point = predict_position(x, y, vx, vy, current_time)
        trajectory.append(point)
        current_time += step
    return trajectory
