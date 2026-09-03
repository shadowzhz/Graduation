"""冰壶轨迹简单预测。"""


def predict_position(x, y, vx, vy, time_seconds):
    """根据当前位置和速度预测未来某个时间点的位置。"""
    future_x = x + vx * time_seconds
    future_y = y + vy * time_seconds
    return future_x, future_y

'''
x,y:起始中心点坐标
vx,vy:当前速度分量
duration:预测的总时长
step:时间步长，值越小线条越平滑，计算量越大
'''
def predict_trajectory(x, y, vx, vy, duration=5.0, step=0.1):
    """生成一段匀速运动的预测轨迹。"""
    # 初始化
    trajectory = []
    current_time = 0.0

    while current_time <= duration:
        point = predict_position(x, y, vx, vy, current_time)
        trajectory.append(point)
        current_time += step
    return trajectory       # 返回包含多个坐标点的列表
