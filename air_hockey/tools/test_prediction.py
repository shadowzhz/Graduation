"""轨迹预测模块离线测试。无需摄像头和台面。"""

from prediction.trajectory import TrajectoryPredictor


def main():
    predictor = TrajectoryPredictor()
    result = predictor.predict(
        x=100,
        y=200,
        vx=20,
        vy=-5,
        future_time=1.0,
    )
    print("预测结果:")
    print(f"x={result.x:.2f}, y={result.y:.2f}, t={result.timestamp:.2f}")


if __name__ == "__main__":
    main()
