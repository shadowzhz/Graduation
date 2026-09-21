"""算法评估模块：自动统计状态估计与轨迹预测效果。"""

from .experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentReport,
    run_experiment,
)
from .metrics import (
    ErrorStats,
    EvaluationResult,
    EvaluationSource,
    endpoint_error,
    evaluate_result,
    position_errors,
)

__all__ = [
    "ErrorStats",
    "EvaluationResult",
    "EvaluationSource",
    "Experiment",
    "ExperimentConfig",
    "ExperimentReport",
    "endpoint_error",
    "evaluate_result",
    "position_errors",
    "run_experiment",
]
