"""Validation for the first bounded B601 model trial; no hardware access."""

import numpy as np


def validate_joint_path(model, current, target, *, max_delta):
    current, target = np.asarray(current, float), np.asarray(target, float)
    if current.shape != (7,) or target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("模型关节目标无效")
    if np.any(np.abs(target-current) > np.asarray(max_delta)+1e-6):
        raise ValueError("模型单步关节变化超限")
    for fraction in np.linspace(0, 1, 201):
        model.check(current+(target-current)*fraction)
    return target


def bounded_target(model, current, prediction):
    """Keep action direction, with <=0.5 arm degrees, 2 gripper degrees, 6 mm."""
    current, prediction = np.asarray(current, float), np.asarray(prediction, float)
    if prediction.shape != (7,) or not np.isfinite(prediction).all():
        raise ValueError("模型预测无效")
    delta = prediction-current
    scale = min(1., 1./max(1., float(np.max(np.abs(delta)/np.array([.5]*6+[2.])))))
    target = current+delta*scale
    for _ in range(12):
        if np.linalg.norm(model.pose(target)[:3, 3]-model.pose(current)[:3, 3]) <= .006:
            return validate_joint_path(model, current, target, max_delta=[.5]*6+[2.])
        target = current+(target-current)*.5
    raise ValueError("模型单步位移超限")


def validate_observation(current, observed, observed_ns, now_ns):
    current = np.asarray(current, float)
    observed = np.asarray(observed, float)
    if (current.shape != (7,) or not np.isfinite(current).all()
            or observed.shape != (7,) or not np.isfinite(observed).all()
            or not 0 <= now_ns-observed_ns <= 1_000_000_000):
        raise ValueError("模型输入已过期")
    if np.max(np.abs(current-observed)) > .3:
        raise ValueError("模型观测后机械臂已移动")


def prepare_execution(model, current, observed, target, observed_ns, now_ns):
    """Validate the proposal, then apply the same step bound to dispatch feedback."""
    validate_observation(current, observed, observed_ns, now_ns)
    validate_joint_path(model, observed, target, max_delta=[.5]*6+[2.])
    return bounded_target(model, current, target)
