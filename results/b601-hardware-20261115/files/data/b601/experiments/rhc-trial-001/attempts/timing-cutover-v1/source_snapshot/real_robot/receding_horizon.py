"""Timestamped receding-horizon plans; portable and free of hardware I/O."""

import numpy as np

DT_NS = 100_000_000
MAX_STEP = np.array([2.]*6+[8.])


def validate_sequence(model, observed, actions):
    actions, previous = np.asarray(actions, float), np.asarray(observed, float)
    if actions.shape != (15, 7) or previous.shape != (7,) or not np.isfinite(actions).all():
        raise ValueError("滚动时域目标必须为有限的 15×7 序列")
    for index, target in enumerate(actions):
        if index > 0 and np.any(np.abs(target-previous) > MAX_STEP+1e-6):
            raise ValueError("预测动作序列变化超过采集时的单步范围")
        for fraction in np.linspace(0, 1, 21):
            model.check(previous+(target-previous)*fraction)
        previous = target
    return actions


def project_dispatch_target(model, current, target):
    """Match the collector driver's per-motor relative-target clipping."""
    current, target = np.asarray(current,float), np.asarray(target,float)
    if current.shape != (7,) or target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("模型目标无效")
    projected = current+np.clip(target-current, -MAX_STEP, MAX_STEP)
    for fraction in np.linspace(0,1,21):
        model.check(current+(projected-current)*fraction)
    return projected


class TimedActionQueue:
    def __init__(self):
        self.points = []
        self.last_timestamp_ns = None

    def replace(self, actions, timestamps, now_ns, sequence_id):
        actions, timestamps = np.asarray(actions, float), np.asarray(timestamps, np.int64)
        if (actions.shape != (15, 7) or timestamps.shape != (15,) or not np.isfinite(actions).all()
                or not np.all(np.diff(timestamps) == DT_NS)):
            raise ValueError("动作序列或 10 Hz 时间戳无效")
        keep = np.flatnonzero(timestamps > now_ns+10_000_000)
        if not len(keep):
            raise ValueError("整段模型预测已过期，已保持")
        first = int(timestamps[keep[0]])
        prefix = [point for point in self.points if point["timestamp_ns"] < first]
        new = [{"target": actions[i].copy(), "timestamp_ns": int(timestamps[i]),
                "sequence_id": sequence_id, "action_index": int(i)} for i in keep]
        self.points = prefix+new
        self.last_timestamp_ns = int(timestamps[-1])
        return {"accepted_action_indices": keep.tolist(), "discarded_expired_actions": int(keep[0]),
                "first_timestamp_ns": first, "last_timestamp_ns": self.last_timestamp_ns}

    def due(self, now_ns):
        if not self.points or self.points[0]["timestamp_ns"] > now_ns:
            return None
        point = self.points[0]
        if now_ns-point["timestamp_ns"] > 75_000_000:
            raise ValueError("动作调度迟到超过 75 ms，已保持")
        self.points.pop(0)
        return point

    def clear(self):
        self.points = []
        self.last_timestamp_ns = None
