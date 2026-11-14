"""Portable B601 Cartesian jog planning; no hardware access."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MOVES = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0),
         "d": (0, -1, 0), "Up": (0, 0, 1), "Down": (0, 0, -1)}
GRIPPER_KEYS = {"Left": -1, "Right": 1}


def check_arm_tracking(current, last_sent, limit=2.):
    """The six arm joints track transmitted targets; gripper travel has its own guard."""
    error = np.abs(np.asarray(current)[:6]-np.asarray(last_sent)[:6])
    index = int(np.argmax(error))
    if not np.isfinite(error).all() or error[index] > limit:
        raise ValueError(f"机械臂关节 {index+1} 跟踪误差 {error[index]:.2f}°，已保持当前位置")
    return error


class GripperProgress:
    def __init__(self, start, target, now):
        if not np.isfinite([start, target, now]).all() or abs(target-start) > 8.0001:
            raise ValueError("夹爪单次开合目标无效")
        self.start, self.target, self.started = start, target, now
        self.last_position, self.last_progress = start, now
        self.direction = np.sign(target-start)

    def update(self, position, now):
        if not np.isfinite([position, now]).all():
            raise ValueError("夹爪反馈无效")
        if position < min(self.start, self.target)-.5 or position > max(self.start, self.target)+.5:
            raise ValueError("夹爪超出本次开合行程，已保持")
        if abs(position-self.target) < .15:
            return True
        if (position-self.last_position)*self.direction >= .1:
            self.last_position, self.last_progress = position, now
        if now-self.last_progress > 1.5:
            raise ValueError("夹爪 1.5 秒无开合进展，可能遇阻或到限位，已保持")
        if now-self.started > 10.:
            raise ValueError("夹爪开合超过 10 秒未到位，已保持")
        return False


class KeyLatch:
    def __init__(self):
        self.pressed = set()

    def press(self, key):
        if key in self.pressed:
            return False
        self.pressed.add(key)
        return True

    def release(self, key):
        self.pressed.discard(key)


class JogModel:
    def __init__(self, setup):
        self.origins = np.array(setup["origins"], dtype=float)
        self.axes = np.array(setup["axes"], dtype=float)
        self.tool = np.array(setup["tool"], dtype=float)
        self.limits = np.array(setup["limits"], dtype=float)
        self.bounds = np.array(setup["bounds"], dtype=float)
        if (self.origins.shape != (6, 4, 4) or self.axes.shape != (6, 3)
                or self.tool.shape != (4, 4) or self.limits.shape != (7, 2)
                or self.bounds.shape != (3, 2)):
            raise ValueError("Invalid kinematics or bounds shape")
        for value in (self.origins, self.axes, self.tool, self.limits, self.bounds):
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite kinematics or bounds")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]) or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Bounds must be increasing")

    def pose(self, degrees):
        transform = np.eye(4)
        for angle, origin, axis in zip(np.deg2rad(degrees[:6]), self.origins, self.axes):
            transform = transform@origin
            transform[:3, :3] = transform[:3, :3]@Rotation.from_rotvec(angle*axis).as_matrix()
        return transform@self.tool

    def check(self, degrees, workspace_tolerance=0.):
        degrees = np.asarray(degrees)
        if degrees.shape != (7,) or not np.isfinite(degrees).all():
            raise ValueError("关节反馈无效")
        if np.any(degrees < self.limits[:, 0]-.1) or np.any(degrees > self.limits[:, 1]+.1):
            raise ValueError("当前位置超出关节限位，已保持")
        point = self.pose(degrees)[:3, 3]
        if np.any(point < self.bounds[:, 0]-workspace_tolerance) or np.any(point > self.bounds[:, 1]+workspace_tolerance):
            raise ValueError("末端位置超出工作区，已保持")
        return point

    def goal(self, current, key):
        current = np.asarray(current, dtype=float)
        start = self.check(current, .0005)
        target = current.copy()
        if key in GRIPPER_KEYS:
            target[6] = np.clip(current[6]+GRIPPER_KEYS[key]*8., *self.limits[6])
            if abs(target[6]-current[6]) < .01:
                raise ValueError("夹爪已到对应限位")
            return target, self.pose(target)[:3, 3]
        if key not in MOVES:
            raise ValueError("未知操作键")
        destination = start+.006*np.array(MOVES[key])
        if np.any(destination < self.bounds[:, 0]) or np.any(destination > self.bounds[:, 1]):
            raise ValueError("6 mm 目标越界，本次按键未执行")
        orientation = self.pose(current)[:3, :3]
        lower = np.maximum(self.limits[:6, 0], current[:6]-8.)
        upper = np.minimum(self.limits[:6, 1], current[:6]+8.)

        def residual(joints):
            pose = self.pose(joints)
            rotation = Rotation.from_matrix(orientation.T@pose[:3, :3]).as_rotvec()
            return np.r_[pose[:3, 3]-destination, .12*rotation]

        solution = least_squares(residual, np.clip(current[:6], lower, upper),
                                 bounds=(lower, upper), max_nfev=100,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > .0001 or np.linalg.norm(error[3:]) > .001:
            raise ValueError("无法在局部关节范围内完成 6 mm，本次按键未执行")
        target[:6] = solution.x
        # Check joint-interpolated waypoints, not only the Cartesian endpoint.
        for fraction in np.linspace(0, 1, 41):
            self.check(current+(target-current)*fraction, .0003)
        return target, destination
