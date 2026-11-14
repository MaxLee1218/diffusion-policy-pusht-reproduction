"""Native Windows B601 operator window; one Cartesian keypress is 6 mm."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from real_robot.manual_control import (GRIPPER_KEYS, MOVES, GripperProgress,
                                      JogModel, KeyLatch, check_arm_tracking)


class Hardware:
    def __init__(self, setup, output, external_control=False):
        self.setup, self.output = setup, output
        self.model = JogModel(setup)
        self.speed = np.asarray(setup.get("joint_velocity_deg_s", [5.]*6+[12.]), float)
        if (self.speed.shape != (7,) or not np.isfinite(self.speed).all()
                or np.any(self.speed <= 0) or np.any(self.speed > np.array([30.]*6+[12.]))):
            raise ValueError("速度必须在已使用的采集设置范围内：机械臂 ≤30°/s，夹爪 ≤12°/s")
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.current = self.target = None
        self.pending = None
        self.busy = False
        self.paused = True
        self.connected = False
        self.message = "等待接管串口；电机由原进程保持"
        self.events = 0
        self.last_fault = None
        self.snapshot = {}
        self.feedback_ns = 0
        self.external_enabled = external_control
        threading.Thread(target=self.run, daemon=True).start()

    def request(self, key):
        with self.lock:
            if not self.connected or self.current is None:
                self.message = "尚未接管串口"
                return
            if self.paused:
                self.message = "当前已暂停，请点“开始键盘控制”"
                return
            if self.busy:
                self.message = "等待上一步到位；按键不会排队"
                return
            self.pending, self.busy = key, True

    def pause(self):
        with self.lock:
            self.paused = True
            self.pause_event.set()
            self.pending = None
            self.message = "已暂停，保持当前位置"

    def resume(self):
        with self.lock:
            if not self.connected:
                return
            self.paused = False
            self.pause_event.clear()
            self.message = "键盘控制已开启；每次平移目标 6 mm"

    def run(self):
        from motorbridge import Controller
        controller, motors = None, None
        while controller is None:
            try:
                controller = Controller.from_dm_serial(self.setup["port"], 921600)
            except Exception:
                time.sleep(.4)
        try:
            motors = [controller.add_damiao_motor(*ids, "4340P" if i < 3 else "4310")
                      for i, ids in enumerate(self.setup["motor_ids"])]

            def read():
                for motor in motors:
                    motor.request_feedback()
                for _ in range(3):
                    time.sleep(.003)
                    controller.poll_feedback_once()
                states = [motor.get_state() for motor in motors]
                if any(state is None for state in states):
                    raise RuntimeError("七维关节反馈不完整")
                q = np.array([math.degrees(state.pos) for state in states])
                if not np.isfinite(q).all():
                    raise RuntimeError("关节反馈出现非有限数值")
                return states, q

            def send(goal):
                for i, (motor, value) in enumerate(zip(motors, goal)):
                    if i < 6:
                        motor.send_pos_vel(math.radians(float(value)), math.radians(float(self.speed[i])))
                    else:
                        motor.send_force_pos(math.radians(float(value)), math.radians(float(self.speed[i])), .1)

            states, current = read()
            self.model.check(current, .0005)
            if any(state.status_code != 1 for state in states):
                raise RuntimeError("电机未全部使能；窗口不会自动使能或改模式")
            goal = current.copy()
            self.current = self.target = current.copy()
            self.feedback_ns = time.perf_counter_ns()
            self.connected = True
            self.message = "已接管并保持；点“开始键盘控制”后按键"
            trajectory, trajectory_start, duration = None, 0., 0.
            start_q, endpoint = current.copy(), current.copy()
            last_sent, gripper_progress = current.copy(), None
            speed = self.speed
            from scripts.b601_model_camera import ExternalControl
            external = ExternalControl(self) if self.external_enabled else None
            log = (self.output/"manual-events.jsonl").open("a", encoding="utf-8")
            while True:
                tick = time.perf_counter()
                states, current = read()
                with self.lock:
                    self.current = current.copy()
                    self.feedback_ns = time.perf_counter_ns()
                    key, self.pending = self.pending, None
                    paused = self.paused
                if self.pause_event.is_set():
                    if external:
                        external.finish("操作员暂停")
                    goal, trajectory = current.copy(), None
                    key, paused = None, True
                    with self.lock:
                        self.target, self.busy = goal.copy(), False
                    self.pause_event.clear()
                try:
                    self.model.check(current, .0005)
                    if any(state.status_code != 1 for state in states):
                        raise ValueError("电机状态变化，已暂停")
                    check_arm_tracking(current, last_sent)
                    external_target = external.poll(current) if external else None
                    if external_target is not None:
                        endpoint = external_target.copy()
                        start_q = current.copy()
                        duration = max(.15, float(np.max(np.abs(endpoint-start_q)/speed))*1.25)
                        trajectory_start = time.perf_counter()
                        trajectory, gripper_progress = "model", None
                        with self.lock:
                            self.target, self.busy, self.paused = endpoint.copy(), True, False
                        paused = False
                        self.message = "模型联调：低速执行；Esc 暂停"
                        log.write(json.dumps({"type": "model_command", "request": external.active,
                            "start_motor_degrees": current.tolist(), "target_motor_degrees": endpoint.tolist(),
                            "time_ns": time.perf_counter_ns()}, allow_nan=False)+"\n")
                        log.flush()
                    if key is not None and not paused:
                        endpoint, xyz = self.model.goal(current, key)
                        if self.pause_event.is_set():
                            continue
                        start_q = current.copy()
                        duration = max(.15, float(np.max(np.abs(endpoint-start_q)/speed))*1.25)
                        trajectory_start = time.perf_counter()
                        trajectory = key
                        gripper_progress = (GripperProgress(current[6], endpoint[6], trajectory_start)
                                            if key in GRIPPER_KEYS else None)
                        self.target = endpoint.copy()
                        self.events += 1
                        log.write(json.dumps({"type": "command", "id": self.events, "key": key,
                            "start_motor_degrees": current.tolist(), "target_motor_degrees": endpoint.tolist(),
                            "start_xyz_m": self.model.pose(current)[:3, 3].tolist(), "target_xyz_m": xyz.tolist(),
                            "step_m": .006 if key in MOVES else None, "time_ns": time.perf_counter_ns()}, allow_nan=False)+"\n")
                        log.flush()
                        self.message = f"{key}：正在移动到 6 mm 目标" if key in MOVES else f"{key}：夹爪开合中"
                    if trajectory is not None:
                        if gripper_progress is not None:
                            # Match the recorder's bounded 8-motor-degree gripper request.
                            # Its normal travel is not an arm-joint following error.
                            goal = endpoint.copy()
                            finished = gripper_progress.update(current[6], time.perf_counter())
                        else:
                            fraction = min(1., (time.perf_counter()-trajectory_start)/duration)
                            goal = start_q+(endpoint-start_q)*fraction
                            self.model.check(goal, .0003)
                            finished = fraction == 1. and np.max(np.abs(current-endpoint)) < .15
                            if time.perf_counter()-trajectory_start > duration+3.:
                                raise ValueError("机械臂到位超时，已保持当前位置")
                        if finished:
                            trajectory = None
                            with self.lock:
                                self.busy = False
                                if external and external.active:
                                    self.paused = True
                            if external:
                                external.finish()
                            self.message = "已到位，可继续按键"
                except ValueError as error:
                    if external:
                        external.finish(str(error))
                    self.last_fault = {"type": "fault", "id": self.events, "reason": str(error),
                        "trajectory_key": trajectory, "current_motor_degrees": current.tolist(),
                        "last_sent_motor_degrees": last_sent.tolist(),
                        "joint_tracking_errors_deg": np.abs(current-last_sent).tolist(),
                        "endpoint_motor_degrees": endpoint.tolist(), "time_ns": time.perf_counter_ns()}
                    log.write(json.dumps(self.last_fault, allow_nan=False)+"\n")
                    log.flush()
                    goal, trajectory = current.copy(), None
                    with self.lock:
                        self.busy, self.paused, self.target = False, True, goal.copy()
                    self.message = str(error)
                send(goal)
                last_sent = goal.copy()
                self.snapshot = {"state": "paused" if self.paused else "manual_control", "pid": os.getpid(),
                    "connected": True, "busy": self.busy, "message": self.message,
                    "current_motor_degrees": current.tolist(), "target_motor_degrees": self.target.tolist(),
                    "xyz_m": self.model.pose(current)[:3, 3].tolist(),
                    "target_xyz_m": self.model.pose(self.target)[:3, 3].tolist(),
                    "motor_status_codes": [state.status_code for state in states],
                    "accepted_keys": self.events, "updated_ns": time.perf_counter_ns(),
                    "bounded_trajectory_commands": external.executed if external else 0,
                    "model_trajectory_commands": external.policy_executed if external else 0,
                    "model_control_enabled": external is not None,
                    "camera_ready": bool(external and len(external.frames) == 2),
                    "camera_error": external.error if external else None}
                self.snapshot.update(last_fault=self.last_fault, gripper_target_motor_degrees=float(self.target[6]),
                                     control_version=3, joint_velocity_deg_s=self.speed.tolist())
                time.sleep(max(0, .05-(time.perf_counter()-tick)))
        except BaseException as error:
            self.message = "连接或反馈异常："+str(error)
            self.snapshot = {"state": "error", "error": repr(error), "pid": os.getpid()}
            self.connected = False
            # Never invoke controller shutdown, torque disable or a zero trajectory.


def main():
    import tkinter as tk
    from tkinter import ttk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-dir", type=Path, default=ROOT/"data/tools/windows-hardware")
    parser.add_argument("--model-control", action="store_true", help="Enable bounded single-step requests and camera")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sdk_dir))
    args.output.mkdir(parents=True, exist_ok=True)
    setup = json.loads(args.setup.read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("B601 控制 | 6 mm · 模型单步联调" if args.model_control else "B601 末端手动控制 | 6 mm · 开合修正版")
    root.geometry("900x660")
    root.minsize(820, 610)
    font = ("Microsoft YaHei UI", 11)
    style = ttk.Style()
    style.configure("TButton", font=font, padding=10)
    style.configure("TLabel", font=font)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="B601 末端手动控制", font=(font[0], 21, "bold")).pack(anchor="w")
    ttk.Label(frame, text="底座坐标系 · 每次平移 6 mm · 长按不连发 · 不排队").pack(anchor="w", pady=(5, 12))
    hardware = Hardware(setup, args.output, external_control=args.model_control)
    latch = KeyLatch()
    coordinates = tk.StringVar(value="X —     Y —     Z — mm")
    ttk.Label(frame, textvariable=coordinates, font=("Consolas", 22)).pack(anchor="w", pady=8)
    target_text = tk.StringVar(value="目标：保持当前位置")
    ttk.Label(frame, textvariable=target_text).pack(anchor="w")
    gripper_text = tk.StringVar(value="夹爪：等待反馈")
    ttk.Label(frame, textvariable=gripper_text).pack(anchor="w", pady=4)
    message = tk.StringVar(value="等待接管串口")
    ttk.Label(frame, textvariable=message, wraplength=830, foreground="#176045").pack(anchor="w", pady=12)
    pad = ttk.Frame(frame)
    pad.pack(fill="x", pady=8)
    buttons = [("W  X＋6 mm", "w", 0, 1), ("A  Y＋6 mm", "a", 1, 0),
               ("S  X－6 mm", "s", 1, 1), ("D  Y－6 mm", "d", 1, 2),
               ("↑  Z＋6 mm", "Up", 0, 3), ("↓  Z－6 mm", "Down", 1, 3),
               ("←  张开夹爪", "Left", 2, 0), ("→  闭合夹爪", "Right", 2, 2)]
    for text, key, row, column in buttons:
        ttk.Button(pad, text=text, command=lambda key=key: hardware.request(key)).grid(row=row, column=column, padx=5, pady=5, sticky="ew")
    for i in range(4):
        pad.columnconfigure(i, weight=1)
    actions = ttk.Frame(frame)
    actions.pack(fill="x", pady=12)
    ttk.Button(actions, text="开始键盘控制 / 继续", command=hardware.resume).pack(side="left", padx=5)
    ttk.Button(actions, text="Esc  保持 / 暂停", command=hardware.pause).pack(side="left", padx=5)
    ttk.Label(frame, text="夹爪每次 8 电机度，左键张开、右键闭合；没有毫米开度标定。", wraplength=830).pack(anchor="w", pady=5)
    bounds = np.array(setup["bounds"])*1000
    ttk.Label(frame, text="工作区 (mm)："+"   ".join(f"{axis} [{lo:.1f}, {hi:.1f}]" for axis, (lo, hi) in zip("XYZ", bounds)), wraplength=830).pack(anchor="w", pady=5)
    ttk.Label(frame, text="越界或无解会拒绝本次按键。失去焦点会暂停；关闭窗口会最小化并保持电机使能。", wraplength=830).pack(anchor="w", pady=5)

    def key_name(event):
        return event.keysym.lower() if len(event.keysym) == 1 else event.keysym

    def press(event):
        key = key_name(event)
        if key == "Escape":
            hardware.pause()
        elif (key in MOVES or key in GRIPPER_KEYS) and latch.press(key):
            hardware.request(key)
        return "break" if key in MOVES or key in GRIPPER_KEYS else None

    root.bind_all("<KeyPress>", press)
    root.bind_all("<KeyRelease>", lambda event: latch.release(key_name(event)))

    def check_focus():
        if root.focus_displayof() is None:
            hardware.pause()

    root.bind("<FocusOut>", lambda event: root.after(40, check_focus))
    root.protocol("WM_DELETE_WINDOW", lambda: (hardware.pause(), root.iconify()))

    def refresh():
        data = dict(hardware.snapshot)
        message.set(hardware.message)
        if "xyz_m" in data:
            coordinates.set("   ".join(f"{axis} {value*1000:8.2f}" for axis, value in zip("XYZ", data["xyz_m"]))+" mm")
            target_text.set("目标 (mm)："+"   ".join(f"{axis} {value*1000:.2f}" for axis, value in zip("XYZ", data["target_xyz_m"])))
            gripper_text.set(f"夹爪电机角：实测 {data['current_motor_degrees'][6]:.2f}°   目标 {data['target_motor_degrees'][6]:.2f}°")
        data.update(pid=os.getpid(), connected=hardware.connected,
                    window_visible=bool(root.winfo_viewable()), window_title=root.title(), message=hardware.message)
        try:
            (args.output/"teleop-status.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        root.after(150, refresh)

    refresh()
    root.lift()
    root.attributes("-topmost", True)
    root.after(900, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
