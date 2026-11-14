"""Native Windows B601 operator window; one Cartesian keypress is 6 mm."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from real_robot.manual_control import (GRIPPER_KEYS, MOVES, GripperProgress,
                                      JogModel, KeyLatch, check_arm_tracking)
from real_robot.recording import recording_run_finished
from real_robot.io_workers import AsyncJSONLines


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
        self.control_generation = 0
        self.external = None
        self.external_enabled = external_control
        self.event_logs = AsyncJSONLines(self.output)
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
            self.control_generation += 1
            self.paused = True
            self.pause_event.set()
            self.pending = None
            self.message = "已暂停，保持当前位置"

    def resume(self):
        with self.lock:
            if not self.connected:
                return
            if self.event_logs.error:
                self.message = "日志异常，无法继续："+self.event_logs.error
                return
            self.control_generation += 1
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
            self.external = external
            event_logs = self.event_logs
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
                sequence_applied = None
                try:
                    if not self.paused and (event_logs.error or external and external.rpc.error):
                        raise ValueError("后台通信或日志异常，模型已暂停")
                    self.model.check(current, .0005)
                    if any(state.status_code != 1 for state in states):
                        raise ValueError("电机状态变化，已暂停")
                    check_arm_tracking(current, last_sent)
                    external_target = external.poll(current) if external else None
                    if external and external.rhc_running:
                        if external.queue_generation != self.control_generation:
                            external.cancel_sequence()
                            goal = current.copy()
                            self.busy = False
                        else:
                            if (not external.recorder or not external.recorder.accepting or external.recorder.error):
                                raise ValueError("实验记录中断，模型序列已暂停")
                            if (external.error or external.latest_frame is None
                                    or time.perf_counter_ns()-external.latest_frame["camera_ns"] > 500_000_000):
                                raise ValueError("相机中断，模型序列已暂停")
                            sequence_applied = external.due_action(current, time.perf_counter_ns())
                            if sequence_applied:
                                goal = sequence_applied["target"].copy()
                                endpoint = goal.copy()
                                self.target = goal.copy()
                            self.message = "模型滚动时域运行 · 10 Hz 动作序列 · Esc 暂停"
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
                        event_logs.submit("manual-events", {"type": "model_command", "request": external.active,
                            "start_motor_degrees": current.tolist(), "target_motor_degrees": endpoint.tolist(),
                            "time_ns": time.perf_counter_ns()})
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
                        event_logs.submit("manual-events", {"type": "command", "id": self.events, "key": key,
                            "start_motor_degrees": current.tolist(), "target_motor_degrees": endpoint.tolist(),
                            "start_xyz_m": self.model.pose(current)[:3, 3].tolist(), "target_xyz_m": xyz.tolist(),
                            "step_m": .006 if key in MOVES else None, "time_ns": time.perf_counter_ns()})
                        self.message = f"{key}：正在移动到 6 mm 目标" if key in MOVES else f"{key}：夹爪开合中"
                    if trajectory is not None:
                        if (trajectory == "model" and self.setup.get("full_recording")
                                and (not external.recorder or not external.recorder.accepting)):
                            raise ValueError("实验记录已停止，模型已暂停")
                        if trajectory == "model" and external and external.recorder and external.recorder.error:
                            raise ValueError("实验记录异常，模型已暂停："+external.recorder.error)
                        if trajectory == "model" and external and (external.error or external.latest_frame is None
                                or time.perf_counter_ns()-external.latest_frame["camera_ns"] > 500_000_000):
                            raise ValueError("相机画面中断，模型已暂停")
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
                    sequence_applied = None
                    if external:
                        external.finish(str(error))
                    self.last_fault = {"type": "fault", "id": self.events, "reason": str(error),
                        "trajectory_key": trajectory, "current_motor_degrees": current.tolist(),
                        "last_sent_motor_degrees": last_sent.tolist(),
                        "joint_tracking_errors_deg": np.abs(current-last_sent).tolist(),
                        "endpoint_motor_degrees": endpoint.tolist(), "time_ns": time.perf_counter_ns()}
                    event_logs.submit("manual-events", self.last_fault)
                    goal, trajectory = current.copy(), None
                    with self.lock:
                        self.control_generation += 1
                        self.busy, self.paused, self.target = False, True, goal.copy()
                    self.message = str(error)
                decision_ns = time.perf_counter_ns()
                send(goal)
                sent_ns = time.perf_counter_ns()
                if sequence_applied is not None:
                    external.sequence_actions_sent += 1
                    event_logs.submit("sequence-actions", {key: value for key, value in sequence_applied.items() if key != "target"}
                        | {"target_motor_degrees": goal.tolist(), "current_motor_degrees": current.tolist(),
                           "command_sent_ns": sent_ns,
                           "lateness_ms": (sent_ns-sequence_applied["timestamp_ns"])/1e6})
                last_sent = goal.copy()
                if external and external.recorder:
                    external.recorder.submit_control({"cycle_started_ns": int(tick*1e9),
                        "feedback_read_ns": self.feedback_ns, "command_sent_ns": sent_ns,
                        "feedback_stage_ms": (self.feedback_ns-int(tick*1e9))/1e6,
                        "decision_stage_ms": (decision_ns-self.feedback_ns)/1e6,
                        "send_stage_ms": (sent_ns-decision_ns)/1e6,
                        "motor_positions_deg": current.tolist(), "sent_target_motor_degrees": goal.tolist(),
                        "endpoint_motor_degrees": self.target.tolist(),
                        "motor_velocity_deg_s": [math.degrees(state.vel) for state in states],
                        "motor_torque_nm": [float(state.torq) for state in states],
                        "motor_status_codes": [state.status_code for state in states],
                        "motor_mos_temperature_c": [float(state.t_mos) for state in states],
                        "motor_rotor_temperature_c": [float(state.t_rotor) for state in states],
                        "feedback_can_ids": [state.can_id for state in states],
                        "feedback_arbitration_ids": [state.arbitration_id for state in states],
                        "xyz_m": self.model.pose(current)[:3, 3].tolist(),
                        "paused": self.paused, "busy": self.busy, "trajectory": trajectory,
                        "control_generation": self.control_generation,
                        "rhc_running": bool(external and external.rhc_running),
                        "sequence_action": None if sequence_applied is None else
                            {k:v for k,v in sequence_applied.items() if k != "target"}})
                self.snapshot = {"state": "paused" if self.paused else
                    ("model_control" if external and external.rhc_running else "manual_control"), "pid": os.getpid(),
                    "connected": True, "busy": self.busy, "message": self.message,
                    "current_motor_degrees": current.tolist(), "target_motor_degrees": self.target.tolist(),
                    "xyz_m": self.model.pose(current)[:3, 3].tolist(),
                    "target_xyz_m": self.model.pose(self.target)[:3, 3].tolist(),
                    "motor_status_codes": [state.status_code for state in states],
                    "accepted_keys": self.events, "updated_ns": time.perf_counter_ns(),
                    "bounded_trajectory_commands": external.executed+external.sequence_actions_sent if external else 0,
                    "model_trajectory_commands": external.policy_executed+external.sequence_actions_sent if external else 0,
                    "rhc_running": bool(external and external.rhc_running),
                    "sequence_actions_sent": external.sequence_actions_sent if external else 0,
                    "model_control_enabled": external is not None,
                    "control_generation": self.control_generation,
                    "camera_ready": bool(external and len(external.frames) == 2),
                    "camera_error": external.error if external else None}
                self.snapshot.update(last_fault=self.last_fault, gripper_target_motor_degrees=float(self.target[6]),
                                     control_version=7, joint_velocity_deg_s=self.speed.tolist(),
                                     feedback_loop_hz=50, rpc_error=external.rpc.error if external else None,
                                     event_log_error=event_logs.error,
                                     event_logs_submitted=event_logs.submitted,event_logs_written=event_logs.written)
                time.sleep(max(0, .02-(time.perf_counter()-tick)))
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
    root.title("B601 控制 | 实时相机 · 模型运行")
    root.geometry("1280x740")
    root.minsize(1180, 700)
    font = ("Microsoft YaHei UI", 11)
    style = ttk.Style()
    style.configure("TButton", font=font, padding=10)
    style.configure("TLabel", font=font)
    outer = ttk.Frame(root, padding=12)
    outer.pack(fill="both", expand=True)
    camera_panel = ttk.Frame(outer)
    camera_panel.pack(side="left", fill="both", expand=True)
    ttk.Label(camera_panel, text="实时相机", font=(font[0], 20, "bold")).pack(anchor="w", pady=(0, 8))
    initial_image = tk.PhotoImage(width=640, height=480)
    camera_label = tk.Label(camera_panel, image=initial_image, background="#101820")
    camera_label.image = initial_image
    camera_label.pack(anchor="center")
    camera_text = tk.StringVar(value="等待相机画面")
    ttk.Label(camera_panel, textvariable=camera_text).pack(anchor="w", pady=8)
    model_text = tk.StringVar(value="模型尚未运行")
    ttk.Label(camera_panel, textvariable=model_text, wraplength=630).pack(anchor="w", pady=6)
    model_actions = ttk.Frame(camera_panel)
    model_actions.pack(fill="x", pady=8)
    frame = ttk.Frame(outer, padding=(12, 0, 0, 0))
    frame.pack(side="left", fill="both", expand=True)
    ttk.Label(frame, text="B601 末端手动控制", font=(font[0], 21, "bold")).pack(anchor="w")
    ttk.Label(frame, text="底座坐标系 · 每次平移 6 mm · 长按不连发 · 不排队").pack(anchor="w", pady=(5, 12))
    hardware = Hardware(setup, args.output, external_control=args.model_control)
    latch = KeyLatch()
    coordinates = tk.StringVar(value="X —     Y —     Z — mm")
    ttk.Label(frame, textvariable=coordinates, font=("Consolas", 16)).pack(anchor="w", pady=8)
    target_text = tk.StringVar(value="目标：保持当前位置")
    ttk.Label(frame, textvariable=target_text).pack(anchor="w")
    gripper_text = tk.StringVar(value="夹爪：等待反馈")
    ttk.Label(frame, textvariable=gripper_text).pack(anchor="w", pady=4)
    message = tk.StringVar(value="等待接管串口")
    ttk.Label(frame, textvariable=message, wraplength=530, foreground="#176045").pack(anchor="w", pady=12)
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
    ttk.Label(frame, text="夹爪每次 8 电机度，左键张开、右键闭合。", wraplength=530).pack(anchor="w", pady=5)
    bounds = np.array(setup["bounds"])*1000
    ttk.Label(frame, text="工作区 (mm)："+"   ".join(f"{axis} [{lo:.1f}, {hi:.1f}]" for axis, (lo, hi) in zip("XYZ", bounds)), wraplength=530).pack(anchor="w", pady=5)
    ttk.Label(frame, text="Esc 或窗口失焦会暂停模型；关闭窗口会最小化并保持电机使能。", wraplength=530).pack(anchor="w", pady=5)

    model_process = None
    model_log = None
    recording_for_model = False
    recording_generation = None
    model_run_id = None
    ui_action_seen = None
    try:
        ui_action_seen = json.loads((args.output/"ui-action.json").read_text(encoding="utf-8")).get("id")
    except (OSError, ValueError):
        pass
    preview_stats = {}

    def start_model():
        nonlocal model_process, model_log, recording_for_model, recording_generation, model_run_id
        if model_process is not None and model_process.poll() is None:
            model_text.set("模型已在运行；按 Esc 可暂停")
            return
        if not hardware.connected or hardware.busy or hardware.external is None or len(hardware.external.frames) < 2:
            model_text.set("等待机械臂空闲及相机就绪后再启动")
            return
        if model_log is not None:
            model_log.close()
        hardware.pause()
        try:
            recorder = hardware.external.start_recording() if setup.get("full_recording") else None
        except Exception as error:
            model_text.set("记录未就绪，未启动模型："+str(error))
            return
        project = setup.get("wsl_project_dir", "/home/mingx/projects/diffusion-policy-pusht-reproduction")
        session = setup["wsl_session_dir"]
        model_log = (args.output/"model-runner.log").open("ab")
        command = ["wsl.exe", "-d", setup.get("wsl_distribution", "Ubuntu-24.04"), "--cd", project,
                   "--exec", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                   project+"/.venv/bin/python", "-u", "scripts/run_b601_model_trial.py",
                   "--session", session, "--continuous", "--duration", str(setup.get("trial_duration_seconds",300)),
                   "--control-generation", str(hardware.control_generation)]
        model_run_id = uuid.uuid4().hex
        command += ["--run-id", model_run_id]
        if not setup.get("align_reference_on_start", False):
            command += ["--resume"]
        if setup.get("execution_mode") == "receding_horizon":
            command += ["--rhc"]
        if setup.get("hold_validation",False):
            command += ["--hold-validation"]
        if recorder:
            command += ["--recording-id", recorder.directory.name]
        try:
            model_process = subprocess.Popen(command, stdout=model_log, stderr=model_log,
                creationflags=subprocess.CREATE_NO_WINDOW)
            recording_for_model = recorder is not None
            recording_generation = hardware.control_generation
            model_text.set("正在加载已训练模型；随后连续运行，最长 5 分钟")
        except Exception as error:
            if recorder:
                recorder.stop()
            model_log.close()
            model_text.set("启动失败："+str(error))

    ttk.Button(model_actions, text="运行模型", command=start_model).pack(side="left", padx=4)
    ttk.Button(model_actions, text="Esc  暂停并保持", command=hardware.pause).pack(side="left", padx=4)
    ttk.Label(camera_panel, text="模型持续观察并执行，最长 5 分钟；遇到保护条件会暂停。", wraplength=630).pack(anchor="w")

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
    def close_window():
        hardware.pause()
        if hardware.external and hardware.external.recorder:
            root.after(150, hardware.external.recorder.stop)
        root.iconify()
    root.protocol("WM_DELETE_WINDOW", close_window)

    def refresh():
        nonlocal ui_action_seen, recording_for_model
        data = dict(hardware.snapshot)
        message.set(hardware.message)
        if "xyz_m" in data:
            coordinates.set("   ".join(f"{axis} {value*1000:8.2f}" for axis, value in zip("XYZ", data["xyz_m"]))+" mm")
            target_text.set("目标 (mm)："+"   ".join(f"{axis} {value*1000:.2f}" for axis, value in zip("XYZ", data["target_xyz_m"])))
            gripper_text.set(f"夹爪电机角：实测 {data['current_motor_degrees'][6]:.2f}°   目标 {data['target_motor_degrees'][6]:.2f}°")
        data.update(pid=os.getpid(), connected=hardware.connected,
                    window_visible=bool(root.winfo_viewable()), window_title=root.title(), message=hardware.message,
                    control_generation=hardware.control_generation,
                    model_process_running=bool(model_process and model_process.poll() is None))
        data["model_run_id"] = model_run_id
        data.update(preview_stats)
        if hardware.external and hardware.external.recorder:
            data["recording"] = hardware.external.recorder.stats()
        try:
            progress = json.loads((args.output/"model-trial.json").read_text(encoding="utf-8"))
            status_labels = {"loading": "加载模型", "aligning_reference": "检查起点", "running": "模型持续运行",
                             "complete_holding": "本次运行结束，保持当前位置", "stopped": "模型已暂停"}
            if model_run_id is None or progress.get("run_id") == model_run_id:
                model_text.set(f"{status_labels.get(progress['status'], progress['status'])} · 已执行 {progress['completed_steps']} 步"
                               +(f" · {progress['error']}" if progress.get("error") else ""))
            if recording_for_model and recording_run_finished(progress, model_run_id, recording_generation):
                hardware.external.recorder.stop()
                recording_for_model = False
        except (OSError, ValueError, KeyError):
            pass
        if recording_for_model and model_process and model_process.poll() is not None:
            hardware.external.recorder.stop()
            recording_for_model = False
        try:
            action_path = args.output/"ui-action.json"
            action = json.loads(action_path.read_text(encoding="utf-8")) if action_path.exists() else {}
            if action.get("id") is not None and action.get("id") != ui_action_seen:
                ui_action_seen = action["id"]
                if action.get("action") == "start_model":
                    start_model()
                elif action.get("action") == "start_recording" and hardware.external:
                    hardware.external.start_recording()
                elif action.get("action") == "stop_recording" and hardware.external and hardware.external.recorder:
                    if recording_for_model:
                        hardware.pause()
                    hardware.external.recorder.stop()
                elif action.get("action") == "pause":
                    hardware.pause()
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            model_text.set(str(error))
        try:
            (args.output/"teleop-status.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        root.after(150, refresh)

    preview_count, last_preview_id = 0, None
    preview_started = time.perf_counter()

    def refresh_preview():
        nonlocal preview_count, last_preview_id
        external = hardware.external
        latest = external.latest_frame if external else None
        if latest is not None:
            age = (time.perf_counter_ns()-latest["camera_ns"])/1e6
            if latest["frame_id"] != last_preview_id:
                rgb = latest["rgb"]
                picture = tk.PhotoImage(data=b"P6\n640 480\n255\n"+rgb.tobytes(), format="PPM")
                camera_label.configure(image=picture)
                camera_label.image = picture
                last_preview_id = latest["frame_id"]
                preview_count += 1
            camera_text.set(f"RealSense D405 · 640 × 480 · {'实时' if age < 500 else '画面中断'}")
            if external.recorder:
                stats = external.recorder.stats()
                state = "正在记录" if stats["accepting"] else ("记录已完成" if stats["finalized"] else "正在写完记录")
                if stats["error"]:
                    state = "记录异常"
                camera_text.set(camera_text.get()+f" · {state}：{stats['rgb_depth_frames_written']} 帧 / {stats['control_cycles_written']} 周期")
            preview_stats.update(preview_frames_rendered=preview_count, preview_frame_id=last_preview_id,
                preview_age_ms=age, preview_elapsed_seconds=time.perf_counter()-preview_started)
        elif external and external.error:
            camera_text.set("相机异常："+external.error)
        root.after(50, refresh_preview)

    refresh()
    refresh_preview()
    root.lift()
    root.attributes("-topmost", True)
    root.after(900, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
