"""Camera and bounded file requests inside the sole native serial controller."""

from collections import deque
import json
import re
import threading
import time

import numpy as np

from real_robot.model_control import prepare_execution


class ExternalControl:
    def __init__(self, hardware):
        self.hardware = hardware
        self.directory = hardware.output
        self.seen = set()
        self.frames = deque(maxlen=2)
        self.latest_frame = None
        self.error = None
        self.active = None
        self.executed = 0
        self.policy_executed = 0
        threading.Thread(target=self.capture, daemon=True).start()

    def capture(self):
        pipe = None
        try:
            import pyrealsense2 as rs
            pipe, config = rs.pipeline(), rs.config()
            config.enable_device("260322279327")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipe.start(config)
            for _ in range(15):
                pipe.wait_for_frames(5000)
            last = 0.
            while True:
                frame = pipe.wait_for_frames(5000).get_color_frame()
                now = time.perf_counter()
                camera_ns = time.perf_counter_ns()
                rgb = np.asanyarray(frame.get_data()).copy()
                self.latest_frame = {"rgb": rgb, "camera_ns": camera_ns,
                                     "frame_id": frame.get_frame_number()}
                if now-last < .09:
                    continue
                last = now
                hardware = self.hardware
                with hardware.lock:
                    q = hardware.current.copy()
                    feedback_ns = hardware.feedback_ns
                self.frames.append({"rgb": rgb,
                    "q": q, "camera_ns": camera_ns, "feedback_ns": feedback_ns,
                    "frame_id": frame.get_frame_number()})
        except BaseException as error:
            self.error = repr(error)
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass

    def reply(self, request, **result):
        (self.directory/(request["id"]+".response.json")).write_text(
            json.dumps({"id": request["id"], **result}, allow_nan=False), encoding="utf-8")

    def finish(self, error=None):
        if self.active is not None:
            self.reply(self.active, ok=error is None, error=error,
                       actual_motor_degrees=self.hardware.current.tolist(),
                       accepted_target_motor_degrees=self.hardware.target.tolist())
            if error is None and self.active["command"] == "execute":
                self.executed += 1
                self.policy_executed += self.active.get("purpose") == "policy"
            self.active = None

    def poll(self, current):
        path = self.directory/"model-request.json"
        if not path.exists():
            return None
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        identifier = request.get("id", "")
        if not re.fullmatch(r"[0-9a-f]{32}", identifier) or identifier in self.seen:
            return None
        self.seen.add(identifier)
        try:
            import os
            if request.get("controller_pid") != os.getpid():
                raise ValueError("请求不属于当前控制器")
            if request.get("control_generation") != self.hardware.control_generation:
                raise ValueError("操作员已暂停或接管，模型运行已取消")
            if not 0 <= time.perf_counter_ns()-request["controller_lease_ns"] < 3_000_000_000:
                raise ValueError("控制请求已过期")
            if self.active or self.hardware.busy or not self.hardware.paused:
                raise ValueError("请先暂停并等待当前操作结束")
            command = request["command"]
            if command == "sample":
                frames = list(self.frames)
                if self.error:
                    raise ValueError(self.error)
                if len(frames) != 2:
                    raise ValueError("相机尚未就绪")
                if (not 0 < frames[1]["camera_ns"]-frames[0]["camera_ns"] <= 200_000_000
                        or time.perf_counter_ns()-frames[-1]["camera_ns"] > 200_000_000
                        or any(abs(f["camera_ns"]-f["feedback_ns"]) > 100_000_000 for f in frames)):
                    raise ValueError("相机或关节观测过期")
                np.savez(self.directory/(identifier+".npz"),
                    rgb=np.stack([f["rgb"] for f in frames]), q=np.stack([f["q"] for f in frames]))
                self.reply(request, ok=True, observed_ns=frames[-1]["camera_ns"],
                           observed_q=frames[-1]["q"].tolist(), frame_ids=[f["frame_id"] for f in frames])
                return None
            if command == "execute":
                target = prepare_execution(self.hardware.model, current, request["observed_q"],
                    request["target"], request["observed_ns"], time.perf_counter_ns())
            else:
                raise ValueError("未知模型控制请求")
            self.active = request
            return target
        except Exception as error:
            self.reply(request, ok=False, error=str(error))
            return None
