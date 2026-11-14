"""Camera and bounded file requests inside the sole native serial controller."""

from collections import deque
import json
from pathlib import Path
import re
import threading
import time

import numpy as np

from real_robot.model_control import validate_joint_path, validate_observation


class ExternalControl:
    def __init__(self, hardware):
        self.hardware = hardware
        self.directory = hardware.output
        self.seen = set()
        self.frames = deque(maxlen=2)
        self.error = None
        self.active = None
        self.executed = 0
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
                if now-last < .09:
                    continue
                last = now
                hardware = self.hardware
                with hardware.lock:
                    q = hardware.current.copy()
                    feedback_ns = hardware.feedback_ns
                self.frames.append({"rgb": np.asanyarray(frame.get_data()).copy(),
                    "q": q, "camera_ns": time.perf_counter_ns(), "feedback_ns": feedback_ns,
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
                       actual_motor_degrees=self.hardware.current.tolist())
            if error is None and self.active["command"] == "execute":
                self.executed += 1
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
            if not 0 <= time.time()-request["issued_unix"] < 5:
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
            if command == "align_reference":
                reference = json.loads(Path(self.hardware.setup["reference_file"]).read_text(encoding="utf-8"))
                target = np.asarray(list(reference["observation.state"].values()), float)
                target[6] = current[6]
                validate_joint_path(self.hardware.model, current, target, max_delta=[30]*6+[0])
            elif command == "execute":
                validate_observation(current, request["observed_q"], request["observed_ns"], time.perf_counter_ns())
                target = validate_joint_path(self.hardware.model, current, request["target"], max_delta=[.5]*6+[2.])
                if np.linalg.norm(self.hardware.model.pose(target)[:3, 3]-self.hardware.model.pose(current)[:3, 3]) > .006:
                    raise ValueError("模型单步末端位移超过 6 mm")
            else:
                raise ValueError("未知模型控制请求")
            self.active = request
            return target
        except Exception as error:
            self.reply(request, ok=False, error=str(error))
            return None
