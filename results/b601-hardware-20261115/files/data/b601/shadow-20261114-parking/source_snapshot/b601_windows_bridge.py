"""Windows camera/feedback worker. Its RPC surface has no actuation commands."""

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real_robot.shadow import JOINTS, driver_config, write_packet


class Acquisition:
    def __init__(self, args, config):
        import numpy as np
        import pyrealsense2 as rs
        from motorbridge import Controller
        self.np, self.rs = np, rs
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.frames = deque(maxlen=2)
        self.error = None
        self.count = 0
        self.config = config
        self.controller = Controller.from_dm_serial(args.port, 921600)
        self.motors = [self.controller.add_damiao_motor(
            *config["motor_can_ids"][name], "4340P" if i < 3 else "4310")
            for i, name in enumerate(JOINTS)]
        context = rs.context()
        devices = [device for device in context.query_devices()
                   if device.get_info(rs.camera_info.serial_number) == args.camera_serial]
        if len(devices) != 1:
            raise RuntimeError("Requested RealSense serial is unavailable")
        self.pipe = rs.pipeline(context)
        options = rs.config()
        options.enable_device(args.camera_serial)
        options.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = self.pipe.start(options)
        self.camera = {"name": profile.get_device().get_info(rs.camera_info.name),
                       "serial": args.camera_serial, "format": "RGB8", "size": [640, 480]}
        self.output = args.output
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        try:
            for _ in range(15):
                self.pipe.wait_for_frames(5000)
            next_tick = time.perf_counter()
            with (self.output/"acquisition.jsonl").open("w", encoding="utf-8") as log:
                while not self.stop_event.is_set():
                    if self.stop_event.wait(max(0, next_tick-time.perf_counter())):
                        break
                    frames = self.pipe.wait_for_frames(5000)
                    frame = frames.get_color_frame()
                    if not frame:
                        raise RuntimeError("No RealSense color frame")
                    camera_ns = time.perf_counter_ns()
                    rgb = self.np.asanyarray(frame.get_data()).copy()
                    request_start = time.perf_counter_ns()
                    for motor in self.motors:
                        motor.request_feedback()
                    # Explicit query/poll, without enable/mode/configuration calls.
                    for _ in range(3):
                        time.sleep(.003)
                        self.controller.poll_feedback_once()
                    states = [motor.get_state() for motor in self.motors]
                    if any(state is None for state in states):
                        raise RuntimeError("Incomplete seven-motor feedback cache")
                    feedback_ns = time.perf_counter_ns()
                    positions = [math.degrees(state.pos) for state in states]
                    if not all(math.isfinite(position) for position in positions):
                        raise RuntimeError("Nonfinite feedback")
                    metadata = {"frame_id": self.count, "camera_frame_number": frame.get_frame_number(),
                                "camera_received_ns": camera_ns, "feedback_request_start_ns": request_start,
                                "feedback_read_complete_ns": feedback_ns,
                                "camera_sensor_timestamp_ms": frame.get_timestamp(),
                                "joint_pos_motor_degrees": positions,
                                "motor_status_codes": [state.status_code for state in states],
                                "motor_mos_temperature_c": [state.t_mos for state in states],
                                "feedback_per_motor_freshness": "unknown: SDK state has no receive timestamp"}
                    log.write(json.dumps(metadata, allow_nan=False)+"\n")
                    log.flush()
                    with self.condition:
                        self.frames.append((metadata, rgb))
                        self.count += 1
                        self.condition.notify_all()
                    next_tick += .1
                    if time.perf_counter()-next_tick > .1:
                        next_tick = time.perf_counter()
        except BaseException as error:
            with self.condition:
                self.error = repr(error)
                self.condition.notify_all()
        finally:
            self.pipe.stop()

    def sample(self):
        deadline = time.perf_counter()+7
        with self.condition:
            while len(self.frames) < 2 and self.error is None:
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for two observations")
                self.condition.wait(remaining)
            if self.error:
                raise RuntimeError(self.error)
            selected = list(self.frames)
        return {"kind": "sample", "clock_domain": "Windows perf_counter_ns",
                "frames": [row[0] for row in selected], "image_shape": [2, 480, 640, 3],
                "snapshot_selected_ns": time.perf_counter_ns()}, b"".join(row[1].tobytes() for row in selected)


def serve(args, stream):
    config = driver_config(args.driver_source)
    acquisition = Acquisition(args, config)
    try:
        write_packet(stream, {"kind": "hello", "protocol": 1, "mode": "predict_only",
                              "port": args.port, "camera": acquisition.camera, "joint_names": list(JOINTS),
                              "driver_config": config,
                              "driver_source_sha256": hashlib.sha256(args.driver_source.read_bytes()).hexdigest(),
                              "allowed_requests": ["ping", "sample", "stop"],
                              "motion_commands_sent": 0, "enable_commands_sent": 0})
        for line in sys.stdin.buffer:
            request = json.loads(line)
            if request == {"command": "ping"}:
                write_packet(stream, {"kind": "pong", "remote_ns": time.perf_counter_ns()})
            elif request == {"command": "sample"}:
                write_packet(stream, *acquisition.sample())
            elif request == {"command": "stop"}:
                acquisition.stop_event.set()
                acquisition.thread.join(timeout=6)
                if acquisition.thread.is_alive():
                    raise RuntimeError("Capture worker did not stop")
                write_packet(stream, {"kind": "stopped", "captured_frames": acquisition.count,
                                      "motion_commands_sent": 0, "enable_commands_sent": 0,
                                      "disable_commands_sent": 0, "calibration_commands_sent": 0})
                return
            else:
                raise ValueError("Unsupported request; this bridge has no actuation endpoint")
    finally:
        acquisition.stop_event.set()
        acquisition.thread.join(timeout=6)
        # Controller is deliberately never used as a context manager or shut down.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-dir", type=Path, default=ROOT/"data/tools/windows-hardware")
    parser.add_argument("--driver-source", type=Path,
                        default=ROOT/"data/tools/b601-driver-review/lerobot_robot_seeed_b601/config_seeed_b601_dm_follower.py")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sdk_dir))
    args.output.mkdir(parents=True, exist_ok=True)
    # Reserve the binary pipe; route Python/native diagnostic output to stderr.
    stream = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(stream.fileno(), os.O_BINARY)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    code = 0
    try:
        serve(args, stream)
    except BaseException as error:
        print(repr(error), file=sys.stderr, flush=True)
        try:
            write_packet(stream, {"kind": "error", "error": repr(error)})
        except BaseException:
            pass
        code = 1
    stream.flush()
    # Match the existing recorder's read-only probe: no SDK shutdown/torque commands.
    os._exit(code)


if __name__ == "__main__":
    main()
