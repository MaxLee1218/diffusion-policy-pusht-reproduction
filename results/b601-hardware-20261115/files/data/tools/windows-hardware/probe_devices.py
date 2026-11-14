"""Local, non-actuating Windows probes; dependencies live beside this script."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
KEEP_ALIVE = []


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(value, allow_nan=False), flush=True)


def feedback(output, port):
    from motorbridge import Controller
    began = time.perf_counter()
    controller = Controller.from_dm_serial(port, 921600)
    KEEP_ALIVE.append(controller)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
    rows = []
    for motor_id, name in enumerate(names, 1):
        motor = controller.add_damiao_motor(motor_id, motor_id+0x10,
                                            "4340P" if motor_id <= 3 else "4310")
        KEEP_ALIVE.append(motor)
        state = None
        for attempt in range(3):
            motor.request_feedback()
            time.sleep(.06)
            controller.poll_feedback_once()
            state = motor.get_state()
            if state is not None:
                break
        row = {"name": name, "motor_id": motor_id, "responded": state is not None,
               "attempts": attempt+1, "read_complete_monotonic_ns": time.perf_counter_ns()}
        if state is not None:
            row.update(position_motor_degrees=math.degrees(state.pos),
                       velocity_degrees_per_second=math.degrees(state.vel),
                       torque_nm=state.torq, mos_temperature_c=state.t_mos,
                       rotor_temperature_c=state.t_rotor, status_code=state.status_code,
                       feedback_can_id=state.can_id, arbitration_id=state.arbitration_id)
        rows.append(row)
    dump(output/"robot-feedback.json", {"checked_at": datetime.now(timezone.utc).isoformat(),
         "port": port, "baud": 921600, "feedback_only": True,
         "motion_commands_sent": 0, "enable_commands_sent": 0,
         "calibration_commands_sent": 0, "disable_commands_sent": 0,
         "all_seven_responded": all(row["responded"] for row in rows),
         "elapsed_seconds": time.perf_counter()-began, "motors": rows})
    # Same exit strategy as the recorder's existing _feedback probe: avoid SDK shutdown.
    os._exit(0)


def camera(output, serial):
    import numpy as np
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(config)
    try:
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        for _ in range(30):
            pipe.wait_for_frames(5000)
        timestamps, frame_numbers, sensor_times = [], [], []
        for _ in range(90):
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("Missing paired color/depth frame")
            timestamps.append(time.perf_counter_ns())
            frame_numbers.append(color.get_frame_number())
            sensor_times.append(color.get_timestamp())
            if len(timestamps) in (1, 45, 90):
                rgb = np.asanyarray(color.get_data()).copy()
                z16 = np.asanyarray(depth.get_data()).copy()
                np.save(output/f"color-{len(timestamps):03d}.npy", rgb)
                if len(timestamps) == 90:
                    np.save(output/"depth-last.npy", z16)
        intervals = np.diff(timestamps)/1e6
        valid = z16[z16 > 0]*scale
        dump(output/"camera.json", {"checked_at": datetime.now(timezone.utc).isoformat(),
            "name": device.get_info(rs.camera_info.name), "serial": serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            "configured_fps": 30, "warmup_frames": 30, "checked_frame_pairs": len(timestamps),
            "color_shape": list(rgb.shape), "color_dtype": str(rgb.dtype), "color_format": "RGB8",
            "depth_shape": list(z16.shape), "depth_dtype": str(z16.dtype),
            "receive_fps": (len(timestamps)-1)*1e9/(timestamps[-1]-timestamps[0]),
            "receive_interval_ms": {"median": float(np.median(intervals)), "p95": float(np.quantile(intervals,.95)), "max": float(intervals.max())},
            "color_frame_number_first": frame_numbers[0], "color_frame_number_last": frame_numbers[-1],
            "frame_number_gaps": int(np.sum(np.maximum(np.diff(frame_numbers)-1, 0))),
            "sensor_timestamps_ms": sensor_times,
            "depth_scale_meters": scale, "depth_valid_fraction": float(np.mean(z16 > 0)),
            "valid_depth_meters": {"min": float(valid.min()), "median": float(np.median(valid)), "max": float(valid.max())} if len(valid) else None,
            "rgb_channel_mean": rgb.mean(axis=(0,1)).tolist(), "rgb_pixel_std": float(rgb.std())})
    finally:
        pipe.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["feedback", "camera"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--serial", default="260322279327")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "feedback":
            feedback(args.output, args.port)
        else:
            camera(args.output, args.serial)
    except BaseException as error:
        dump(args.output/f"{args.mode}-error.json", {"status": "failed", "error": repr(error)})
        if args.mode == "feedback":
            os._exit(1)
        raise


if __name__ == "__main__":
    main()
