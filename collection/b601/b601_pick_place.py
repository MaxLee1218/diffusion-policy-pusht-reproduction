"""Teach a single B601-DM pick-and-place episode from a Mac terminal."""

import argparse
import curses
import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_yaw", "wrist_roll",
)
TRACKING_FIELDS = [(f"{joint}.{field}", f"{joint}_{unit}")
                   for joint in JOINTS
                   for field, unit in (("vel", "velocity_deg_s"), ("torque", "torque_nm"))]
FEATURE_NAMES = [f"{name}.pos" for name in (*JOINTS, "gripper")]
URDF = Path(__file__).resolve().parent / "models/ReBot_Arm_DM.urdf"
FPS = 10
MOVES = {
    "w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0),
    "e": (0, 0, 1), "c": (0, 0, -1),
}
HORIZONTAL_KEYS = ("w", "s", "a", "d")
ARM_STEP = 0.002
FAST_ARM_STEP = 0.006
Z_STEP = 0.004
FAST_Z_STEP = 0.006
WORKSPACE_MARGIN = 0.001
MAX_JOINT_STEP = 2.0
GRIPPER_STEP = 8.0
CAMERA_KEY = "observation.images.front"
TASK = "Pick up the chess piece and place it in the marked square."


class Kinematics:
    def __init__(self, urdf):
        model = ElementTree.parse(urdf).getroot()
        chain = [model.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
        self.origins = []
        self.axes = []
        limits = []
        for joint in chain:
            self.origins.append(self.transform(joint.find("origin")))
            self.axes.append(np.fromstring(joint.find("axis").get("xyz"), sep=" "))
            limit = joint.find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        self.tool = self.transform(model.find("joint[@name='end_joint']/origin"))
        self.limits_deg = np.rad2deg(limits)

    @staticmethod
    def transform(origin):
        result = np.eye(4)
        result[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy"), sep=" ")
        ).as_matrix()
        return result

    def pose(self, joint_deg):
        result = np.eye(4)
        for angle, origin, axis in zip(np.deg2rad(joint_deg), self.origins, self.axes):
            result = result @ origin
            result[:3, :3] = result[:3, :3] @ Rotation.from_rotvec(angle * axis).as_matrix()
        return result @ self.tool

    def move(self, current, translation, orientation, limits):
        target = self.pose(current)[:3, 3] + translation
        local_limits = np.column_stack((
            np.maximum(limits[:, 0], current - MAX_JOINT_STEP),
            np.minimum(limits[:, 1], current + MAX_JOINT_STEP),
        ))
        lower, upper = local_limits.T
        if np.any(lower >= upper):
            return None

        def residual(joints):
            pose = self.pose(joints)
            rot_error = Rotation.from_matrix(orientation.T @ pose[:3, :3]).as_rotvec()
            return np.r_[pose[:3, 3] - target, 0.12 * rot_error]

        result = least_squares(residual, np.clip(current, lower, upper),
                               bounds=(lower, upper), max_nfev=30)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.0003 or np.linalg.norm(error[3:]) > 0.002:
            return None
        return result.x


def open_camera(index):
    import cv2

    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera {index}. Check Camera permissions and the index.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def read_camera(camera):
    import cv2

    ok, frame = camera.read()
    received_ns = time.monotonic_ns()
    if not ok or frame is None:
        raise RuntimeError("Camera frame unavailable.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), received_ns


def create_dataset(root, image):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    vector = {"dtype": "float32", "shape": (7,), "names": FEATURE_NAMES}
    return LeRobotDataset.create(
        repo_id=f"local/{root.name}", root=root, fps=FPS,
        robot_type="seeed_b601_dm_follower", use_videos=False,
        features={
            "observation.state": vector.copy(),
            "action": vector.copy(),
            CAMERA_KEY: {
                "dtype": "image", "shape": image.shape,
                "names": ["height", "width", "channels"],
            },
        },
    )


def add_frame(dataset, image, state, applied):
    dataset.add_frame({
        CAMERA_KEY: image,
        "observation.state": np.asarray(state, dtype=np.float32),
        "action": np.asarray(applied, dtype=np.float32),
        "task": TASK,
    })


def save_episode(dataset, root, timings):
    dataset.save_episode()
    dataset.finalize()
    with (root / "timing.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        header = ["frame_index", "camera_received_monotonic_ns",
                  "feedback_read_complete_monotonic_ns", "command_sent_complete_monotonic_ns"]
        if len(timings[0]) > len(header):
            header.extend(["input_key", *(label for _, label in TRACKING_FIELDS)])
        writer.writerow(header)
        writer.writerows(timings)


def readonly_feedback(port):
    from motorbridge import Controller

    controller = Controller.from_dm_serial(port, 921600)
    values = []
    for motor_id in range(1, 8):
        motor = controller.add_damiao_motor(
            motor_id, motor_id + 0x10, "4340P" if motor_id <= 3 else "4310"
        )
        for _ in range(3):
            motor.request_feedback()
            time.sleep(0.06)
            controller.poll_feedback_once()
            state = motor.get_state()
            if state is not None:
                values.append(math.degrees(state.pos))
                break
        else:
            raise RuntimeError(f"No feedback from motor {motor_id}.")
    return controller, values


def pose_probe(port):
    # Keep the controller alive until process exit so no SDK shutdown command is sent.
    try:
        controller, values = readonly_feedback(port)
        print(json.dumps(values, allow_nan=False), flush=True)
    except Exception as error:
        print(error, file=sys.stderr, flush=True)
        os._exit(1)
    os._exit(0)


def probe_joint_degrees(port):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_feedback", "--port", port],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Motor feedback probe failed.")
    return np.asarray(json.loads(result.stdout), dtype=float)


def motor_limits(model, config):
    limits = np.array([config.joint_limits[name] for name in (*JOINTS, "gripper")])
    limits[:6, 0] = np.maximum(limits[:6, 0], model.limits_deg[:, 0])
    limits[:6, 1] = np.minimum(limits[:6, 1], model.limits_deg[:, 1])
    return limits


def check_pose(state, model, limits, bounds=None):
    outside = [
        f"{key}={value:.2f} deg outside [{low:.2f}, {high:.2f}]"
        for key, value, (low, high) in zip(FEATURE_NAMES, state, limits)
        if not low - 0.1 <= value <= high + 0.1
    ]
    if outside:
        raise RuntimeError("Check B601-DM calibration and physical pose: " + "; ".join(outside))
    xyz = model.pose(state[:6])[:3, 3]
    if bounds is not None and (np.any(xyz < bounds[:, 0]) or np.any(xyz > bounds[:, 1])):
        raise RuntimeError(f"Tool XYZ {xyz} is outside the measured workspace bounds.")
    return xyz


def read_state(robot, tracking=None):
    observation = robot.get_observation()
    if any(robot.motors[name].get_state() is None for name in (*JOINTS, "gripper")):
        raise RuntimeError("Incomplete motor feedback.")
    state = np.array([observation[name] for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(state).all():
        raise RuntimeError("Non-finite motor feedback.")
    if tracking is not None:
        for feature, _ in TRACKING_FIELDS:
            value = observation.get(feature)
            tracking.append(value if value is not None and math.isfinite(value) else None)
    return state


def keyboard_loop(screen, robot, camera, model, bounds, limits, root):
    screen.nodelay(True)
    def read_key():
        try:
            key = screen.getkey().lower()
            curses.flushinp()
            return key
        except curses.error:
            return ""

    def render(view):
        lines = [
            "W/S: X   A/D: Y   E/C: Z   O/P: open/close gripper",
            f"F: XY {view['xy_mm']:g} mm / Z {view['z_mm']:g} mm   R: record   Enter: save   Q/Esc: exit",
            f"Tool XYZ (m) {view['xyz'][0]:+.3f} {view['xyz'][1]:+.3f} {view['xyz'][2]:+.3f}",
            f"Gripper motor {view['gripper']:+.1f} deg    Samples {view['samples']}",
            view["message"],
        ]
        screen.erase()
        height, width = screen.getmaxyx()
        for index, line in enumerate(lines[:height - 1]):
            screen.addnstr(index, 0, line, width - 1)
        screen.refresh()

    return control_loop(read_key, render, robot, camera, model, bounds, limits, root)


def control_loop(read_key, render, robot, camera, model, bounds, limits, root,
                 next_output=None, horizontal_input=nullcontext):
    state = read_state(robot)
    orientation = model.pose(state[:6])[:3, :3]
    applied = state.copy()
    image, _ = read_camera(camera)
    dataset = None
    timings = []
    recording = False
    saved = False
    saved_samples = 0
    last_saved = None
    message = "Ready. R starts recording after positioning."
    arm_step = ARM_STEP
    active_press = None
    margin = np.minimum(WORKSPACE_MARGIN, (bounds[:, 1] - bounds[:, 0]) / 4)
    inner_lower = bounds[:, 0] + margin
    inner_upper = bounds[:, 1] - margin
    while True:
        tick = time.monotonic()
        key = read_key()
        if key in ("q", "\x1b"):
            return saved_samples
        record_key = key
        if key == "f":
            arm_step = FAST_ARM_STEP if arm_step == ARM_STEP else ARM_STEP
            message = "Fast steps selected." if arm_step == FAST_ARM_STEP else "Fine steps selected."
        image, camera_ns = read_camera(camera)
        tracking = [] if recording or record_key == "r" else None
        state = read_state(robot, tracking)
        feedback_ns = time.monotonic_ns()
        xyz = check_pose(state, model, limits, bounds)
        command_ns = None
        requested_step = None
        goal = applied.copy()
        fast = arm_step == FAST_ARM_STEP
        if fast and horizontal_input is not nullcontext and key in HORIZONTAL_KEYS:
            key = ""
        with horizontal_input() as press:
            press = press if fast and not key else None
        if press is not None:
            key = press[0]
        if key in MOVES and (active_press is None or active_press is press):
            direction = np.asarray(MOVES[key])
            # Near an edge, keep the current position available so the arm can move inward.
            lower = np.minimum(inner_lower, xyz)
            upper = np.maximum(inner_upper, xyz)
            if np.max(np.abs(state[:6] - applied[:6])) > MAX_JOINT_STEP:
                message = "Waiting for arm to reach the previous target."
            else:
                if key in ("e", "c"):
                    steps = (FAST_Z_STEP, Z_STEP) if fast else (Z_STEP,)
                    steps += (ARM_STEP, 0.001, 0.0005)
                else:
                    steps = (FAST_ARM_STEP, 0.004, ARM_STEP) if fast else (ARM_STEP,)
                # Advance the previous target when feasible. Solve from measured joints
                # so the same relative joint limit applies even if the arm is lagging.
                reference = model.pose(applied[:6])[:3, 3] if active_press is not None else None
                origins = (reference, xyz) if reference is not None else (xyz,)
                for origin in origins:
                    for step in steps:
                        destination = origin + step * direction
                        if np.any(destination < lower) or np.any(destination > upper):
                            message = "Near workspace boundary. Move inward."
                            continue
                        if reference is not None and (destination - reference) @ direction <= 0.0003:
                            continue
                        solution = model.move(state[:6], destination - xyz, orientation, limits[:6])
                        if solution is None:
                            message = "No nearby solution with the current wrist orientation."
                            continue
                        solved_xyz = model.pose(solution)[:3, 3]
                        if np.any(solved_xyz < lower) or np.any(solved_xyz > upper):
                            message = "Solved target is too close to the workspace boundary."
                            continue
                        goal[:6] = solution
                        requested_step = step
                        break
                    if requested_step is not None:
                        break
        if key in ("o", "p"):
            low, high = robot.config.joint_limits["gripper"]
            goal[6] = np.clip(state[6] + (-1 if key == "o" else 1) * GRIPPER_STEP, low, high)

        # Recheck after IK. Serialize release with dispatch so a cancelled press
        # cannot send a newly solved target after the GUI has cleared it.
        with horizontal_input() as current_press:
            stopping_jog = active_press is not None and (
                active_press is not press or press is not current_press
            )
            if stopping_jog:
                goal = applied.copy()
                goal[:6] = np.clip(state[:6], limits[:6, 0], limits[:6, 1])
                check_pose(goal, model, limits, bounds)
                active_press = None
                requested_step = None
                key = "horizontal_release"
                message = "Horizontal jog released. Holding position."
            elif press is not None and press is not current_press:
                goal = applied
                key = ""
            if not np.array_equal(applied, goal):
                # Driver input has leader scaling. Feedback and returned goals are motor degrees.
                action = {
                    feature: float(value / robot.config.joint_directions[name])
                    for name, feature, value in zip((*JOINTS, "gripper"), FEATURE_NAMES, goal)
                }
                sent = robot.send_action(action)
                applied = np.asarray([sent[feature] for feature in FEATURE_NAMES], dtype=float)
                command_ns = time.monotonic_ns()
                if not stopping_jog:
                    if press is not None:
                        active_press = press
                    if requested_step is not None:
                        axis = "Z" if key in ("e", "c") else "X" if key in ("w", "s") else "Y"
                        message = f"{key.upper()}: {requested_step * 1000:g} mm {axis} target sent."
                    else:
                        message = "Target sent."

        # Finish any pending hold before dataset initialization or saving blocks.
        if record_key == "r" and not recording and (not saved or next_output):
            if saved:
                root = next_output()
                timings = []
                saved = False
            dataset = create_dataset(root, image)
            recording = True
            message = "Recording."
        if record_key in ("\n", "\r", "key_enter") and recording and timings:
            save_episode(dataset, root, timings)
            recording, saved = False, True
            saved_samples = len(timings)
            last_saved = str(root)
            message = f"Saved {saved_samples} samples."
        if recording:
            add_frame(dataset, image, state, applied)
            timings.append((len(timings), camera_ns, feedback_ns, command_ns, key, *tracking))

        render(dict(image=image, xyz=xyz, xy_mm=arm_step * 1000,
                    z_mm=(FAST_Z_STEP if arm_step == FAST_ARM_STEP else Z_STEP) * 1000,
                    gripper=state[6], samples=len(timings), recording=recording,
                    saved_path=last_saved, message=message))
        time.sleep(max(0, 1 / FPS - (time.monotonic() - tick)))


class LimitedArmMotor:
    def __init__(self, motor):
        self.motor = motor

    def __getattr__(self, name):
        return getattr(self.motor, name)

    def send_pos_vel(self, position, velocity):
        # Seeed 1.0.5 hardcodes the arm speed at the call site.
        self.motor.send_pos_vel(position, min(abs(velocity), math.radians(30.0)))


def run_hardware(args, model, confirm=None, run_controls=None):
    from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
    from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower

    class TeachFollower(SeeedB601DMFollower):
        def _add_motors_to_bus(self):
            super()._add_motors_to_bus()
            for name in JOINTS:
                self.motors[name] = LimitedArmMotor(self.motors[name])

        def safe_zero(self, *args, **kwargs):
            # A generic reset trajectory is unsuitable around a tabletop.
            return None

    if args.output.exists():
        raise FileExistsError(args.output)
    bounds = np.asarray(args.bounds).reshape(3, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Use finite, increasing min/max bounds for each axis.")
    config = SeeedB601DMFollowerConfig(
        id="b601-dm-chess", port=args.port, can_adapter="damiao",
        max_relative_target=dict.fromkeys(JOINTS, MAX_JOINT_STEP) | {"gripper": GRIPPER_STEP},
        pos_vel_velocity=12.0, disable_torque_on_disconnect=True,
    )
    limits = motor_limits(model, config)
    check_pose(probe_joint_degrees(args.port), model, limits, bounds)
    camera = open_camera(args.camera)
    try:
        image, _ = read_camera(camera)
        if confirm is None:
            print("Live session enables the motors and disables torque on exit.")
            print("Confirm the factory/model zero and gripper direction, clear the workcell,")
            print("and support the arm as prescribed by its operator before enabling or exiting.")
            confirmed = input("Type ENABLE to proceed: ").strip() == "ENABLE"
        else:
            confirmed = confirm(image)
        if not confirmed:
            return
        if confirm is not None:
            check_pose(probe_joint_degrees(args.port), model, limits, bounds)
        robot = TeachFollower(config)
        try:
            robot.connect(calibrate=False)
            check_pose(read_state(robot), model, limits, bounds)
            if run_controls is None:
                saved_samples = curses.wrapper(
                    keyboard_loop, robot, camera, model, bounds, limits, args.output
                )
            else:
                return run_controls(robot, camera, model, bounds, limits, args.output)
        finally:
            if robot.is_connected:
                robot.disconnect()
        if saved_samples:
            print(f"Saved one episode with {saved_samples} samples to {args.output}", flush=True)
        else:
            print("No episode was saved. R starts recording and Enter saves it.", flush=True)
    finally:
        camera.release()


def smoke(root, model):
    # Bent, nonsingular model pose. This path does not construct a hardware driver.
    joints = np.array([12., -40., -65., 20., 15., 10.])
    pose = model.pose(joints)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    state = np.r_[joints, -40.]
    dataset = create_dataset(root, image)
    timings = []
    for direction in MOVES.values():
        solved = model.move(joints, ARM_STEP * np.asarray(direction),
                            pose[:3, :3], model.limits_deg)
        if solved is None:
            raise RuntimeError("Synthetic Cartesian move failed.")
        image = np.roll(image + 20, 1, axis=1)
        add_frame(dataset, image, state, np.r_[solved, -48.])
        timings.append((len(timings), 0, 0, None))
    save_episode(dataset, root, timings)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recorded = LeRobotDataset(f"local/{root.name}", root=root)
    if recorded.num_episodes != 1 or len(recorded) != len(MOVES):
        raise RuntimeError("Synthetic episode could not be read back.")
    print(f"Verified 1 local episode with {len(recorded)} samples: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=URDF)
    subcommands = parser.add_subparsers(dest="command", required=True)
    probe = subcommands.add_parser("camera")
    probe.add_argument("--camera", type=int, required=True)
    probe.add_argument("--output", type=Path, required=True)
    pose = subcommands.add_parser("pose", help="Read joint feedback and compute model tool XYZ.")
    pose.add_argument("--port", required=True)
    feedback = subcommands.add_parser("_feedback", help=argparse.SUPPRESS)
    feedback.add_argument("--port", required=True)
    simulation = subcommands.add_parser("smoke")
    simulation.add_argument("--output", type=Path, required=True)
    live = subcommands.add_parser("record")
    live.add_argument("--port", required=True)
    live.add_argument("--camera", type=int, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--bounds", type=float, nargs=6, required=True,
                      metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    args = parser.parse_args()
    if args.command == "camera":
        import cv2

        camera = open_camera(args.camera)
        try:
            deadline = time.monotonic() + 1.0
            image, _ = read_camera(camera)
            while time.monotonic() < deadline:
                image, _ = read_camera(camera)
            if not cv2.imwrite(str(args.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save preview: {args.output}")
            print(f"Camera {args.camera}: {image.shape}; saved {args.output}")
        finally:
            camera.release()
    elif args.command == "_feedback":
        pose_probe(args.port)
    elif args.command == "pose":
        from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig

        model = Kinematics(args.urdf)
        config = SeeedB601DMFollowerConfig(port=args.port, can_adapter="damiao")
        state = probe_joint_degrees(args.port)
        for feature, value in zip(FEATURE_NAMES, state):
            print(f"{feature}: {value:+.3f} deg")
        print("Model tool XYZ (m):", model.pose(state[:6])[:3, 3])
        try:
            check_pose(state, model, motor_limits(model, config))
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
        print("Motor positions are inside the model and driver joint limits.")
    elif args.command == "smoke":
        smoke(args.output, Kinematics(args.urdf))
    else:
        run_hardware(args, Kinematics(args.urdf))


if __name__ == "__main__":
    main()
