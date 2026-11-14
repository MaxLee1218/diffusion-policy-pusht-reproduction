"""Framing, observation conversion and diagnostic mapping for read-only B601 runs."""

import ast
import json
from pathlib import Path
import struct

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_yaw", "wrist_roll", "gripper")
MAGIC = b"B6SHDW1\n"
PREFIX = struct.Struct("!8sII")
MAX_HEADER, MAX_PAYLOAD = 65536, 4*1024*1024


def read_exact(stream, size):
    pieces = bytearray()
    while len(pieces) < size:
        chunk = stream.read(size-len(pieces))
        if not chunk:
            raise EOFError("Bridge ended before a complete packet arrived")
        pieces.extend(chunk)
    return bytes(pieces)


def write_packet(stream, header, payload=b""):
    encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError("Bridge packet exceeds size limits")
    for piece in (PREFIX.pack(MAGIC, len(encoded), len(payload)), encoded, payload):
        remaining = memoryview(piece)
        while remaining:
            written = stream.write(remaining)
            if written is None or written <= 0:
                raise OSError("Bridge output stopped accepting bytes")
            remaining = remaining[written:]
    stream.flush()


def read_packet(stream):
    magic, header_size, payload_size = PREFIX.unpack(read_exact(stream, PREFIX.size))
    if magic != MAGIC or not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError("Invalid bridge packet prefix")
    header = json.loads(read_exact(stream, header_size))
    if not isinstance(header, dict):
        raise ValueError("Bridge header must be an object")
    return header, read_exact(stream, payload_size)


def driver_config(source):
    """Read literal configuration without importing the actuation-capable driver."""
    tree = ast.parse(Path(source).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "SeeedB601DMFollowerConfig")
    result = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and node.target.id in (
                "motor_can_ids", "joint_limits", "joint_directions"):
            factory = next(keyword.value for keyword in node.value.keywords
                           if keyword.arg == "default_factory")
            result[node.target.id] = ast.literal_eval(factory.body)
    for field in ("motor_can_ids", "joint_limits", "joint_directions"):
        if set(result[field]) != set(JOINTS):
            raise ValueError(f"Driver {field} does not cover all seven joints")
    directions = np.array([result["joint_directions"][name] for name in JOINTS])
    limits = np.array([result["joint_limits"][name] for name in JOINTS])
    if (not np.isfinite(directions).all() or np.any(directions == 0) or
            not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError("Invalid driver directions or limits")
    return result


def preprocess_pair(rgb, state):
    import cv2
    rgb, state = np.asarray(rgb), np.asarray(state, dtype=np.float32)
    if rgb.dtype != np.uint8 or rgb.shape != (2, 480, 640, 3):
        raise ValueError("Expected two 640x480 RGB8 camera frames")
    if state.shape != (2, 7) or not np.isfinite(state).all():
        raise ValueError("Expected two finite seven-joint state vectors")
    resized = np.stack([cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
                        for image in rgb])
    image = np.ascontiguousarray(resized.transpose(0, 3, 1, 2), dtype=np.float32)/255.
    return {"image": image, "joint_pos": state}, resized


def map_targets(action, state, config):
    action, state = np.asarray(action), np.asarray(state)
    if action.shape != (8, 7) or state.shape != (7,) or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Invalid action or measured state")
    direction = np.array([config["joint_directions"][name] for name in JOINTS])
    limits = np.array([config["joint_limits"][name] for name in JOINTS])
    proposed_input = action/direction
    mapped = proposed_input*direction
    clipped = np.clip(mapped, limits[:, 0], limits[:, 1])
    return {"proposed_driver_input": proposed_input.tolist(),
            "roundtrip_max_error_motor_degrees": float(np.max(np.abs(mapped-action))),
            "joint_limit_projected_motor_targets": clipped.tolist(),
            "joint_limit_clip_max_motor_degrees": float(np.max(np.abs(clipped-action))),
            "first_action_delta_from_measured_motor_degrees": (clipped[0]-state).tolist(),
            "max_first_action_delta_motor_degrees": float(np.max(np.abs(clipped[0]-state))),
            "relative_target_cap_applied": False, "sent_to_robot": False}


def clock_alignment(samples):
    """Remote-minus-local clock offset with a round-trip uncertainty bound."""
    if not samples or any(end < start for start, remote, end in samples):
        raise ValueError("Invalid clock calibration exchanges")
    start, remote, end = min(samples, key=lambda row: row[2]-row[0])
    return {"remote_minus_local_ns": remote-(start+end)//2,
            "uncertainty_ns": (end-start+1)//2,
            "minimum_roundtrip_ns": end-start, "exchanges": len(samples)}


def remote_to_local(remote_ns, points):
    """Interpolate bracketed calibration points; do not assume equal clock rates."""
    remote = np.array([row[1]-points[0][1] for row in points], dtype=np.float64)
    origin = (points[0][0]+points[0][2])//2
    local = np.array([(row[0]+row[2])//2-origin for row in points], dtype=np.float64)
    value = remote_ns-points[0][1]
    if len(points) < 2 or np.any(np.diff(remote) <= 0) or not remote[0] <= value <= remote[-1]:
        raise ValueError("Timestamp must be bracketed by increasing calibration points")
    return origin+float(np.interp(value, remote, local))
