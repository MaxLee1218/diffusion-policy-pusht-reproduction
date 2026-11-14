import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest

from real_robot.shadow import (JOINTS, MAGIC, MAX_PAYLOAD, PREFIX, clock_alignment,
                               driver_config, map_targets, preprocess_pair,
                               read_packet, remote_to_local, write_packet)


class Fragmented(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))

    def write(self, value):
        return super().write(value[:5])


def test_binary_protocol_survives_partial_reads_and_writes():
    stream = Fragmented()
    payload = bytes(range(256))*5
    write_packet(stream, {"kind": "sample", "frames": [1, 2]}, payload)
    stream.seek(0)
    header, received = read_packet(stream)
    assert header == {"kind": "sample", "frames": [1, 2]} and received == payload


def test_protocol_rejects_truncation_bad_magic_and_unbounded_payloads():
    stream = io.BytesIO()
    write_packet(stream, {"kind": "sample"}, b"pixels")
    with pytest.raises(EOFError):
        read_packet(io.BytesIO(stream.getvalue()[:-1]))
    for prefix in (PREFIX.pack(b"BADMAGIC", 2, 0), PREFIX.pack(MAGIC, 2, MAX_PAYLOAD+1)):
        with pytest.raises(ValueError):
            read_packet(io.BytesIO(prefix))


def config_file(tmp_path, malicious=False):
    values = {"motor_can_ids": {name: (i, i+16) for i, name in enumerate(JOINTS, 1)},
              "joint_limits": {name: (-200., 200.) for name in JOINTS},
              "joint_directions": dict(zip(JOINTS, [-1., -1., 1., 1., 1., -1., -6.]))}
    text = "class SeeedB601DMFollowerConfig:\n"
    for key, value in values.items():
        expression = "__import__('os').abort()" if malicious and key == "joint_directions" else repr(value)
        text += f"    {key}: dict = field(default_factory=lambda: {expression})\n"
    path = tmp_path/"driver.py"
    path.write_text(text)
    return path


def test_configuration_is_read_without_importing_or_executing_driver(tmp_path):
    result = driver_config(config_file(tmp_path))
    assert result["joint_directions"]["gripper"] == -6.
    with pytest.raises((ValueError, TypeError)):
        driver_config(config_file(tmp_path, malicious=True))


def test_absolute_motor_targets_use_inverse_driver_scale_not_state_delta(tmp_path):
    config = driver_config(config_file(tmp_path))
    config["joint_limits"]["gripper"] = (-100., 0.)
    action = np.tile([10., -30., -60., 50., -4., -20., -120.], (8, 1))
    result = map_targets(action, np.zeros(7), config)
    proposed = np.array(result["proposed_driver_input"])
    np.testing.assert_array_equal(proposed[0], [-10., 30., -60., 50., -4., 20., 20.])
    assert result["joint_limit_projected_motor_targets"][0][-1] == -100.
    assert result["joint_limit_clip_max_motor_degrees"] == 20.
    assert result["roundtrip_max_error_motor_degrees"] == 0.
    assert result["sent_to_robot"] is False
    with pytest.raises(ValueError):
        map_targets(action*float("nan"), np.zeros(7), config)


def test_live_preprocessing_preserves_rgb_order_temporal_order_and_unclipped_state():
    rgb = np.empty((2, 480, 640, 3), dtype=np.uint8)
    rgb[0] = [220, 80, 10]
    rgb[1] = [10, 40, 180]
    state = np.arange(14).reshape(2, 7)+1000
    obs, resized = preprocess_pair(rgb, state)
    assert obs["image"].shape == (2, 3, 240, 320)
    np.testing.assert_allclose(obs["image"][:, :, 0, 0], np.array([[220, 80, 10], [10, 40, 180]])/255)
    np.testing.assert_array_equal(obs["joint_pos"], state.astype(np.float32))
    assert resized.dtype == np.uint8
    with pytest.raises(ValueError):
        preprocess_pair(rgb.astype(np.float32), state)


def test_clock_domains_are_aligned_with_lowest_roundtrip_and_uncertainty_bound():
    alignment = clock_alignment([(100, 1120, 140), (200, 1205, 210), (300, 1310, 330)])
    assert alignment["remote_minus_local_ns"] == 1000
    assert alignment["uncertainty_ns"] == 5
    assert 1250-alignment["remote_minus_local_ns"] == 250
    with pytest.raises(ValueError):
        clock_alignment([(100, 1100, 90)])


def test_clock_interpolation_handles_different_clock_rates_without_extrapolation():
    points = [(90, 1000, 110), (190, 1200, 210), (290, 1400, 310)]
    assert remote_to_local(1100, points) == 150
    assert remote_to_local(1300, points) == 250
    with pytest.raises(ValueError):
        remote_to_local(1500, points)


def test_windows_worker_has_no_actuation_or_controller_context_manager_calls():
    source = Path(__file__).resolve().parents[1]/"scripts/b601_windows_bridge.py"
    tree = ast.parse(source.read_text())
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    forbidden = {"enable", "enable_all", "disable", "disable_all", "send_action", "send_mit",
                 "send_force_pos", "send_pos_vel", "send_vel", "ensure_mode", "set_zero_position",
                 "store_parameters", "shutdown", "close_bus", "write_register_u32", "write_register_f32"}
    assert not calls & forbidden
    assert {"from_dm_serial", "add_damiao_motor", "request_feedback", "poll_feedback_once", "get_state"} <= calls
    # No driver import: only AST-read literals, and SDK construction without a with-block.
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            assert all("controller" not in ast.unparse(item.context_expr).lower() for item in node.items)
