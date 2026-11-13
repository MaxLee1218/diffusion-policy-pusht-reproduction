"""Checks for rejecting broken recordings without relabeling source actions."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
spec = importlib.util.spec_from_file_location(
    "prepare_b601", Path(__file__).resolve().parents[1] / "scripts/prepare_b601.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.fixture
def recording(tmp_path):
    folder = tmp_path / "recording"
    (folder / "data/chunk-000").mkdir(parents=True)
    (folder / "meta/episodes/chunk-000").mkdir(parents=True)
    state = np.arange(21, dtype=np.float32).reshape(3, 7)
    action = state + 2
    _, png = cv2.imencode(".png", np.full((480, 640, 3), (20, 80, 170), dtype=np.uint8))
    table = pa.table({
        "observation.state": state.tolist(), "action": action.tolist(),
        "observation.images.front": [{"bytes": png.tobytes(), "path": f"frame-{i}.png"} for i in range(3)],
        "timestamp": [0., .1, .2], "frame_index": [0, 1, 2], "index": [0, 1, 2],
        "episode_index": [0, 0, 0], "task_index": [0, 0, 0]})
    pq.write_table(table, folder / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "length": 3, "dataset_from_index": 0, "dataset_to_index": 3, "tasks": ["pick-place"]
    }]), folder / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0]}), folder / "meta/tasks.parquet")
    (folder / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "total_frames": 3, "fps": 10,
        "features": {key: {"shape": [7], "names": prep.JOINTS}
                     for key in ("action", "observation.state")}}))
    (folder / "meta/stats.json").write_text(json.dumps({
        key: {"min": value.min(0).tolist(), "max": value.max(0).tolist()}
        for key, value in (("action", action), ("observation.state", state))}))
    (folder / "timing.csv").write_text(
        "frame_index,camera_received_monotonic_ns,feedback_read_complete_monotonic_ns,command_sent_complete_monotonic_ns\n"
        "0,1000000000,1001000000,1002000000\n"
        "1,1100000000,1101000000,1102000000\n"
        "2,1200000000,1201000000,1202000000\n")
    return folder, state, action


def folder_spec(folder, kind="recording"):
    files = [p for p in folder.rglob("*") if p.is_file()]
    return {"folder": folder.name, "file_count": len(files), "bytes": sum(p.stat().st_size for p in files),
            "frames": 3, "episodes": 1, "type": kind}


def test_seven_joint_actions_and_rgb_are_preserved(recording, tmp_path):
    folder, state, action = recording
    row, values = prep.audit_recording(tmp_path, folder_spec(folder), tmp_path, 320, 240)
    assert row["included"] and row["status"] == "valid"
    np.testing.assert_array_equal(values["state"], state)
    np.testing.assert_array_equal(values["action"], action)
    np.testing.assert_array_equal(values["img"][0, 0, 0], [170, 80, 20])
    assert row["timing"]["effective_fps"] == 10


def test_pilot_is_audited_and_excluded(recording, tmp_path):
    folder, _, _ = recording
    row, _ = prep.audit_recording(tmp_path, folder_spec(folder, "pilot"), tmp_path, 32, 24)
    assert row["status"] == "valid" and not row["included"]


def test_broken_frame_order_is_rejected(recording, tmp_path):
    folder, _, _ = recording
    path = folder / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.column_names.index("frame_index"), "frame_index", pa.array([0, 2, 1]))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="Nonconsecutive frame_index"):
        prep.audit_recording(tmp_path, folder_spec(folder), tmp_path, 32, 24)


def test_images_without_episode_are_not_recovered_as_actions(tmp_path):
    folder = tmp_path / "interrupted"
    folder.mkdir()
    spec = {"folder": folder.name, "file_count": 0, "bytes": 0, "frames": 0,
            "episodes": 0, "type": "recording"}
    row, data = prep.audit_recording(tmp_path, spec, tmp_path, 32, 24)
    assert row["status"] == "incomplete" and not row["included"] and data is None
