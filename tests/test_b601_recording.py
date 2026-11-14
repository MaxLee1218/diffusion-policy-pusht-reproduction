import json
import sys

import imageio_ffmpeg
import numpy as np
import pytest

from real_robot.recording import ExperimentRecorder, encoder_arguments


def test_lossless_video_depth_and_control_round_trip_and_partial_chunk(tmp_path):
    import av
    output = tmp_path/"recording"
    command = encoder_arguments(imageio_ffmpeg.get_ffmpeg_exe(), output/"rgb-%05d.mkv", fps=4, segment_seconds=1)
    recorder = ExperimentRecorder(output, command, {"depth_unit_m": .0001}, depth_batch=3)
    rng = np.random.default_rng(42)
    originals = []
    depths = []
    for index in range(5):
        rgb = np.full((480, 640, 3), index*30, np.uint8)
        rgb[10:30, 20:40] = rng.integers(0, 256, (20, 20, 3), np.uint8)
        depth = np.full((480, 640), 3000+index, np.uint16)
        recorder.submit_frame(rgb, depth, {"camera_received_ns": 1000+index,
            "color_frame_number": 42+index, "depth_frame_number": 84+index})
        recorder.submit_control({"feedback_read_ns": 2000+index, "motor_positions_deg": [index]*7})
        originals.append(rgb)
        depths.append(depth)
    recorder.stop()
    assert recorder.finalized.wait(20)
    assert recorder.error is None
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["state"] == "complete" and manifest["finalized"]
    assert manifest["rgb_depth_frames_written"] == manifest["rgb_depth_frames_submitted"] == 5
    assert manifest["control_cycles_written"] == manifest["control_cycles_submitted"] == 5
    reconstructed = []
    for segment in sorted(output.glob("rgb-*.mkv")):
        with av.open(str(segment)) as container:
            reconstructed.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
    np.testing.assert_array_equal(reconstructed, originals)
    restored_depth, indices = [], []
    for chunk in sorted((output/"depth").glob("*.npz")):
        with np.load(chunk) as data:
            restored_depth.extend(data["depth"])
            indices.extend(data["record_index"].tolist())
    np.testing.assert_array_equal(restored_depth, depths)
    assert indices == list(range(5))
    assert len((output/"frames.jsonl").read_text().splitlines()) == 5
    assert len((output/"control-cycles.jsonl").read_text().splitlines()) == 5


def test_failed_encoder_is_reported_and_stops_recording(tmp_path):
    recorder = ExperimentRecorder(tmp_path/"recording", [sys.executable, "-c", "raise SystemExit(9)"], {})
    recorder.stop()
    assert recorder.finalized.wait(10)
    assert recorder.error is not None and not recorder.accepting
    manifest = json.loads((tmp_path/"recording/manifest.json").read_text())
    assert manifest["state"] == "failed"


def test_full_prediction_record_preserves_all_steps_and_rejects_incomplete_output():
    import io
    from real_robot.recording import write_prediction
    full = np.arange(112, dtype=np.float32).reshape(16, 7)
    row = {"recording_id": "trial-42", "sample_id": "input-7", "sampling_seed": 101,
           "predicted_action_8": full[1:9].tolist(), "full_prediction_16": full.tolist()}
    stream = io.StringIO()
    write_prediction(stream, row)
    restored = json.loads(stream.getvalue())
    np.testing.assert_array_equal(restored["full_prediction_16"], full)
    assert restored["recording_id"] == "trial-42"
    with pytest.raises(ValueError):
        write_prediction(io.StringIO(), {**row, "full_prediction_16": full[:8].tolist()})


def test_old_stopped_run_cannot_stop_new_recording_after_controller_restart():
    from real_robot.recording import recording_run_finished
    old = {"run_id":"old", "control_generation":1,"status":"stopped"}
    assert not recording_run_finished(old,"new",1)
    current = {**old,"run_id":"new"}
    assert recording_run_finished(current,"new",1)
    assert not recording_run_finished({**current,"status":"loading"},"new",1)
    assert not recording_run_finished(current,"new",2)
