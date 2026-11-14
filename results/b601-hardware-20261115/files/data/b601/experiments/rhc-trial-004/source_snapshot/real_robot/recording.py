"""Lossless RGB/depth and control-cycle recording, off the control thread."""

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np


def encoder_arguments(executable, output, fps=30, segment_seconds=5):
    return [str(executable), "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
            "-g", str(fps), "-f", "segment", "-segment_time", str(segment_seconds),
            "-reset_timestamps", "1", str(output)]


def write_prediction(stream, row):
    actions = np.asarray(row["predicted_action_8"])
    full = np.asarray(row["full_prediction_16"])
    if (actions.shape != (8, 7) or full.shape != (16, 7) or not np.isfinite(full).all()
            or not np.array_equal(actions, full[1:9])):
        raise ValueError("Prediction must contain the complete matching 16-step and executable 8-step sequences")
    stream.write(json.dumps(row, allow_nan=False)+"\n")
    stream.flush()


def recording_run_finished(progress, run_id, generation):
    return (progress.get("run_id") == run_id and progress.get("control_generation") == generation
            and progress.get("status") in ("stopped", "complete_holding"))


class ExperimentRecorder:
    def __init__(self, directory, command, camera_metadata, *, creationflags=0, depth_batch=15):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/"depth").mkdir()
        self.frame_queue, self.control_queue = queue.Queue(60), queue.Queue(400)
        self.stop_event, self.finalized = threading.Event(), threading.Event()
        self.accepting, self.error = True, None
        self.frames_submitted = self.frames_written = self.controls_submitted = self.controls_written = 0
        self.depth_chunks = 0
        self.depth_batch = depth_batch
        self.camera_metadata = camera_metadata
        self.started_unix_ns, self.started_ns = time.time_ns(), time.perf_counter_ns()
        self.encoder_log = (self.directory/"encoder.log").open("wb")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.encoder_log, creationflags=creationflags)
        except BaseException:
            self.encoder_log.close()
            raise
        self._manifest("recording")
        self.camera_thread = threading.Thread(target=self._write_camera, daemon=True)
        self.control_thread = threading.Thread(target=self._write_controls, daemon=True)
        self.camera_thread.start()
        self.control_thread.start()
        threading.Thread(target=self._finalize, daemon=True).start()

    def stats(self):
        return {"directory": str(self.directory), "accepting": self.accepting, "error": self.error,
                "finalized": self.finalized.is_set(), "rgb_depth_frames_submitted": self.frames_submitted,
                "rgb_depth_frames_written": self.frames_written, "control_cycles_submitted": self.controls_submitted,
                "control_cycles_written": self.controls_written, "depth_chunks_written": self.depth_chunks}

    def _manifest(self, state):
        value = {"state": state, **self.stats(), "finalized": state != "recording", "camera": self.camera_metadata,
            "started_unix_ns": self.started_unix_ns, "started_windows_qpc_ns": self.started_ns,
            "clock_domain": "Windows perf_counter_ns for camera/control; sensor clocks also recorded",
            "rgb": "lossless libx264rgb CRF 0, 640x480 RGB24, 30 fps, 5-second MKV segments",
            "depth": "lossless NPZ uint16 sensor Z16, 15 frames per chunk, units in camera metadata",
            "feedback_freshness": "SDK cache lacks per-motor reception timestamps; explicit requests/polls recorded"}
        (self.directory/"manifest.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    def _fail(self, error):
        if self.error is None:
            self.error = repr(error)
        self.accepting = False
        self.stop_event.set()

    def submit_frame(self, rgb, depth, metadata):
        if not self.accepting:
            return
        try:
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or depth.shape != (480, 640) or depth.dtype != np.uint16:
                raise ValueError("Unexpected RGB/depth shape or dtype")
            self.frame_queue.put_nowait((rgb, depth, dict(metadata)))
            self.frames_submitted += 1
        except Exception as error:
            self._fail(error)

    def submit_control(self, row):
        if not self.accepting:
            return
        try:
            self.control_queue.put_nowait(row)
            self.controls_submitted += 1
        except Exception as error:
            self._fail(error)

    def stop(self):
        self.accepting = False
        self.stop_event.set()

    def _write_depth(self, batch):
        if not batch:
            return
        np.savez_compressed(self.directory/"depth"/f"{self.depth_chunks:06d}.npz",
            depth=np.stack([item[0] for item in batch]),
            record_index=np.array([item[1]["record_index"] for item in batch], np.int64),
            camera_received_ns=np.array([item[1]["camera_received_ns"] for item in batch], np.int64),
            color_frame_number=np.array([item[1]["color_frame_number"] for item in batch], np.int64),
            depth_frame_number=np.array([item[1]["depth_frame_number"] for item in batch], np.int64))
        self.depth_chunks += 1
        batch.clear()

    def _write_camera(self):
        batch = []
        try:
            with (self.directory/"frames.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                while not self.stop_event.is_set() or not self.frame_queue.empty():
                    try:
                        rgb, depth, metadata = self.frame_queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    metadata["record_index"] = self.frames_written
                    remaining = memoryview(rgb).cast("B")
                    while remaining:
                        written = self.process.stdin.write(remaining)
                        if written is None or written <= 0:
                            raise OSError("Video encoder stopped accepting frames")
                        remaining = remaining[written:]
                    log.write(json.dumps(metadata, allow_nan=False)+"\n")
                    batch.append((depth, metadata))
                    self.frames_written += 1
                    if len(batch) >= self.depth_batch:
                        self._write_depth(batch)
        except BaseException as error:
            self._fail(error)
        finally:
            try:
                self._write_depth(batch)
                self.process.stdin.close()
                if self.process.wait(timeout=15) != 0:
                    raise RuntimeError("Video encoder failed; inspect encoder.log")
            except BaseException as error:
                self._fail(error)
                if self.process.poll() is None:
                    self.process.terminate()
            self.encoder_log.close()

    def _write_controls(self):
        try:
            with (self.directory/"control-cycles.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                while not self.stop_event.is_set() or not self.control_queue.empty():
                    try:
                        row = self.control_queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    log.write(json.dumps(row, allow_nan=False)+"\n")
                    self.controls_written += 1
        except BaseException as error:
            self._fail(error)

    def _finalize(self):
        self.camera_thread.join()
        self.control_thread.join()
        try:
            self._manifest("failed" if self.error else "complete")
        except BaseException as error:
            self._fail(error)
        finally:
            self.finalized.set()
