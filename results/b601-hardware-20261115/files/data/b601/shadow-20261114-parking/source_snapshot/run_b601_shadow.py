"""Run a bounded Windows-to-WSL B601 predict-only integration session."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"upstream/diffusion_policy")]

import cv2
import dill
import hydra
import numpy as np
import torch
from torch.utils.data import default_collate

from real_robot.shadow import (JOINTS, clock_alignment, map_targets,
                               preprocess_pair, read_packet)
from scripts.benchmark_b601_sampling import check_reference_predictions


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def stats(values):
    return {"count": len(values), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))} if values else None


class BridgeClient:
    def __init__(self, args):
        def windows(path):
            return subprocess.check_output(["wslpath", "-w", str(Path(path).resolve())], text=True).strip()
        self.log = (args.output/"windows-bridge.log").open("wb")
        self.process = subprocess.Popen([
            str(args.windows_python), windows(ROOT/"scripts/b601_windows_bridge.py"),
            "--port", args.port, "--camera-serial", args.camera_serial,
            "--output", windows(args.output)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log)
        self.responses = queue.Queue()

        def receive():
            try:
                while True:
                    self.responses.put(read_packet(self.process.stdout))
            except BaseException as error:
                self.responses.put(error)
        threading.Thread(target=receive, daemon=True).start()

    def receive(self, timeout=20):
        result = self.responses.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise RuntimeError(f"Windows bridge exited: {result}")
        if result[0].get("kind") == "error":
            raise RuntimeError(result[0]["error"])
        return result

    def request(self, command):
        self.process.stdin.write((json.dumps({"command": command})+"\n").encode())
        self.process.stdin.flush()
        return self.receive()

    def stop(self):
        stopped = None
        try:
            if self.process.poll() is None:
                stopped, _ = self.request("stop")
        finally:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.log.close()
        return stopped


def verify_replay(policy, cfg, reference, device):
    """Check actual archived full-resolution frames and the frozen DDIM predictions."""
    import pyarrow.parquet as pq
    protocol = json.loads((reference/"benchmark.json").read_text())
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    indices = protocol["protocol"]["validation_dataset_indices"]
    episodes = json.loads((Path(cfg.task.dataset.zarr_path).parent/"episodes.json").read_text())
    image_checks = []
    for ordinal in (0, 32, 64, 96):
        index = indices[ordinal]
        sample = validation[index]
        start, end, sample_start, _ = validation.sampler.indices[index]
        frames = np.clip(start+np.arange(2)-sample_start, start, end-1)
        episode = int(np.searchsorted(dataset.episode_ends, frames[0], side="right"))
        folder = Path("data/b601")/episodes[episode]["source_folder"]
        parquet = next((folder/"data").rglob("*.parquet"))
        table = pq.read_table(parquet, columns=["observation.images.front"])
        rgb = []
        for frame in frames:
            cell = table.column(0)[int(frame-dataset.episode_starts[episode])].as_py()
            decoded = cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR)
            rgb.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        converted, _ = preprocess_pair(np.stack(rgb), sample["obs"]["joint_pos"].numpy())
        for key in converted:
            np.testing.assert_array_equal(converted[key], sample["obs"][key].numpy())
        image_checks.append({"episode": episode, "validation_index": index, "exact": True})
    batch = default_collate([validation[index] for index in indices[:8]])
    with torch.inference_mode():
        torch.manual_seed(101)
        predicted = policy.predict_action({key: value.to(device) for key, value in batch["obs"].items()})["action"].cpu().numpy()
    with np.load(reference/"predictions.npz") as saved:
        scale = policy.normalizer["action"].params_dict["scale"].detach().cpu().numpy()
        reproduction = check_reference_predictions(predicted[None], saved["ddim_10"][:1, :8],
                                                     saved["target"][:8], scale)
    return {"full_resolution_preprocessing_checks": image_checks,
            "archived_ddim_prediction_reproduction": reproduction}


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"inputs").mkdir()
    candidate = json.loads(args.candidate.read_text())
    checkpoint = Path(candidate["checkpoint"])
    if digest(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the evaluated candidate")
    status = {"status": "loading", "mode": "predict_only", "started_at": datetime.now(timezone.utc).isoformat(),
              "output": str(args.output), "motion_commands_sent": 0, "enable_commands_sent": 0}
    write_json(args.output/"status.json", status)
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"])
    del payload
    scheduler = hydra.utils.instantiate(candidate["noise_scheduler"])
    torch.testing.assert_close(scheduler.betas, policy.noise_scheduler.betas, rtol=0, atol=0)
    policy.noise_scheduler = scheduler
    policy.num_inference_steps = candidate["num_inference_steps"]
    policy.kwargs = candidate["step_kwargs"]
    device = torch.device(args.device)
    policy.to(device).eval()
    replay = verify_replay(policy, cfg, args.reference, device)
    write_json(args.output/"replay-check.json", replay)
    joint_stats = policy.normalizer.get_input_stats()["joint_pos"]
    state_low, state_high = [joint_stats[key].detach().cpu().numpy() for key in ("min", "max")]
    rows, rejects, clock_samples, bridge, stopped = [], [], [], None, None
    error = None
    try:
        bridge = BridgeClient(args)
        hello, _ = bridge.receive()
        if hello.get("kind") != "hello" or hello.get("mode") != "predict_only" or hello["joint_names"] != list(JOINTS):
            raise ValueError("Unexpected bridge handshake")
        write_json(args.output/"bridge-hello.json", hello)
        for _ in range(20):
            began = time.perf_counter_ns()
            pong, _ = bridge.request("ping")
            ended = time.perf_counter_ns()
            clock_samples.append((began, pong["remote_ns"], ended))
        alignment = clock_alignment(clock_samples)
        write_json(args.output/"clock-alignment.json", {**alignment, "samples": clock_samples,
                   "note": "Camera receive and feedback read times use Windows QPC, not exposure or per-motor sample time."})
        status.update(status="running", hardware_connected=True)
        write_json(args.output/"status.json", status)
        began = next_tick = time.perf_counter()
        with (args.output/"predictions.jsonl").open("w") as log, torch.inference_mode():
            while time.perf_counter()-began < args.duration:
                time.sleep(max(0, next_tick-time.perf_counter()))
                if time.perf_counter()-began >= args.duration:
                    break
                request_ns = time.perf_counter_ns()
                header, raw = bridge.request("sample")
                received_ns = time.perf_counter_ns()
                metadata = header["frames"]
                if (header.get("kind") != "sample" or header.get("image_shape") != [2, 480, 640, 3]
                        or len(raw) != 2*480*640*3 or len(metadata) != 2):
                    raise ValueError("Malformed camera/feedback packet")
                timestamps = [frame["camera_received_ns"] for frame in metadata]
                local_camera_ns = timestamps[-1]-alignment["remote_minus_local_ns"]
                pair_gap_ms = (timestamps[1]-timestamps[0])/1e6
                age_ms = (received_ns-local_camera_ns)/1e6
                if (metadata[1]["frame_id"] != metadata[0]["frame_id"]+1 or not 0 < pair_gap_ms <= 250
                        or age_ms-alignment["uncertainty_ns"]/1e6 > 250):
                    rejects.append({"frame_ids": [frame["frame_id"] for frame in metadata],
                                    "pair_gap_ms": pair_gap_ms, "receive_age_estimate_ms": age_ms})
                    next_tick += args.interval
                    continue
                rgb = np.frombuffer(raw, np.uint8).reshape(2, 480, 640, 3)
                state = np.array([frame["joint_pos_motor_degrees"] for frame in metadata], np.float32)
                obs, resized = preprocess_pair(rgb, state)
                seed = 101+len(rows)
                torch.manual_seed(seed)
                torch.cuda.synchronize(device)
                infer_start_ns = time.perf_counter_ns()
                result = policy.predict_action({key: torch.from_numpy(value[None]).to(device) for key, value in obs.items()})
                prediction = result["action"].cpu().numpy()[0]
                torch.cuda.synchronize(device)
                infer_end_ns = time.perf_counter_ns()
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite model predictions")
                mapping = map_targets(prediction, state[-1], hello["driver_config"])
                outside = (state[-1] < state_low) | (state[-1] > state_high)
                row = {"prediction_id": len(rows), "sampling_seed": seed, "frames": metadata,
                       "input_file": f"inputs/{len(rows):04d}.npz", "clock_uncertainty_ms": alignment["uncertainty_ns"]/1e6,
                       "bridge_roundtrip_ms": (received_ns-request_ns)/1e6, "pair_gap_ms": pair_gap_ms,
                       "inference_ms": (infer_end_ns-infer_start_ns)/1e6,
                       "camera_receive_to_prediction_estimate_ms": (infer_end_ns-local_camera_ns)/1e6,
                       "receive_age_estimate_ms": age_ms,
                       "outside_training_state_joints": [name for name, flag in zip(JOINTS, outside) if flag],
                       "predicted_motor_targets": prediction.tolist(), "mapping": mapping,
                       "execution_enabled": False, "motion_commands_sent": 0}
                np.savez_compressed(args.output/row["input_file"], image_rgb=resized, joint_pos=state,
                                    predicted_motor_targets=prediction)
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                rows.append(row)
                cv2.imwrite(str(args.output/"latest-input.png"), cv2.cvtColor(resized[-1], cv2.COLOR_RGB2BGR))
                write_json(args.output/"latest.json", row)
                if len(rows) == 1 or len(rows) % 10 == 0:
                    print(json.dumps({"predictions": len(rows), "inference_ms": row["inference_ms"],
                                      "camera_to_prediction_ms": row["camera_receive_to_prediction_estimate_ms"],
                                      "ood_joints": row["outside_training_state_joints"],
                                      "max_first_action_delta_deg": mapping["max_first_action_delta_motor_degrees"]}), flush=True)
                next_tick += args.interval
                if next_tick < time.perf_counter():
                    next_tick = time.perf_counter()
    except BaseException as caught:
        error = repr(caught)
    finally:
        if bridge is not None:
            try:
                stopped = bridge.stop()
            except BaseException as caught:
                error = error or repr(caught)
        unchanged = digest(checkpoint) == candidate["checkpoint_sha256"]
        summary = {"status": "complete" if error is None and rows and stopped else "failed",
                   "mode": "predict_only", "error": error, "predictions": len(rows),
                   "rejected_stale_pairs": rejects, "bridge_stop": stopped,
                   "checkpoint_sha256": candidate["checkpoint_sha256"], "checkpoint_unchanged": unchanged,
                   "inference_ms": stats([row["inference_ms"] for row in rows]),
                   "bridge_roundtrip_ms": stats([row["bridge_roundtrip_ms"] for row in rows]),
                   "camera_receive_to_prediction_estimate_ms": stats([row["camera_receive_to_prediction_estimate_ms"] for row in rows]),
                   "pair_gap_ms": stats([row["pair_gap_ms"] for row in rows]),
                   "max_first_action_delta_motor_degrees": max((row["mapping"]["max_first_action_delta_motor_degrees"] for row in rows), default=None),
                   "out_of_training_range_counts": {name: sum(name in row["outside_training_state_joints"] for row in rows) for name in JOINTS},
                   "execution_readiness_verified": False, "motion_commands_sent": 0,
                   "limitations": ["Current parked pose/zero correspondence and camera viewpoint are not yet aligned to demonstrations.",
                                   "SDK cached motor states lack per-motor reception timestamps; fresh response for each request is not proven.",
                                   "Camera timestamps are receive times, not exposure times; end-to-end estimates include clock uncertainty.",
                                   "No actuator commands, live execution, task success or measured workspace verification."]}
        write_json(args.output/"summary.json", summary)
        status.update(status=summary["status"], error=error, predictions=len(rows), stopped=stopped,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output/"status.json", status)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if error is not None or not rows:
        raise RuntimeError(error or "No valid live predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-python", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--interval", type=float, default=.8)
    parser.add_argument("--candidate", type=Path, default=ROOT/"results/b601-seed42/sampler-benchmark/candidate.json")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/b601-seed42/sampler-benchmark")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.duration <= 0 or args.interval < .1:
        parser.error("Positive duration and interval >= 0.1 seconds required")
    run(args)


if __name__ == "__main__":
    main()
