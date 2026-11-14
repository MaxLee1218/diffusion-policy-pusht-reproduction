"""Run a bounded or continuous model session through the sole operator controller."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"upstream/diffusion_policy")]

import dill
import hydra
import numpy as np
import torch

from real_robot.manual_control import JogModel
from real_robot.model_control import bounded_target
from real_robot.recording import write_prediction
from real_robot.shadow import preprocess_pair


def read_json(path):
    for _ in range(20):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(.02)
    raise RuntimeError(f"Cannot read {path}")


class Trial:
    def __init__(self, session, control_generation=None):
        self.session = session
        status = read_json(session/"teleop-status.json")
        if not status.get("model_control_enabled") or not status.get("connected"):
            raise RuntimeError("Model controller is not connected")
        self.pid = status["pid"]
        self.control_generation = status["control_generation"] if control_generation is None else control_generation
        if status["control_generation"] != self.control_generation:
            raise RuntimeError("操作员已暂停或接管，模型运行已取消")

    def request(self, command, **values):
        identifier = uuid.uuid4().hex
        status = read_json(self.session/"teleop-status.json")
        if status.get("pid") != self.pid:
            raise RuntimeError("Controller changed during trial")
        if status.get("control_generation") != self.control_generation:
            raise RuntimeError("操作员已暂停或接管，模型运行已取消")
        request = {"id": identifier, "command": command, "controller_pid": self.pid,
                   "control_generation": self.control_generation,
                   "controller_lease_ns": status["updated_ns"], "issued_unix": time.time(), **values}
        (self.session/"model-request.json").write_text(json.dumps(request, allow_nan=False))
        path = self.session/(identifier+".response.json")
        deadline = time.perf_counter()+20
        while time.perf_counter() < deadline:
            if path.exists():
                response = read_json(path)
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "Controller rejected request"))
                return identifier, response
            time.sleep(.02)
        raise TimeoutError(f"Controller response timed out: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--resume", action="store_true", help="Continue at the current pose without reference return")
    parser.add_argument("--continuous", action="store_true", help="Run until the duration limit or operator pause")
    parser.add_argument("--control-generation", type=int, help="Cancel this run if the operator has paused since launch")
    parser.add_argument("--recording-id", help="Join predictions to the native RGB/depth/control recording")
    parser.add_argument("--rhc", action="store_true", help="10 Hz timestamped sequences, replan every six steps")
    parser.add_argument("--run-id", help="Unique launch identity owned by the operator window")
    parser.add_argument("--hold-validation", action="store_true", help="Infer under full load but schedule a fixed measured pose")
    args = parser.parse_args()
    if args.hold_validation:
        args.rhc, args.resume = True, True
    if (not args.continuous and (not 1 <= args.steps <= 30 or not 0 < args.duration <= 60)
            or args.continuous and not 0 < args.duration <= 300):
        raise ValueError("Short trials allow 30 steps/60 seconds; continuous sessions allow up to 300 seconds")
    candidate = read_json(ROOT/"results/b601-seed42/sampler-benchmark/candidate.json")
    checkpoint = ROOT/candidate["checkpoint"]
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != candidate["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
    result = {"status": "loading", "completed_steps": 0, "checkpoint_sha256": candidate["checkpoint_sha256"],
              "max_arm_step_degrees": .5, "max_gripper_step_degrees": 2., "success_evaluated": False,
              "continuous": args.continuous, "duration_limit_seconds": args.duration,
              "runner_pid": __import__("os").getpid(), "control_generation": args.control_generation}
    result["recording_id"] = args.recording_id
    result["run_id"] = args.run_id or uuid.uuid4().hex
    if args.rhc:
        result.update(max_arm_step_degrees=2., max_gripper_step_degrees=8., trial_type="full_task_rhc")
    if args.hold_validation:
        result.update(trial_type="fixed_pose_timing_validation",hold_validation=True)

    def save():
        (args.session/"model-trial.json").write_text(json.dumps(result, indent=2))
    save()
    try:
        trial = Trial(args.session, args.control_generation)
        model = JogModel(read_json(args.session/"setup.json"))
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        policy = hydra.utils.instantiate(payload["cfg"].policy)
        policy.load_state_dict(payload["state_dicts"][candidate["weights_key"]])
        del payload
        scheduler = hydra.utils.instantiate(candidate["noise_scheduler"])
        torch.testing.assert_close(scheduler.betas, policy.noise_scheduler.betas, rtol=0, atol=0)
        policy.noise_scheduler = scheduler
        policy.num_inference_steps = candidate["num_inference_steps"]
        policy.kwargs = candidate["step_kwargs"]
        if args.rhc:
            # Same runtime horizon override as upstream eval_real_robot.py.
            policy.n_action_steps = policy.horizon-policy.n_obs_steps+1
        policy.to("cuda").eval()
        stats = policy.normalizer.get_input_stats()["joint_pos"]
        low, high = [stats[key].cpu().numpy() for key in ("min", "max")]
        with torch.inference_mode():
            # Warm GPU before selecting any live observations.
            warmup_obs = {"image": torch.zeros((1, 2, 3, 240, 320), device="cuda"),
                          "joint_pos": torch.from_numpy(((low+high)/2)[None, None].repeat(2, axis=1)).to("cuda")}
            policy.predict_action(warmup_obs)
            torch.cuda.synchronize()
            result["status"] = "checking_current_pose" if args.resume else "aligning_reference"
            save()
            if args.resume:
                reference = np.asarray(read_json(args.session/"teleop-status.json")["current_motor_degrees"])
                result["resumed_at_current_pose"] = True
                result["reference_alignment_skipped"] = True
            else:
                setup=read_json(args.session/"setup.json")
                reference_file=Path(setup.get("wsl_reference_file",ROOT/"data/b601/references/b601-start-pose-20261112-180049.json"))
                reference = np.array(list(read_json(reference_file)["observation.state"].values()))
            alignment_started = time.perf_counter()
            for alignment_step in range(100):
                # Allow the 10 Hz camera pair to refresh after the prior motion.
                time.sleep(.25)
                identifier, sample = trial.request("sample")
                q = np.asarray(sample["observed_q"])
                if args.resume or args.hold_validation:
                    break
                if not args.rhc:
                    reference[6] = q[6]
                if np.max(np.abs(reference-q) if args.rhc else np.abs(reference[:6]-q[:6])) < .15:
                    break
                if time.perf_counter()-alignment_started > 120:
                    raise TimeoutError("Reference alignment did not converge")
                target = bounded_target(model, q, reference)
                _, aligned = trial.request("execute", observed_q=sample["observed_q"],
                    observed_ns=sample["observed_ns"], target=target.tolist(), purpose="reference_alignment")
                result.update(alignment_steps=alignment_step+1,
                    alignment_error_deg=float(np.max(np.abs(reference[:6]-np.asarray(aligned["actual_motor_degrees"])[:6]))))
                save()
                if alignment_step % 10 == 0:
                    print(json.dumps({"alignment_step": alignment_step+1,
                                      "remaining_deg": result["alignment_error_deg"]}), flush=True)
            else:
                raise TimeoutError("Reference alignment exhausted its step limit")
            result["aligned_motor_degrees"] = q.tolist()
            result["start_motor_degrees"] = q.tolist()
            if result.get("alignment_steps", 0):
                # Alignment can leave the GPU idle for a minute. Warm it again
                # before taking the fresh observation used for dispatch.
                warmup_started = time.perf_counter()
                policy.predict_action(warmup_obs)
                torch.cuda.synchronize()
                result["post_alignment_warmup_ms"] = (time.perf_counter()-warmup_started)*1000
            result["status"] = "running"
            save()
            if args.rhc:
                from scripts.b601_rhc_loop import run_rhc
                run_rhc(args, trial, model, policy, result, low, high, save)
                return
            began = time.perf_counter()
            with (args.session/"model-trial-events.jsonl").open("a") as log, (args.session/"predictions.jsonl").open("a", buffering=1) as prediction_log:
                for index in range(10000 if args.continuous else args.steps):
                    if time.perf_counter()-began >= args.duration:
                        break
                    time.sleep(.25)
                    identifier, sample = trial.request("sample")
                    with np.load(args.session/(identifier+".npz")) as data:
                        rgb, q = data["rgb"], data["q"]
                    if np.any(q[-1] < low-.05) or np.any(q[-1] > high+.05):
                        raise ValueError("Live pose is outside training range; holding")
                    obs, resized = preprocess_pair(rgb, q.astype(np.float32))
                    torch.manual_seed(101+index)
                    start = time.perf_counter()
                    inference_start_ns = time.perf_counter_ns()
                    output = policy.predict_action({key: torch.from_numpy(value[None]).to("cuda")
                                                     for key, value in obs.items()})
                    prediction = output["action"][0].cpu().numpy()
                    full_prediction = output["action_pred"][0].cpu().numpy()
                    write_prediction(prediction_log, {"run_id": result["run_id"], "recording_id": args.recording_id,
                        "step": index+1, "sample_id": identifier, "sampling_seed": 101+index,
                        "input_observed_windows_ns": sample["observed_ns"],
                        "infer_start_wsl_ns": inference_start_ns, "infer_end_wsl_ns": time.perf_counter_ns(),
                        "predicted_action_8": prediction.tolist(), "full_prediction_16": full_prediction.tolist(),
                        "checkpoint_sha256": candidate["checkpoint_sha256"]})
                    target = bounded_target(model, q[-1], prediction[0])
                    _, completed = trial.request("execute", observed_q=sample["observed_q"],
                        observed_ns=sample["observed_ns"], target=target.tolist(), purpose="policy")
                    row = {"step": index+1, "sample_id": identifier, "observed_motor_degrees": q[-1].tolist(),
                           "run_id": result["run_id"], "recording_id": args.recording_id,
                           "raw_model_target": prediction[0].tolist(), "bounded_target": target.tolist(),
                           "accepted_target_motor_degrees": completed["accepted_target_motor_degrees"],
                           "actual_motor_degrees": completed["actual_motor_degrees"],
                           "inference_and_execute_seconds": time.perf_counter()-start}
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    result.update(completed_steps=index+1, latest=row)
                    save()
                    print(json.dumps({"step": index+1, "actual": completed["actual_motor_degrees"]}), flush=True)
            result["status"] = "complete_holding"
    except BaseException as error:
        result.update(status="stopped", error=(result.get("hardware_fault") or {}).get("reason",repr(error)),
                      caught_exception=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
