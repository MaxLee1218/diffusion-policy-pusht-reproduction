"""The upstream real-robot timing pattern, adapted to B601 motor targets."""

import json
import time

import numpy as np
import torch

from real_robot.recording import write_prediction
from real_robot.receding_horizon import DT_NS, validate_sequence
from real_robot.shadow import preprocess_pair


def run_rhc(args, trial, model, policy, result, low, high, save):
    from scripts.run_b601_model_trial import read_json
    began, next_tick = time.perf_counter(), time.perf_counter()
    result.update(execution_mode="receding_horizon", action_frequency_hz=10,
                  steps_per_inference=6, future_actions_per_prediction=15,
                  completed_inferences=0, wait_for_each_action_arrival=False)
    save()
    try:
        with (args.session/"predictions.jsonl").open("a", buffering=1) as prediction_log, \
                (args.session/"model-trial-events.jsonl").open("a", buffering=1) as events:
            index = 0
            while time.perf_counter()-began < args.duration:
                time.sleep(max(0., min(next_tick, began+args.duration)-time.perf_counter()))
                if time.perf_counter()-began >= args.duration:
                    break
                identifier, sample = trial.request("sample")
                with np.load(args.session/(identifier+".npz")) as data:
                    obs, _ = preprocess_pair(data["rgb"], data["q"].astype(np.float32))
                    q = data["q"][-1].copy()
                if np.any(q < low-.05) or np.any(q > high+.05):
                    raise ValueError("Live pose is outside training range; holding")
                torch.manual_seed(101+index)
                inference_start = time.perf_counter_ns()
                output = policy.predict_action({key:torch.from_numpy(value[None]).to("cuda") for key,value in obs.items()})
                full = output["action_pred"][0].cpu().numpy()
                actions = output["action"][0].cpu().numpy()
                inference_end = time.perf_counter_ns()
                timestamps = sample["observed_ns"]+np.arange(15, dtype=np.int64)*DT_NS
                write_prediction(prediction_log, {"run_id":result["run_id"], "recording_id":args.recording_id,
                    "sample_id":identifier, "sampling_seed":101+index, "inference_index":index,
                    "input_observed_windows_ns":sample["observed_ns"],
                    "infer_start_wsl_ns":inference_start, "infer_end_wsl_ns":inference_end,
                    "predicted_action_8":full[1:9].tolist(), "full_prediction_16":full.tolist(),
                    "rhc_actions_15":actions.tolist(), "timestamps_windows_ns":timestamps.tolist(),
                    "checkpoint_sha256":result["checkpoint_sha256"]})
                validate_sequence(model, q, actions)
                plan_id, accepted = trial.request("execute_sequence", actions=actions.tolist(),
                    timestamps_ns=timestamps.tolist(), observed_ns=sample["observed_ns"], purpose="policy")
                status = read_json(args.session/"teleop-status.json")
                result.update(completed_inferences=index+1, completed_steps=status.get("sequence_actions_sent",0),
                    elapsed_seconds=time.perf_counter()-began, latest_plan_id=plan_id,
                    last_inference_ms=(inference_end-inference_start)/1e6)
                events.write(json.dumps({"type":"rhc_plan", "inference_index":index,
                    "plan_id":plan_id, "sample_id":identifier, "accepted":accepted,
                    "run_id":result["run_id"], "recording_id":args.recording_id})+"\n")
                save()
                print(json.dumps({"inference":index+1, "actions_sent":result["completed_steps"],
                    "accepted_actions":len(accepted["accepted_action_indices"]),
                    "inference_ms":result["last_inference_ms"]}), flush=True)
                index += 1
                next_tick += .6
                if next_tick < time.perf_counter():
                    next_tick = time.perf_counter()
        result.update(status="complete_holding", termination="duration_limit")
    finally:
        try:
            _, stopped = trial.request("stop_sequence")
            result["completed_steps"] = stopped["sequence_actions_sent"]
        except Exception:
            # An operator pause or hardware fault already invalidates this generation.
            status = read_json(args.session/"teleop-status.json")
            result["completed_steps"] = status.get("sequence_actions_sent",0)
        result["elapsed_seconds"] = time.perf_counter()-began
        save()
