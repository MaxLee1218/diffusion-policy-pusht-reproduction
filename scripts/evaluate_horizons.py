"""Compare action execution lengths with one trusted image PushT checkpoint."""

import argparse
import csv
import hashlib
import json
from contextlib import closing
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from tqdm import tqdm

from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from evaluate import load_policy


class EvaluationPushT(PushTImageEnv):
    def __init__(self, runner, disturbance_step, delta_v):
        super().__init__(
            legacy=runner.get("legacy_test", False),
            render_size=runner.get("render_size", 96),
        )
        self.disturbance_step = disturbance_step
        self.delta_v = delta_v

    def reset(self):
        self.coverage = []
        self.success = False
        self.disturbed = False
        return super().reset()

    def step(self, action):
        if len(self.coverage) + 1 == self.disturbance_step:
            # A fixed world-X impulse at the center of gravity changes velocity
            # without teleporting the block through walls or the pusher.
            self.block.apply_impulse_at_world_point(
                (self.block.mass * self.delta_v, 0),
                self.block.local_to_world(self.block.center_of_gravity),
            )
            self.disturbed = True

        obs, score, self.success, info = super().step(action)
        coverage = float(score) * self.success_threshold
        if score == 1.0:
            goal = pymunk_to_shapely(self._get_goal_pose_body(self.goal_pose), self.block.shapes)
            block = pymunk_to_shapely(self.block, self.block.shapes)
            coverage = goal.intersection(block).area / goal.area
        self.coverage.append(coverage)
        return obs, score, self.success, info


def evaluate_episode(policy, runner, horizon, seed, sampling_seed, max_steps,
                     disturbance_step, delta_v, video_path):
    env = EvaluationPushT(runner, disturbance_step, delta_v)
    fps = runner.get("fps", 10)
    recorder = VideoRecordingWrapper(
        env,
        video_recoder=VideoRecorder.create_h264(
            fps=fps, codec="h264", input_pix_fmt="rgb24", crf=runner.get("crf", 22),
            thread_type="FRAME", thread_count=1,
        ),
        file_path=str(video_path) if video_path is not None else None,
        steps_per_render=max(env.control_hz // fps, 1),
    )
    wrapper = MultiStepWrapper(
        recorder, n_obs_steps=policy.n_obs_steps, n_action_steps=horizon,
        max_episode_steps=max_steps,
    )
    with closing(wrapper):
        env.seed(seed)
        torch.manual_seed(sampling_seed)
        obs = wrapper.reset()
        policy.reset()
        done = False
        inference_calls = 0
        inference_seconds = 0.0
        try:
            while not done:
                start = perf_counter()
                with torch.no_grad():
                    batch = {
                        key: torch.from_numpy(value[None]).to(policy.device)
                        for key, value in obs.items()
                    }
                    actions = policy.predict_action(batch)["action"][0].cpu().numpy()
                inference_seconds += perf_counter() - start
                inference_calls += 1
                obs, _, done, _ = wrapper.step(actions)
        finally:
            recorder.render()

    steps = len(env.coverage)
    return {
        "max_normalized_coverage_score": float(max(wrapper.reward)),
        "max_coverage": max(env.coverage),
        "final_coverage": env.coverage[-1],
        "success": bool(env.success),
        "control_steps": steps,
        "disturbance_applied": env.disturbed,
        "post_disturbance_success": bool(env.success) if env.disturbed else None,
        "post_disturbance_steps_to_success": (
            steps - disturbance_step + 1 if env.disturbed and env.success else None
        ),
        "inference_calls": inference_calls,
        "total_inference_ms": 1000 * inference_seconds,
        "mean_inference_ms": 1000 * inference_seconds / inference_calls,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    exposed = [row for row in rows if row["disturbance_applied"]]
    recovery_steps = [
        row["post_disturbance_steps_to_success"]
        for row in exposed if row["post_disturbance_success"]
    ]
    return {
        "scenario": rows[0]["scenario"],
        "action_steps": rows[0]["action_steps"],
        "episodes": len(rows),
        "mean_score": mean(row["max_normalized_coverage_score"] for row in rows),
        "mean_max_coverage": mean(row["max_coverage"] for row in rows),
        "mean_final_coverage": mean(row["final_coverage"] for row in rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": mean(row["success"] for row in rows),
        "mean_control_steps": mean(row["control_steps"] for row in rows),
        "disturbance_applied_count": len(exposed),
        "post_disturbance_successes": len(recovery_steps),
        "post_disturbance_success_rate": len(recovery_steps) / len(exposed) if exposed else None,
        "mean_post_disturbance_steps_to_success": mean(recovery_steps) if recovery_steps else None,
        "mean_inference_calls": mean(row["inference_calls"] for row in rows),
        "mean_total_inference_ms": mean(row["total_inference_ms"] for row in rows),
        "mean_inference_ms": sum(row["total_inference_ms"] for row in rows)
        / sum(row["inference_calls"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-steps", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--test-start-seed", type=int, default=200000)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record-first", type=int, default=0)
    parser.add_argument("--disturbance-step", type=int, help="Also run disturbed trials; one-based control step.")
    parser.add_argument("--disturbance-delta-v", type=float, default=120.0, help="Velocity kick in +X, in simulation units/s.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.n_test < 1 or not 0 <= args.record_first <= args.n_test:
        parser.error("Require at least one test episode and 0 <= record-first <= n-test.")
    if not 0 <= args.test_start_seed <= args.test_start_seed + args.n_test - 1 < 2**32:
        parser.error("Environment seeds must be in [0, 2**32).")
    if not 0 <= args.sampling_seed <= args.sampling_seed + args.n_test - 1 < 2**32:
        parser.error("Sampling seeds must be in [0, 2**32).")
    if len(set(args.action_steps)) != len(args.action_steps):
        parser.error("Action execution lengths must be unique.")

    config, policy = load_policy(args.checkpoint, args.device)
    runner = config.task.env_runner
    max_steps = runner.get("max_steps", 200) if args.max_steps is None else args.max_steps
    max_horizon = policy.horizon - policy.n_obs_steps + 1
    if not all(1 <= horizon <= max_horizon for horizon in args.action_steps):
        parser.error(f"Action execution lengths must be in [1, {max_horizon}].")
    if max_steps < 1:
        parser.error("max-steps must be positive.")
    scenarios = {"clean": None}
    if args.disturbance_step is not None:
        if not 1 <= args.disturbance_step <= max_steps:
            parser.error("disturbance-step must be within the episode's control steps.")
        if not 0 < args.disturbance_delta_v < float("inf"):
            parser.error("disturbance-delta-v must be positive and finite.")
        scenarios["disturbed"] = args.disturbance_step

    with args.checkpoint.open("rb") as checkpoint:
        checkpoint_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    args.output_dir.mkdir(parents=True)
    metadata = vars(args) | {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "training_seed": config.training.seed,
        "weights": "ema_model" if config.training.use_ema else "model",
        "observation_steps": policy.n_obs_steps,
        "prediction_steps": policy.horizon,
        "diffusion_inference_steps": policy.num_inference_steps,
        "max_steps": max_steps,
        "legacy_test": runner.get("legacy_test", False),
        "render_size": runner.get("render_size", 96),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "scenarios": list(scenarios),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    summaries = []
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for scenario, disturbance_step in scenarios.items():
            for horizon in args.action_steps:
                policy.n_action_steps = horizon
                rows = []
                for index in tqdm(range(args.n_test), desc=f"{scenario}, {horizon} steps"):
                    seed = args.test_start_seed + index
                    sampling_seed = args.sampling_seed + index
                    video = None
                    if index < args.record_first:
                        video = Path("videos") / scenario / f"h{horizon:02d}-seed{seed}.mp4"
                        (args.output_dir / video.parent).mkdir(parents=True, exist_ok=True)
                    row = {
                        "scenario": scenario,
                        "action_steps": horizon,
                        "test_seed": seed,
                        "sampling_seed": sampling_seed,
                        "video": video.as_posix() if video else None,
                    } | evaluate_episode(
                        policy, runner, horizon, seed, sampling_seed, max_steps,
                        disturbance_step, args.disturbance_delta_v,
                        args.output_dir / video if video else None,
                    )
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                    file.flush()
                    rows.append(row)
                summaries.append(summarize(rows))
    write_csv(args.output_dir / "summary.csv", summaries)
    print(args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
