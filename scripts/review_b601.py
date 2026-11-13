"""Audit B601 action/command alignment and export evidence for visual review.

Requires requirements-real-data.txt and requirements-analysis.txt.
Full-frame sheets and full-resolution videos cover the first seven episodes.
Event sheets for every episode are samples, not automatic success labels.
"""

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr


def percentiles(values):
    if not len(values):
        return None
    return dict(zip(("min", "median", "p95", "max"),
                    np.percentile(values, [0, 50, 95, 100]).tolist()))


def label_metrics(state, action, timing):
    """Test held-target behavior independently of measured tracking error."""
    delta = np.diff(action, axis=0)
    changed = np.r_[False, np.any(np.abs(delta) > 1e-5, axis=1)]
    command = timing.command_sent_complete_monotonic_ns.notna().to_numpy()
    sent = timing.command_sent_complete_monotonic_ns.to_numpy()[command]
    feedback = timing.feedback_read_complete_monotonic_ns.to_numpy()[command]
    camera = timing.camera_received_monotonic_ns.to_numpy()[command]
    no_command_transition = ~command[1:]
    holds = np.max(np.abs(delta[no_command_transition])) if no_command_transition.any() else 0.
    starts = np.r_[0, np.flatnonzero(changed)]
    lengths = np.diff(np.r_[starts, len(action)])
    return {
        "frames": len(action), "command_rows": int(command.sum()),
        "action_change_rows": int(changed.sum()),
        "action_changes_without_command": int((changed & ~command).sum()),
        "commands_without_action_change_after_first_frame": int((command[1:] & ~changed[1:]).sum()),
        "held_transition_max_abs_change": float(holds),
        "longest_unchanged_target_frames": int(lengths.max()),
        "command_after_feedback_ms": percentiles((sent - feedback) / 1e6),
        "command_after_camera_ms": percentiles((sent - camera) / 1e6),
        "command_precedes_feedback_rows": int((sent < feedback).sum()),
        "first_action": action[0].tolist(), "last_action": action[-1].tolist(),
        "state_joint_span": np.ptp(state, axis=0).tolist(),
        "action_joint_span": np.ptp(action, axis=0).tolist(),
        "action_state_mean_abs_error": np.abs(action-state).mean(0).tolist(),
        "input_key_available": "input_key" in timing,
        "input_key_counts": timing.input_key.fillna("").value_counts().to_dict() if "input_key" in timing else None,
        "initial_command_status": "sent" if command[0] else "no_send_timestamp",
    }


def write_sheet(frames, indices, output, timing, columns=6, size=(240, 180)):
    width, height = size
    cells = []
    for frame, index in zip(frames, indices):
        cell = np.full((height + 24, width, 3), 255, np.uint8)
        cell[24:] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        key = str(timing.input_key.iloc[index]) if "input_key" in timing and pd.notna(timing.input_key.iloc[index]) else "-"
        cv2.putText(cell, f"f{index}  {index/10:.1f}s  key:{key}", (3, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        cells.append(cell)
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], 255))
    image = np.concatenate([np.concatenate(cells[i:i+columns], axis=1)
                            for i in range(0, len(cells), columns)], axis=0)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Cannot write image {output}")


def plot_episode(state, action, timing, names, output):
    time = (timing.camera_received_monotonic_ns.to_numpy(dtype=np.int64) -
            int(timing.camera_received_monotonic_ns.iloc[0])) / 1e9
    command = timing.command_sent_complete_monotonic_ns.notna().to_numpy()
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), layout="constrained")
    for j, ax in enumerate(axes.flat[:7]):
        ax.plot(time, state[:, j], label="measured state", linewidth=1)
        ax.step(time, action[:, j], label="stored target", linewidth=1, where="post")
        ax.set_title(names[j])
        ax.set_xlabel("actual camera receive time (s)")
        ax.set_ylabel("motor degrees")
        ax.grid(alpha=.2)
    ax = axes.flat[7]
    ax.eventplot(time[command], lineoffsets=0, linelengths=.5)
    ax.set_title(f"Command send frames: {int(command.sum())}/{len(state)}")
    ax.set_xlabel("actual camera receive time (s)")
    ax.set_yticks([])
    axes.flat[0].legend()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/b601"))
    parser.add_argument("--output", type=Path, default=Path("results/b601-data-check/review"))
    parser.add_argument("--full-episodes", type=int, nargs="*", default=list(range(7)))
    args = parser.parse_args()
    for directory in ("plots", "event_sheets", "all_frames", "videos"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    episodes = json.loads((args.raw / "merged/episodes.json").read_text())
    replay = zarr.open_group(str(args.raw / "merged/demonstrations.zarr"), mode="r")
    names = replay.attrs["joint_names"]
    rows = []
    for episode in episodes:
        number, folder = episode["episode_index"], episode["source_folder"]
        source = args.raw / folder
        timing = pd.read_csv(source / "timing.csv")
        start, end = episode["start"], episode["end"]
        state, action = replay["data/state"][start:end], replay["data/action"][start:end]
        row = label_metrics(state, action, timing)
        row.update(episode_index=number, source_folder=folder)
        rows.append(row)
        prefix = f"episode-{number:02d}"
        plot_episode(state, action, timing, names, args.output / "plots" / f"{prefix}.png")
        n = len(state)
        indices = set(np.linspace(0, n-1, 12, dtype=int).tolist())
        # Supplement uniform samples with the gripper's extremes and release/close events.
        for anchor in (int(np.argmax(action[:, -1])), int(np.argmin(action[:, -1]))):
            indices.update([max(0, anchor-10), anchor, min(n-1, anchor+10)])
        if "input_key" in timing:
            for key in ("o", "p"):
                matches = np.flatnonzero(timing.input_key.fillna("").to_numpy() == key)
                if len(matches):
                    for anchor in (int(matches[0]), int(matches[-1])):
                        indices.update([max(0, anchor-5), anchor, min(n-1, anchor+5)])
        indices = sorted(indices)
        frames = [cv2.cvtColor(replay["data/img"][start+i], cv2.COLOR_RGB2BGR) for i in indices]
        write_sheet(frames, indices, args.output / "event_sheets" / f"{prefix}.jpg", timing)

        if number in args.full_episodes:
            table = pq.read_table(sorted(source.glob("data/**/*.parquet")), columns=["observation.images.front"])
            payloads = table["observation.images.front"].to_pylist()
            video_path = args.output / "videos" / f"{prefix}.mp4"
            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            stream.options = {"crf": "21", "preset": "fast"}
            page_frames = []
            for i, cell in enumerate(payloads):
                frame = cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Cannot decode {folder} frame {i}")
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                    container.mux(packet)
                page_frames.append(frame)
                if len(page_frames) == 100 or i == n-1:
                    first = i-len(page_frames)+1
                    write_sheet(page_frames, list(range(first, i+1)),
                                args.output / "all_frames" / f"{prefix}-frames-{first:04d}-{i:04d}.jpg",
                                timing, columns=10, size=(160, 120))
                    page_frames = []
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            with av.open(str(video_path)) as check:
                decoded = sum(1 for _ in check.decode(video=0))
            if decoded != n:
                raise ValueError(f"Video frame count differs: {video_path}")
            row["all_frames_exported"] = n
            row["video_decoded_frames"] = decoded
            row["video_fps"] = 10
        print(f"Reviewed label alignment and exported evidence: {prefix} ({n} frames)", flush=True)

    source_verification = "not yet available"
    source_audit_path = args.output / "source_verification.json"
    if source_audit_path.exists():
        source_audit = json.loads(source_audit_path.read_text())
        inputs_match = all(
            Path(item["path"]).is_file()
            and hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
            for item in source_audit.get("files", [])
        )
        if source_audit.get("files") and inputs_match:
            source_verification = {
                "status": source_audit["status"],
                "repository_commit": source_audit["repository_commit"],
                "details": "source_verification.json",
            }
        else:
            source_verification = "existing source audit needs rechecking: source inputs changed or are missing"
    summary = {
        "episodes": len(rows), "frames": sum(row["frames"] for row in rows),
        "command_rows": sum(row["command_rows"] for row in rows),
        "action_change_rows": sum(row["action_change_rows"] for row in rows),
        "action_changes_without_command": sum(row["action_changes_without_command"] for row in rows),
        "commands_without_action_change_after_first_frame": sum(row["commands_without_action_change_after_first_frame"] for row in rows),
        "command_precedes_feedback_rows": sum(row["command_precedes_feedback_rows"] for row in rows),
        "held_transition_max_abs_change": max(row["held_transition_max_abs_change"] for row in rows),
        "full_frame_export_episode_indices": args.full_episodes,
        "full_frame_export_frames": sum(row.get("all_frames_exported", 0) for row in rows),
        "source_code_verification": source_verification,
        "interpretation": "Stored actions are consistent with persistent joint-position targets. Target changes match command-send rows exactly if mismatch counts are zero. This is log evidence, not source-code or hardware packet verification.",
        "time_alignment": "In each row, camera receive and feedback-read completion precede command-send completion. Do not shift actions based on measured tracking lag alone. Camera exposure timing is not recorded.",
        "visual_success": "Exported evidence requires visual review; no automatic success labels are created.",
        "recordings": rows,
    }
    (args.output / "label_alignment.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, (dict, list))}
                  for row in rows]).to_csv(args.output / "label_alignment.csv", index=False)
    print(json.dumps({key: value for key, value in summary.items() if key != "recordings"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
