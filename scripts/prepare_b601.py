"""Restore, audit, and merge the B601 LeRobot v3 demonstration backup.

The raw Parquet files retain the original 640x480 images. The merged Zarr
contains RGB images resized to 320x240, seven position features and actions.
Only complete, structurally valid, nonduplicate formal recordings are merged.
Pilot, interrupted and stationary recordings are retained and audited.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc


JOINTS = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper")]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def restore(source, raw):
    manifest = json.loads((source / "manifest.json").read_text())
    require(not raw.exists() or not any(raw.iterdir()), f"Extraction needs an empty directory: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    results = []
    for volume in manifest["volumes"]:
        archive = source / volume["name"]
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        require(archive.stat().st_size == volume["bytes"], f"Archive size: {archive}")
        require(digest.hexdigest() == volume["sha256"], f"Archive checksum: {archive}")
        with tarfile.open(archive) as tar:
            members = []
            for original in tar.getmembers():
                member = copy.copy(original)
                if member.name.startswith("data/b601/"):
                    member.name = member.name[len("data/b601/"):]
                require(member.isfile() or member.isdir(), f"Unsupported member: {member.name}")
                target = (raw / member.name).resolve()
                require(target.is_relative_to(raw.resolve()), f"Unsafe member: {member.name}")
                require(not target.exists(), f"Refusing overwrite: {target}")
                members.append(member)
            tar.extractall(raw, members=members, filter="data")
        results.append({"name": archive.name, "sha256": digest.hexdigest(),
                        "status": "pass", "bytes": archive.stat().st_size})
        print(f"Verified and extracted {archive.name}", flush=True)
    shutil.copy2(source / "manifest.json", raw / "source_manifest.json")
    shutil.copy2(source / "SHA256SUMS.txt", raw / "source_SHA256SUMS.txt")
    write_json(raw / "archive_validation.json", {"volumes": results})


def percentiles(values):
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    np.percentile(values, [0, 50, 95, 99, 100]).tolist()))


def finalize_report(report_path, raw):
    """Summarize the completed audit and check the separately supplied start pose."""
    report = json.loads(report_path.read_text())
    report["archive_validation"] = json.loads((raw / "archive_validation.json").read_text())
    included = [r for r in report["recordings"] if r["included"]]
    report["formal_timing"] = {
        "effective_fps_min": min(r["timing"]["effective_fps"] for r in included),
        "effective_fps_max": max(r["timing"]["effective_fps"] for r in included),
        "gaps_over_200ms": sum(r["timing"]["gaps_over_200ms"] for r in included),
        "max_camera_interval_ms": max(r["timing"]["camera_interval_ms"]["max"] for r in included),
        "command_timestamp_rows": sum(r["timing"]["command_timestamp_rows"] for r in included),
        "duplicate_image_frames_within_recordings": sum(r["duplicate_image_frames"] for r in included),
    }
    reference_path = raw / "references/b601-start-pose-20261112-180049.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        stationary = next(r for r in report["recordings"] if r["type"] == "stationary-pose-reference")
        table = pq.read_table(sorted((raw / stationary["folder"]).glob("data/**/*.parquet")),
                              columns=["observation.state"])
        state = np.asarray(table["observation.state"].to_pylist())
        expected = np.asarray([reference["observation.state"][name] for name in JOINTS])
        require(len(state) == reference["samples"], "Start-pose reference sample count")
        require(np.allclose(np.median(state, axis=0), expected, atol=1e-5, rtol=0),
                "Start-pose reference differs from measured joint medians")
        report["pose_reference_check"] = "pass"
        report["position_units"] = reference["position_units"]
    write_json(report_path, report)

    previews = report_path.parent / "previews"
    panels = []
    for row in report["recordings"]:
        image = cv2.imread(str(previews / f"{row['folder']}.jpg")) if row["status"] == "valid" else None
        if image is None:
            continue
        panel = np.full((270, 960, 3), 255, dtype=np.uint8)
        panel[30:] = np.concatenate([image[:, :320], image[:, 960:1280], image[:, -320:]], axis=1)
        cv2.putText(panel, row["folder"], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
        panels.append(panel)
    for offset in range(0, len(panels), 6):
        cv2.imwrite(str(report_path.parent / f"overview-{offset//6+1:02d}.jpg"),
                    np.concatenate(panels[offset:offset+6], axis=0))
    timing = report["formal_timing"]
    summary = (
        f"B601 demonstration audit\n"
        f"Archive SHA-256: {len(report['archive_validation']['volumes'])} verified volumes\n"
        f"Raw recordings: {report['raw_root']}\n"
        f"Merged Zarr: {report['merged_path']}\n"
        f"Formal recordings: {report['included_episodes']} episodes, {report['included_frames']} frames\n"
        f"Valid pilot episodes (excluded): {report['valid_pilot_episodes']}\n"
        f"Incomplete recordings (excluded): {report['incomplete_recordings']}\n"
        f"Stationary references (excluded): {report['stationary_references']}\n"
        f"Validation errors: {report['errors']}\n"
        f"Joint state/action dimensions: 7, units: {report.get('position_units', 'not verified')}\n"
        f"Images: RGB {report['image_size']}; originals retained at 640x480\n"
        f"Nominal FPS: 10; measured FPS: {timing['effective_fps_min']:.4f} to {timing['effective_fps_max']:.4f}\n"
        f"Camera receive gaps >200ms: {timing['gaps_over_200ms']}; max gap: {timing['max_camera_interval_ms']:.3f} ms\n"
        f"Command send timestamp rows: {timing['command_timestamp_rows']}/{report['included_frames']}\n"
        f"Start-pose reference: {report.get('pose_reference_check', 'not present')}\n\n"
        "Validation covers archive integrity, recording inventory, embedded image decoding,\n"
        "finite seven-joint states/actions, frame order, episode boundaries, timing and metadata.\n"
        "Blank command timestamps must be interpreted using the recorder's command/hold behavior.\n"
        "Nominal timestamps were preserved; no action shifts, resampling or trimming were applied.\n"
        "Visual previews are sampled frames; task success is not labeled by these checks.\n"
        "The existing PushT training profile uses two-dimensional actions and needs a separate\n"
        "seven-joint dataset adapter and task configuration for these recordings.\n")
    (report_path.parent / "summary.txt").write_text(summary)
    return report


def audit_recording(raw, spec, preview, width, height):
    folder = raw / spec["folder"]
    row = {"folder": spec["folder"], "type": spec["type"],
           "expected_frames": spec["frames"], "errors": [], "warnings": []}
    files = [p for p in folder.rglob("*") if p.is_file()]
    require(len(files) == spec["file_count"], "File count differs from source manifest")
    require(sum(p.stat().st_size for p in files) == spec["bytes"], "Byte count differs from source manifest")
    info_path = folder / "meta/info.json"
    if not info_path.exists() or not spec["episodes"]:
        images = list(folder.glob("images/**/*.png"))
        for image in images:
            decoded = cv2.imread(str(image))
            require(decoded is not None and decoded.shape == (480, 640, 3), f"Bad orphan image: {image}")
        row.update(status="incomplete", orphan_images=len(images), included=False)
        row["warnings"].append("No finalized episode; images alone cannot recover synchronized states/actions.")
        return row, None

    info = json.loads(info_path.read_text())
    require(info["total_episodes"] == spec["episodes"] == 1, "Expected one episode per recording")
    require(info["fps"] == 10, "Unexpected frame rate")
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        require(feature["shape"] == [7] and feature["names"] == JOINTS, f"Feature layout: {key}")
    table = pq.read_table(sorted(folder.glob("data/**/*.parquet")))
    n = table.num_rows
    require(n == info["total_frames"] == spec["frames"], "Frame count differs from metadata")
    require(all(table[c].null_count == 0 for c in table.column_names), "Null data cells")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    require(state.shape == action.shape == (n, 7), "State/action dimensions")
    require(np.isfinite(state).all() and np.isfinite(action).all(), "Nonfinite state/action")
    timestamp = table["timestamp"].to_numpy()
    require(np.allclose(timestamp, np.arange(n) / info["fps"], atol=1e-4, rtol=0), "Timestamp sequence")
    for key in ("frame_index", "index"):
        require(np.array_equal(table[key].to_numpy(), np.arange(n)), f"Nonconsecutive {key}")
    for key in ("episode_index", "task_index"):
        require(np.all(table[key].to_numpy() == 0), f"Unexpected {key}")
    metadata = pq.read_table(sorted(folder.glob("meta/episodes/**/*.parquet"))).to_pylist()
    require(len(metadata) == 1 and metadata[0]["length"] == n, "Episode metadata length")
    require(metadata[0]["dataset_from_index"] == 0 and metadata[0]["dataset_to_index"] == n,
            "Episode boundaries in metadata")
    tasks = pq.read_table(folder / "meta/tasks.parquet").to_pylist()
    require(len(tasks) == 1 and tasks[0]["task_index"] == 0, "Task metadata")
    row["task"] = metadata[0]["tasks"]
    stats = json.loads((folder / "meta/stats.json").read_text())
    for key, values in (("observation.state", state), ("action", action)):
        require(np.allclose(stats[key]["min"], values.min(0), atol=1e-4), f"Stats min: {key}")
        require(np.allclose(stats[key]["max"], values.max(0), atol=1e-4), f"Stats max: {key}")

    timing = pd.read_csv(folder / "timing.csv")
    require(len(timing) == n, "Timing row count")
    require(np.array_equal(timing.frame_index.to_numpy(), np.arange(n)), "Timing frame alignment")
    camera = timing.camera_received_monotonic_ns.to_numpy(dtype=np.int64)
    feedback = timing.feedback_read_complete_monotonic_ns.to_numpy(dtype=np.int64)
    require(np.all(np.diff(camera) > 0), "Nonmonotonic camera receive timestamps")
    require(np.all(feedback >= camera), "Feedback timestamp precedes camera timestamp")
    intervals = np.diff(camera) / 1e6
    row["timing"] = {
        "camera_interval_ms": percentiles(intervals),
        "effective_fps": float((n - 1) * 1e9 / (camera[-1] - camera[0])),
        "gaps_over_200ms": int((intervals > 200).sum()),
        "feedback_after_camera_ms": percentiles((feedback - camera) / 1e6),
        "command_timestamp_rows": int(timing.command_sent_complete_monotonic_ns.notna().sum()),
    }
    if row["timing"]["gaps_over_200ms"]:
        row["warnings"].append("Camera receive intervals exceed two nominal frame periods.")
    if not row["timing"]["command_timestamp_rows"]:
        row["warnings"].append("Command send timestamps absent; command execution timing is not verified.")

    images = np.empty((n, height, width, 3), dtype=np.uint8)
    digest = hashlib.sha256(state.tobytes() + action.tobytes())
    image_hashes = set()
    selections = set(np.linspace(0, n - 1, 6, dtype=int).tolist())
    thumbnails = []
    for i, cell in enumerate(table["observation.images.front"].to_pylist()):
        payload = cell["bytes"]
        require(payload is not None, f"Missing embedded image {i}")
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        require(frame is not None and frame.shape == (480, 640, 3), f"Invalid image {i}")
        digest.update(payload)
        image_hashes.add(hashlib.sha256(payload).digest())
        images[i] = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if i in selections:
            thumb = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"frame {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            thumbnails.append(thumb)
    cv2.imwrite(str(preview / f"{folder.name}.jpg"), np.concatenate(thumbnails, axis=1))
    row.update(status="valid", frames=n, fps=10, duplicate_image_frames=n-len(image_hashes),
               content_sha256=digest.hexdigest(),
               state_min=state.min(0).tolist(), state_max=state.max(0).tolist(),
               action_min=action.min(0).tolist(), action_max=action.max(0).tolist(),
               action_equals_state_fraction=float(np.all(action == state, axis=1).mean()),
               action_state_abs_error_mean=np.abs(action-state).mean(0).tolist(),
               state_step_abs_max=np.abs(np.diff(state, axis=0)).max(0).tolist(),
               action_step_abs_max=np.abs(np.diff(action, axis=0)).max(0).tolist(),
               included=spec["type"] == "recording")
    if row["action_equals_state_fraction"] > .99:
        row["warnings"].append("Actions equal measured state in almost all frames; verify label semantics before training.")
    return row, {"img": images, "state": state, "action": action,
                 "timestamp": timestamp, "frame_index": np.arange(n, dtype=np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/d/下载/b601-github-backup-20261112"))
    parser.add_argument("--raw", type=Path, default=Path("data/b601"))
    parser.add_argument("--output", type=Path, default=Path("results/b601-data-check"))
    parser.add_argument("--extract", action="store_true", help="Verify archives and restore into an empty --raw directory")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    require(args.width > 0 and args.height > 0, "Image size must be positive")
    if args.extract:
        restore(args.source, args.raw)
    manifest = json.loads((args.raw / "source_manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    preview = args.output / "previews"
    preview.mkdir(exist_ok=True)
    final = args.raw / "merged/demonstrations.zarr"
    staging = final.with_suffix(".partial.zarr")
    require(not final.exists() and not staging.exists(), f"Refusing to replace merged data: {final}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(staging), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    arrays = {}
    for key, shape, dtype, chunks in (
        ("img", (0, args.height, args.width, 3), "u1", (16, args.height, args.width, 3)),
        ("state", (0, 7), "f4", (1024, 7)),
        ("action", (0, 7), "f4", (1024, 7)),
        ("timestamp", (0,), "f4", (1024,)),
        ("frame_index", (0,), "i8", (1024,)),
    ):
        arrays[key] = data.create_dataset(key, shape=shape, dtype=dtype, chunks=chunks, compressor=compressor)
    rows, ends, episodes, seen = [], [], [], set()
    for spec in manifest["folders"]:
        try:
            row, values = audit_recording(args.raw, spec, preview, args.width, args.height)
            if row.get("content_sha256") in seen:
                row["included"] = False
                row["warnings"].append("Duplicate recording content; excluded from merge.")
            if row["included"]:
                seen.add(row["content_sha256"])
                for key, array in arrays.items():
                    array.append(values[key])
                start = ends[-1] if ends else 0
                ends.append(start + row["frames"])
                episodes.append({"episode_index": len(episodes), "source_folder": spec["folder"],
                                 "start": start, "end": ends[-1], "frames": row["frames"],
                                 "content_sha256": row["content_sha256"]})
        except Exception as exc:
            row = {"folder": spec["folder"], "type": spec["type"], "status": "error",
                   "included": False, "errors": [str(exc)], "warnings": []}
        rows.append(row)
        print(f"{row['status']:10s} {spec['folder']} included={row['included']}", flush=True)
    meta.array("episode_ends", np.asarray(ends, dtype=np.int64))
    root.attrs.update({"source": manifest["source"], "fps": 10, "joint_names": JOINTS,
                       "image_color": "RGB", "image_size": [args.width, args.height],
                       "action_semantics": "unchanged source action; seven joint position features",
                       "source_snapshot": manifest["source_snapshot"]})
    require(all(a.shape[0] == (ends[-1] if ends else 0) for a in arrays.values()), "Merged lengths differ")
    write_json(staging.parent / "episodes.json", episodes)
    summary = {
        "raw_root": str(args.raw.resolve()), "merged_path": str(final.resolve()),
        "folders": len(rows), "included_episodes": len(episodes),
        "included_frames": ends[-1] if ends else 0,
        "valid_pilot_episodes": sum(r["type"] == "pilot" and r["status"] == "valid" for r in rows),
        "incomplete_recordings": sum(r["status"] == "incomplete" for r in rows),
        "stationary_references": sum(r["type"] == "stationary-pose-reference" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows), "joint_names": JOINTS,
        "image_size": [args.width, args.height], "fps": 10,
        "notes": ["Original Parquet and full-resolution image bytes are preserved.",
                  "Pilot, stationary and unfinished recordings are excluded from merged training data.",
                  "Timestamps in Parquet are nominal; timing.csv contains actual camera receive times.",
                  "Visual task success and robot calibration/units are not certified by structural checks.",
                  "No action shift, resampling, trimming or coordinate/unit conversion was applied."],
        "recordings": rows,
    }
    write_json(args.output / "report.json", summary)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(
        args.output / "recordings.csv", index=False)
    require(len(episodes) > 0, "No valid formal demonstrations; see report.json")
    staging.rename(final)
    finalize_report(args.output / "report.json", args.raw)
    print(json.dumps({k: v for k, v in summary.items() if k != "recordings"}, indent=2), flush=True)
    if summary["errors"]:
        raise SystemExit("Some recordings failed validation; inspect report.json before training.")


if __name__ == "__main__":
    main()
