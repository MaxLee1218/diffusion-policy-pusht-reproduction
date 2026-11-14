"""Archive B601 metadata and index retained bulk data without touching hardware."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TEXT = {".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml", ".md", ".log",
        ".py", ".urdf", ".swift", ".toml", ".xml"}
IMAGES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
ZARR_META = {".zarray", ".zattrs", ".zgroup", ".zmetadata"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bulk_reason(path):
    parts = path.parts
    if path.suffix.lower() in {".ckpt", ".pt", ".pth"}:
        return "model checkpoint"
    if path.suffix.lower() in {".npz", ".npy"}:
        return "bulk depth, image/state inputs, or array data"
    if path.suffix.lower() in {".mkv", ".mp4", ".avi", ".parquet"}:
        return "bulk video or demonstration data"
    if "demonstrations.zarr" in parts and path.name not in ZARR_META:
        return "merged demonstration array chunks"
    if any(s.startswith("pick-place-") for s in parts) and path.suffix.lower() in IMAGES:
        return "original demonstration image collection"
    if "__pycache__" in parts or path.suffix == ".pyc":
        return "generated Python cache"
    if parts[:2] == ("results", "b601-data-check"):
        if any(s in {"previews", "all_frames", "event_sheets", "plots", "videos"} for s in parts[2:]):
            return "bulk dataset visual-review media"
        if path.name.startswith("overview-"):
            return "bulk dataset visual-review media"
    return None


def snapshot_bytes(path):
    # A paused GUI still refreshes its status file. Capture a complete JSON
    # document, retaining the captured bytes and recording its snapshot time.
    for _ in range(30):
        data = path.read_bytes()
        try:
            if path.suffix == ".json":
                json.loads(data)
            return data
        except (ValueError, UnicodeDecodeError):
            time.sleep(.02)
    raise ValueError(f"Cannot capture complete JSON: {path}")


def gzip_bytes(data):
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(data)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=float, default=5.)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new archive directory: {output}")
    if not 0 < args.max_file_mib <= 20:
        raise ValueError("Choose a per-file cap of at most 20 MiB")
    limit = int(args.max_file_mib*1024**2)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    sources = [ROOT/"data/b601", *sorted((ROOT/"experiments").glob("b601*")),
               *sorted((ROOT/"results").glob("b601*")), ROOT/"data/tools/b601-driver-review",
               ROOT/"collection/b601/workspace-bounds.local.json"]
    sources += [ROOT/"data/tools/windows-hardware"/s
                for s in ("wheel-sources.json", "hold_current.py", "probe_devices.py")]
    paths = sorted({p for source in sources if source.exists()
                    for p in ([source] if source.is_file() else source.rglob("*")) if p.is_file()})
    output.mkdir(parents=True)
    began = last_update = time.monotonic()
    rows, total_bytes, exported_bytes = [], 0, 0
    for index, path in enumerate(paths):
        rel = path.relative_to(ROOT)
        row = {"source": rel.as_posix(), "source_bytes": path.stat().st_size,
               "source_sha256": "", "disposition": "local_only", "reason": "",
               "repository_path": "", "repository_bytes": "", "repository_sha256": "",
               "encoding": "", "captured_utc": datetime.now(timezone.utc).isoformat()}
        reason = bulk_reason(rel)
        eligible = path.suffix.lower() in TEXT | IMAGES or path.name in ZARR_META or path.suffix == ".whl"
        if rel.as_posix() in tracked:
            row.update(disposition="already_versioned", repository_path=rel.as_posix(), encoding="original")
            row["source_sha256"] = digest(path)
            row.update(repository_bytes=row["source_bytes"], repository_sha256=row["source_sha256"])
        elif eligible and reason is None:
            data = snapshot_bytes(path)
            row.update(source_bytes=len(data), source_sha256=hashlib.sha256(data).hexdigest())
            encoded = data
            if len(data) > limit and path.suffix.lower() in TEXT:
                encoded = gzip_bytes(data)
                row["encoding"] = "gzip"
            else:
                row["encoding"] = "original"
            if len(encoded) <= limit:
                dest = output/"files"/rel
                if row["encoding"] == "gzip":
                    dest = dest.with_name(dest.name+".gz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(encoded)
                row.update(disposition="exported", repository_path=dest.relative_to(ROOT).as_posix(),
                           repository_bytes=len(encoded), repository_sha256=hashlib.sha256(encoded).hexdigest())
                exported_bytes += len(encoded)
            else:
                row["reason"] = "exceeds per-file size cap after lossless compression"
        else:
            row["source_sha256"] = digest(path)
            row["reason"] = reason or "bulk or unsupported binary artifact"
        total_bytes += row["source_bytes"]
        rows.append(row)
        if time.monotonic()-last_update > 10:
            print(json.dumps({"indexed": index+1, "files": len(paths), "hashed_gib": round(total_bytes/1024**3, 2)}), flush=True)
            last_update = time.monotonic()
    for filename, selected in [("inventory.csv.gz", rows),
                               ("local-only-files.csv.gz", [r for r in rows if r["disposition"] == "local_only"]),
                               ("exported-files.csv.gz", [r for r in rows if r["disposition"] == "exported"])]:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
        (output/filename).write_bytes(gzip_bytes(stream.getvalue().encode()))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "max_new_file_bytes": limit,
               "source_file_count": len(rows), "source_bytes": total_bytes,
               "exported_file_count": sum(r["disposition"] == "exported" for r in rows),
               "exported_bytes": exported_bytes,
               "already_versioned_file_count": sum(r["disposition"] == "already_versioned" for r in rows),
               "local_only_file_count": sum(r["disposition"] == "local_only" for r in rows),
               "local_only_bytes": sum(r["source_bytes"] for r in rows if r["disposition"] == "local_only"),
               "all_source_files_retained": True, "hardware_access_or_motion": False,
               "snapshot_note": "Live status files are captured snapshots; their source paths may update after capture.",
               "bulk_policy": "Video, depth arrays, repeated model input snapshots, checkpoints, raw demonstration arrays/images and bulk dataset review media remain local as complete datasets, even where an individual chunk is small.",
               "elapsed_seconds": time.monotonic()-began}
    (output/"archive-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
