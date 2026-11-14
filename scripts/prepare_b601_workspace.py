"""Derive the requested workspace envelope from measured formal demonstrations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"collection/b601"))
from b601_pick_place import Kinematics, URDF

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/b601/merged/demonstrations.zarr")
    parser.add_argument("--output", type=Path, default=ROOT/"collection/b601/workspace-bounds.local.json")
    parser.add_argument("--report", type=Path, default=ROOT/"data/b601/workspace-from-history/report.json")
    args = parser.parse_args()
    replay = zarr.open_group(str(args.replay), mode="r")
    state = replay["data/state"][:]
    ends = replay["meta/episode_ends"][:]
    if state.ndim != 2 or state.shape[1] != 7 or not np.isfinite(state).all() or not len(state):
        raise ValueError("Expected finite measured seven-joint demonstration states")
    if not len(ends) or ends[-1] != len(state) or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("Invalid episode boundaries")
    model = Kinematics(URDF)
    xyz = np.array([model.pose(joints[:6])[:3, 3] for joints in state])
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    lower_padding = np.array([.01, .01, .005])
    upper_padding = np.array([.01, .01, .01])
    bounds = np.column_stack((lower-lower_padding, upper+upper_padding))
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Derived bounds are invalid")
    assert np.all(xyz >= bounds[:, 0]) and np.all(xyz <= bounds[:, 1])
    np.testing.assert_allclose(lower-bounds[:, 0], lower_padding, rtol=0, atol=1e-15)
    np.testing.assert_allclose(bounds[:, 1]-upper, upper_padding, rtol=0, atol=1e-15)
    values = bounds.ravel().tolist()
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "source": "all complete formal episodes in the merged demonstration replay",
              "source_replay": str(args.replay), "position_source": "measured state, not command targets",
              "episodes": len(ends), "frames": len(state), "frame": "robot base", "point": "end_link",
              "units": "meters", "observed_min_xyz": lower.tolist(), "observed_max_xyz": upper.tolist(),
              "lower_padding_xyz": lower_padding.tolist(), "upper_padding_xyz": upper_padding.tolist(),
              "bound_order": ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"], "bounds": values,
              "output": str(args.output), "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
              "measured_states_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
              "extrema": {}, "all_recorded_positions_enclosed": True,
              "provenance": "User requested 1 cm outward expansion, except 0.5 cm downward.",
              "limitations": ["This is a history-derived envelope, not a measured obstacle-free workspace.",
                              "Box interiors and corners are not proven reachable or collision-free.",
                              "Coordinates depend on the recording zero convention and the URDF.",
                              "Parked-pose integration traces and commanded targets were not included."]}
    for axis, name in enumerate("xyz"):
        for side, index in (("min", int(np.argmin(xyz[:, axis]))), ("max", int(np.argmax(xyz[:, axis])))):
            episode = int(np.searchsorted(ends, index, side="right"))
            report["extrema"][name+"_"+side] = {"global_frame": index, "episode_index": episode,
                "local_frame": int(index-(ends[episode-1] if episode else 0)), "xyz": xyz[index].tolist(),
                "joint_motor_degrees": state[index].tolist()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = args.output.read_bytes()
        if json.loads(previous) != values:
            backup = args.report.parent/("previous-workspace-"+hashlib.sha256(previous).hexdigest()[:12]+".json")
            backup.write_bytes(previous)
            report["previous_configuration_backup"] = str(backup)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2)+"\n")
    temporary.replace(args.output)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
