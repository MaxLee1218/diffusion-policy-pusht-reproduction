# B601 hardware progress archive — 2026-11-15

This archive preserves the completed Windows/WSL integration work and all small experiment evidence available at the snapshot. Full local B601 data were indexed and hashed: **10,966 files / 31,999,858,845 bytes**. **3,235 files / 85,055,832 bytes** were copied here, and 77 evidence files were already versioned in their original repository locations. Bulk data remain intact at the original local paths; this Git archive does not contain the complete visual dataset or model weights.

## Experiment results

| Session | Recorded result |
| --- | --- |
| `rhc-trial-001` | Initial engineering trial; 23 inferences / 129 actions; stopped on scheduling lateness. Earlier attempts are retained. |
| `rhc-timing-validation-001` | 60 seconds of real inference and full recording with constant position targets; 100 inferences / 572 targets. Separate focus-loss interruption retained. |
| `rhc-trial-002` | 171.73 seconds / 286 inferences / 1,638 actions; operator reported no contact with the chess piece. Ended on the training-state range check. |
| `rhc-trial-003` | 39 automatic reference-alignment moves, followed by an expired first policy request; zero policy actions. Reference alignment caused the reported retreat toward the base. |
| `rhc-trial-004` | Current pose saved and reference alignment disabled; 54.67 seconds / 91 inferences / 517 actions. Assistant paused negative-X, positive-Y drift. Arm targets matched policy outputs within floating-point roundoff. |
| `rhc-trial-005` | Operator requested reset and retry. Reset verified within 0.044 motor degrees; expired startup attempt preserved. Retry ran 246.00 seconds / 410 inferences / 2,338 actions, ending on a pause/takeover. Drift persisted; successful grasp and placement have not been established. |

The latest policy recording finalized with 7,200 RGB/depth frames, 11,741 control cycles and no source-frame gaps in its frame metadata. Its JSONL/manifest counts were checked; pixel/depth-array decoding was not repeated for that trial. Trial 004 additionally has complete RGB/depth decode checks. SDK cache entries lack individual motor reception timestamps; controller read time is not proof of fresh sensor feedback. Runner logs do not distinguish Esc/manual takeover from window focus loss.

Useful evidence:

- [Latest trial outcome](files/data/b601/experiments/rhc-trial-005/outcome.json), [reset verification](files/data/b601/experiments/rhc-trial-005/reset-result.json), [trial protocol](files/data/b601/experiments/rhc-trial-005/protocol.json).
- [Latest complete predictions](files/data/b601/experiments/rhc-trial-005/predictions.jsonl), [dispatched actions](files/data/b601/experiments/rhc-trial-005/sequence-actions.jsonl), [control cycles, lossless gzip](files/data/b601/experiments/rhc-trial-005/recordings/20261114T170203-c6f9c2fc/control-cycles.jsonl.gz), [recording manifest](files/data/b601/experiments/rhc-trial-005/recordings/20261114T170203-c6f9c2fc/manifest.json).
- [Trial 004 direction audit](files/data/b601/experiments/rhc-trial-004/diagnostics/direction-audit.json), [video timeline](files/data/b601/experiments/rhc-trial-004/diagnostics/video-timeline.png).
- [Earlier direction investigation](files/data/b601/experiments/direction-review/analysis.json).

## Included and retained data

`files/` mirrors the original project-relative source paths. It includes protocols, setup/reference poses, workspace bounds, camera calibration and recording manifests, complete prediction/action/control/timestamp logs, RPC replies, failures, source snapshots, training metadata and diagnostic images. Main implementation and regression tests are in the repository's `real_robot/`, `scripts/` and `tests/` directories. The full CPU suite passed **85 tests** before publication.

New evidence files have a **5 MiB per-file cap**. Text larger than that is retained byte-for-byte through gzip compression. Entire bulk collections remain local even when an individual chunk is small: RGB video segments, raw depth blocks, repeated RGB/state NPZ inputs, demonstration Parquet/Zarr/image data, bulk dataset-review media and checkpoints. Original source data were neither deleted nor changed by the archive operation. Generated Python caches are indexed but omitted from Git. Dependency installations/caches are outside the dataset inventory; the reviewed driver source and wheel provenance are included.

- [Archive counts and policy](archive-summary.json).
- [Complete inventory](inventory.csv.gz): source path, original size/SHA-256, export disposition, repository path, exported size/SHA-256, compression and capture time.
- [Exported-file inventory](exported-files.csv.gz).
- [Local-only inventory](local-only-files.csv.gz): identifies all 7,654 retained files and the reason they were excluded.
- `SHA256SUMS`: checks every file in this archive except itself. Live controller status files are historical snapshots and can differ from the still-refreshing local originals.

Verify the downloaded archive from this directory:

```bash
sha256sum --check SHA256SUMS
```

After decompression, each gzip file must match its `source_sha256` in the inventory. Source paths locate the original local files; raw datasets and weights must be restored separately to reproduce image-conditioned inference. Historical controller PIDs and live-status snapshots describe past sessions, rather than a new machine's configuration.

To make a new archive without accessing or moving hardware:

```bash
python scripts/archive_b601_progress.py --output results/b601-hardware-new --max-file-mib 5
```

The exporter creates inventories and data copies in a fresh directory. This publication additionally includes the experiment summary, verification report and `SHA256SUMS`.

## Next work

Keep task inference paused while comparing the frozen real inputs against training observations and expert approach actions. Check camera field of view, image preprocessing, joint-state conditioning, learned task phases and model behavior on actual demonstrations. The trials do not support flipping motor directions or the X axis: dispatched arm goals match native absolute-motor policy outputs. Preserve the current checkpoint and experiment records; use the comparison to decide whether a model/input fix or targeted new demonstrations are needed. No grasp/placement success rate has been established.
