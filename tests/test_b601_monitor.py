import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("matplotlib")
spec = importlib.util.spec_from_file_location(
    "monitor_b601", Path(__file__).resolve().parents[1] / "scripts/monitor_b601_training.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_partial_log_line_is_counted_once_after_completion(tmp_path):
    path = tmp_path / "metrics.jsonl"
    complete = json.dumps({"epoch": 3, "train_loss": .2})+"\n"
    second = json.dumps({"epoch": 3, "train_loss": .1, "val_loss": .15})+"\n"
    path.write_text(complete+second[:12])
    log = monitor.MetricLog(path)
    log.refresh()
    assert log.steps == 1 and not log.epoch_metrics
    with path.open("a") as handle:
        handle.write(second[12:])
    log.refresh()
    log.refresh()
    assert log.steps == 2 and len(log.epoch_metrics) == 1
    assert log.epoch_batches == {3: 2}


def test_truncated_log_restarts_incremental_reader(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text((json.dumps({"epoch": 3, "train_loss": .2})+"\n")*5)
    log = monitor.MetricLog(path)
    log.refresh()
    assert log.steps == 5
    path.write_text(json.dumps({"epoch": 4, "train_loss": .1})+"\n")
    log.refresh()
    assert log.steps == 1 and log.last["epoch"] == 4
