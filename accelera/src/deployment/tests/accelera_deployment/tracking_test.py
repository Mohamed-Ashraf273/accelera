import json

# import os
from accelera.src.deployment.tracking import PredictionTracker


def test_disabled_tracker_does_not_write_events(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    tracker = PredictionTracker({"enabled": False, "path": str(log_path)})

    tracker.record({"status": "success", "rows": 1})

    assert not log_path.exists()
    assert tracker.describe() == {"enabled": False, "path": str(log_path)}
    assert tracker.summary() == {"enabled": False}


def test_record_writes_jsonl_evet_and_summary_counts(tmp_path):
    log_path = tmp_path / "logs" / "predictions.jsonl"
    tracker = PredictionTracker({"enabled": True, "path": str(log_path)})
    tracker.record({"status": "success", "rows": 2, "prediction": [1, 0]})
    tracker.record({"status": "error", "rows": "1", "message": "bad input"})
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first_evnt = json.loads(lines[0])
    assert first_evnt["status"] == "success"
    assert first_evnt["rows"] == 2
    assert first_evnt["prediction"] == [1, 0]
    assert first_evnt["timestamp"].endswith("Z")
    summary = tracker.summary()
    assert summary["enabled"] is True
    assert summary["path"] == str(log_path)
    assert summary["total_requests"] == 2
    assert summary["total_rows"] == 3
    assert summary["status_counts"] == {"success": 1, "error": 1}
    assert summary["last_event"]["message"] == "bad input"


def test_summary_handles_missing_enabld_log_file(tmp_path):
    log_path = tmp_path / "missing" / "predictions.jsonl"
    tracker = PredictionTracker({"enabled": True, "path": str(log_path)})
    assert tracker.summary() == {
        "enabled": True,
        "path": str(log_path),
        "total_requests": 0,
        "total_rows": 0,
        "status_counts": {},
        "last_event": None,
    }
