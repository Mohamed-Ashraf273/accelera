import json
import os
from datetime import datetime


class PredictionTracker:
    def __init__(self, config=None):
        config = config or {}
        self.enabled = config.get("enabled", False)
        self.path = config.get("path", "prediction_logs/predictions.jsonl")

    def describe(self):
        return {
            "enabled": self.enabled,
            "path": self.path,
        }

    def record(self, event):
        if not self.enabled:
            return

        event = dict(event)
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"

        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=self._json_default) + "\n")

    def summary(self):
        if not self.enabled:
            return {"enabled": False}

        total_requests = 0
        total_rows = 0
        status_counts = {}
        last_event = None

        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    total_requests += 1
                    total_rows += int(event.get("rows", 0))
                    status = event.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    last_event = event

        return {
            "enabled": True,
            "path": self.path,
            "total_requests": total_requests,
            "total_rows": total_rows,
            "status_counts": status_counts,
            "last_event": last_event,
        }

    def _json_default(self, value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)
