"""Structured JSON logging.

Application logs carry identifiers, scores, timings, model versions and error types.
They must never include customer questions, answers, feedback comments, or document
text; pass only such metadata in ``extra={"event": {...}}``.
"""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("rag_support")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
