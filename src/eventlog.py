"""JSONL event log for ingest runs — agent-readable diagnostics.

One event per line, append-only, thread-safe. Events emitted:
  run_start / run_end   : run-level context and summary aggregates
  llm_call              : per-file LLM invocation — timing breakdown, prompt
                          size, related pages + scores, json_repair_used
  decision              : each LLM-proposed new/updated page (title, type)
  merge_violation       : a related page scored ≥ 0.60 but was NOT placed in
                          updated_pages — the MERGE RULE failed
  error                 : any failure, tagged with phase

Pre-computed flags are limited to *deterministic* signals (e.g.
`json_repair_used`, `merge_violation`). Threshold-based signals like "slow"
are intentionally NOT pre-computed so the schema stays stable across retunings
— the reader derives them from raw fields.
"""

import json
import threading
from datetime import datetime
from typing import Any, Optional

from config import WIKI_DIR

EVENTS_DIR = WIKI_DIR / ".logs"
EVENTS_FILE = EVENTS_DIR / "runs.jsonl"

_lock = threading.Lock()
_run_id: Optional[str] = None


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def start_run(**fields: Any) -> str:
    """Open a new run. Emits run_start and sets the run_id used by subsequent emits."""
    global _run_id
    _run_id = f"r-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    emit("run_start", **fields)
    return _run_id


def emit(event: str, **fields: Any) -> None:
    """Append one event line to the JSONL log. No-op if no run is open."""
    if _run_id is None:
        return
    record = {"ts": _ts(), "run_id": _run_id, "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def end_run(**fields: Any) -> None:
    """Close the current run — emits run_end summary and clears run_id."""
    emit("run_end", **fields)
    global _run_id
    _run_id = None
