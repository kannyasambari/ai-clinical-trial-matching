"""
backend/session_manager.py  (v4 — gap-aware session state)
────────────────────────────────────────────────────────────
What changed from v3:
  - Session dict now includes two new fields:
      "asked_gaps":     list[str]  — gap fields already asked this session
      "pending_trials": list[dict] — trial results held back pending user answer

  - Two new functions:
      store_pending_trials(session_id, trial_results)
      get_and_clear_pending_trials(session_id) → list[dict]

  - session_summary() now exposes asked_gaps count

  Everything else is identical to v3 — all existing callers unaffected.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

SESSION_TTL_MINUTES = 60
_MAX_HISTORY_TURNS  = 20

_sessions: dict[str, dict[str, Any]] = {}


def create_session() -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "session_id":      sid,
        "chat_history":    [],
        "patient_profile": {},
        "search_count":    0,
        # ── v4 additions ──────────────────────────────────────────────────────
        "asked_gaps":      [],      # gap fields we've already questioned about
        "pending_trials":  [],      # trial results held while asking follow-ups
        # ─────────────────────────────────────────────────────────────────────
        "created_at":      _now(),
        "updated_at":      _now(),
    }
    logger.info("Session created: %s", sid)
    return sid


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    return _sessions.get(session_id)


def get_or_create_session(session_id: Optional[str]) -> tuple[str, dict[str, Any]]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    new_id = create_session()
    return new_id, _sessions[new_id]


def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
) -> None:
    session = _sessions.get(session_id)
    if not session:
        return
    entry: dict[str, Any] = {
        "role":      role,
        "content":   content,
        "timestamp": _now(),
    }
    if metadata:
        entry["metadata"] = metadata
    session["chat_history"].append(entry)

    if len(session["chat_history"]) > _MAX_HISTORY_TURNS * 2:
        session["chat_history"] = session["chat_history"][-(  _MAX_HISTORY_TURNS * 2):]

    session["updated_at"] = _now()


def update_patient_profile(session_id: str, profile: dict[str, Any]) -> None:
    session = _sessions.get(session_id)
    if session:
        session["patient_profile"] = profile
        session["updated_at"]      = _now()


def increment_search_count(session_id: str) -> int:
    session = _sessions.get(session_id)
    if session:
        session["search_count"] += 1
        return session["search_count"]
    return 0


def get_chat_history(session_id: str, last_n: int = 10) -> list[dict]:
    session = _sessions.get(session_id)
    if not session:
        return []
    history = session["chat_history"]
    return [{"role": m["role"], "content": m["content"]} for m in history[-last_n:]]


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    logger.info("Session deleted: %s", session_id)


# ── v4: pending trials ────────────────────────────────────────────────────────

def store_pending_trials(session_id: str, trial_results: list[dict[str, Any]]) -> None:
    """
    Hold trial results while Aria asks follow-up gap questions.
    On the user's next message, these are retrieved and shown.
    """
    session = _sessions.get(session_id)
    if session:
        session["pending_trials"] = trial_results
        session["updated_at"]     = _now()
        logger.debug("Stored %d pending trials for session %s", len(trial_results), session_id)


def get_and_clear_pending_trials(session_id: str) -> list[dict[str, Any]]:
    """
    Retrieve pending trials and clear them from the session.
    Returns empty list if none stored.
    """
    session = _sessions.get(session_id)
    if not session:
        return []
    trials = session.get("pending_trials") or []
    session["pending_trials"] = []
    session["updated_at"]     = _now()
    return trials


def has_pending_trials(session_id: str) -> bool:
    session = _sessions.get(session_id)
    if not session:
        return False
    return bool(session.get("pending_trials"))


# ── v4: gap tracking ──────────────────────────────────────────────────────────

def get_asked_gaps(session_id: str) -> list[str]:
    session = _sessions.get(session_id)
    return list(session.get("asked_gaps") or []) if session else []


def add_asked_gaps(session_id: str, gap_fields: list[str]) -> None:
    """Record which gap fields have been asked so we don't re-ask them."""
    session = _sessions.get(session_id)
    if session:
        existing = session.get("asked_gaps") or []
        combined = list(dict.fromkeys(existing + gap_fields))
        session["asked_gaps"] = combined
        session["updated_at"] = _now()


# ── Summary ───────────────────────────────────────────────────────────────────

def session_summary(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        return {}
    p = session.get("patient_profile", {})
    return {
        "session_id":     session_id,
        "search_count":   session["search_count"],
        "age":            p.get("age"),
        "sex":            p.get("sex"),
        "conditions":     p.get("conditions", []),
        "symptoms":       p.get("symptoms", []),
        "medications":    p.get("medications", []),
        "message_count":  len(session["chat_history"]),
        "asked_gaps":     session.get("asked_gaps", []),      # v4
        "has_pending":    bool(session.get("pending_trials")), # v4
        "created_at":     session["created_at"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()