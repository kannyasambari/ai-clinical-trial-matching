"""
backend/api.py  (v4 — ask before suggesting)
──────────────────────────────────────────────
What changed from v3:

CORE NEW BEHAVIOUR — "ask before suggesting":
  After rag_reason() returns trial results, the /chat endpoint now runs
  trial_gap_analyser.extract_gaps() to scan every trial's reasoning text
  for UNKNOWN / MISSING fields.

  If meaningful gaps exist AND the user hasn't already answered them:
    → Store trial results in session ("pending_trials")
    → Return Aria's friendly follow-up question message
    → DO NOT show trials yet

  On the NEXT user message:
    → NLU merges the new answers into profile
    → get_and_clear_pending_trials() retrieves the held results
    → Re-run rag_reason() on the updated profile
    → NOW generate and stream the trial response

New imports:
  from backend.trial_gap_analyser import (
      extract_gaps, gaps_to_questions, should_ask_before_showing, mark_gaps_asked
  )
  from backend.response_generator import generate_gap_followup   (new function)
  from backend.session_manager import (
      store_pending_trials, get_and_clear_pending_trials,
      has_pending_trials, add_asked_gaps, get_asked_gaps
  )

Everything else (auth, /match_trials, /health, /cache/*, session endpoints)
is IDENTICAL to v3 — no breaking changes.
"""

import asyncio
import json
import logging
import os

os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import timedelta
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator

from backend.auth import (
    UserInDB,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    require_role,
)
from backend.cache import query_cache
from backend.config import BACKEND_URL, JWT_ACCESS_TOKEN_EXPIRE_MINUTES, RETRIEVAL_K
from backend.logger import logging_middleware, setup_logging
from backend.nlu_extractor import (
    build_patient_text,
    extract_patient_profile,
    merge_profiles,
)
from backend.rag_pipeline import TrialRetriever
from backend.rag_reasoner import rag_reason
from backend.response_generator import (
    generate_gap_followup,    # v4 new
    generate_response,
    stream_response,
)
from backend.session_manager import (
    add_message,
    add_asked_gaps,                   # v4 new
    delete_session,
    get_and_clear_pending_trials,     # v4 new
    get_asked_gaps,                   # v4 new
    get_chat_history,
    get_or_create_session,
    get_session,
    has_pending_trials,               # v4 new
    increment_search_count,
    session_summary,
    store_pending_trials,             # v4 new
    update_patient_profile,
)
from backend.trial_gap_analyser import (  # v4 new
    extract_gaps,
    gaps_to_questions,
    mark_gaps_asked,
    should_ask_before_showing,
)

setup_logging()
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Trial Assistant API",
    description="Conversational AI system for clinical trial matching.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        BACKEND_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)

retriever = TrialRetriever()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role:     str = Field(default="patient")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("admin", "doctor", "patient"):
            raise ValueError("role must be admin, doctor, or patient")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str


class ChatRequest(BaseModel):
    message:    str            = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream:     bool           = False


class ChatResponse(BaseModel):
    session_id:           str
    reply:                str
    patient_profile:      dict
    trials_found:         int
    search_performed:     bool
    clarification_needed: bool
    # v4: tells the frontend whether we're in gap-asking mode
    awaiting_gap_answers: bool = False


class PatientQuery(BaseModel):
    age:         int           = Field(..., ge=0, le=150)
    sex:         str
    conditions:  str           = Field(..., min_length=1)
    medications: str           = Field(default="")
    location:    Optional[str] = None

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v


class TrialResult(BaseModel):
    trial_id:    str
    title:       str
    eligibility: str
    analysis:    str
    distance:    Optional[float] = None


class MatchResponse(BaseModel):
    total_found: int
    page:        int
    page_size:   int
    results:     list[TrialResult]
    cached:      bool = False


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    try:
        create_user(req.username, req.password, req.role)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    return {"message": f"User '{req.username}' created."}


@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token, token_type="bearer", role=user.role)


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


# ─── Main conversational endpoint ─────────────────────────────────────────────

@app.post("/chat")
async def chat(
    req: ChatRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Conversational endpoint — v4 flow:

    ┌─────────────────────────────────────────────────────────────────┐
    │  GATE 1: Profile completeness (NLU clarification)               │
    │  If age / sex / conditions missing → ask (same as v3)           │
    └────────────────────────────────┬────────────────────────────────┘
                                     │ profile complete
    ┌────────────────────────────────▼────────────────────────────────┐
    │  GATE 2: Pending trials from last gap-question turn?            │
    │  If yes → re-reason with updated profile → show results         │
    └────────────────────────────────┬────────────────────────────────┘
                                     │ no pending trials
    ┌────────────────────────────────▼────────────────────────────────┐
    │  FAISS retrieval + GPT reasoning                                │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  GATE 3: Gap analysis                                           │
    │  If significant unknowns found AND not yet asked:               │
    │    → Store trials as pending                                    │
    │    → Return Aria's follow-up question message                   │
    │  Else:                                                          │
    │    → Generate and stream/return full trial response             │
    └─────────────────────────────────────────────────────────────────┘
    """
    session_id, session = get_or_create_session(req.session_id)
    history = get_chat_history(session_id, last_n=8)

    # ── 1. Save user message ─────────────────────────────────────────────────
    add_message(session_id, "user", req.message)

    # ── 2. NLU: extract + merge profile ──────────────────────────────────────
    extracted        = extract_patient_profile(req.message, history)
    existing_profile = session.get("patient_profile") or {}
    merged_profile   = merge_profiles(existing_profile, extracted) if existing_profile else extracted

    # Inject already-asked gaps into profile so should_ask_before_showing
    # can detect them without a separate session lookup
    merged_profile["asked_gaps"] = get_asked_gaps(session_id)

    update_patient_profile(session_id, merged_profile)

    # ── GATE 1: profile completeness clarification (same as v3) ──────────────
    if merged_profile.get("clarification_needed"):
        raw_q = merged_profile.get("clarification_question") or (
            "Could you provide more details about your medical condition?"
        )
        reply = (raw_q)
        add_message(session_id, "assistant", reply)

        body = ChatResponse(
            session_id=session_id,
            reply=reply,
            patient_profile=_safe_profile(merged_profile),
            trials_found=0,
            search_performed=False,
            clarification_needed=True,
            awaiting_gap_answers=False,
        )
        if req.stream:
            return StreamingResponse(
                _stream_json(body.model_dump()),
                media_type="text/event-stream",
            )
        return body

    # ── Build core query inputs ───────────────────────────────────────────────
    patient_text   = build_patient_text(merged_profile)
    conditions_str = ", ".join(merged_profile.get("conditions") or [])
    conditions_list = merged_profile.get("conditions") or []

    # ── GATE 2: does the user's answer resolve a previous gap-question? ───────
    # If there are pending trials, re-reason with the enriched profile
    # (the new info the user just provided is now in merged_profile)
    pending_trials = get_and_clear_pending_trials(session_id)
    if pending_trials:
        logger.info(
            "Session %s: user answered gap questions. Re-reasoning %d pending trials.",
            session_id, len(pending_trials),
        )
        trial_results = _re_reason_pending(patient_text, pending_trials)
        increment_search_count(session_id)
        return await _emit_response(
            req, session_id, merged_profile, trial_results,
            cache_key=_make_cache_key(merged_profile),
            awaiting_gap_answers=False,
        )

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key     = _make_cache_key(merged_profile)
    cached_result = query_cache.get(**cache_key)
    if cached_result:
        reply = cached_result["reply"]
        add_message(session_id, "assistant", reply, {"cached": True})
        increment_search_count(session_id)
        body = ChatResponse(
            session_id=session_id,
            reply=reply,
            patient_profile=_safe_profile(merged_profile),
            trials_found=cached_result.get("trials_found", 0),
            search_performed=True,
            clarification_needed=False,
            awaiting_gap_answers=False,
        )
        if req.stream:
            return StreamingResponse(
                _stream_json(body.model_dump()), media_type="text/event-stream"
            )
        return body

    # ── FAISS retrieval ───────────────────────────────────────────────────────
    try:
        trials = retriever.retrieve_trials(
            patient_text=patient_text,
            raw_conditions=conditions_str,
            age=merged_profile.get("age"),
            sex=merged_profile.get("sex"),
            k=RETRIEVAL_K,
        )
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))

    # ── GPT reasoning ─────────────────────────────────────────────────────────
    trial_results: list[dict] = []
    if not trials.empty:
        try:
            trial_results = rag_reason(patient_text, trials)
        except Exception as exc:
            logger.error("Reasoning failed: %s", exc)

    increment_search_count(session_id)

    # ── GATE 3: gap analysis — should we ask before showing? ─────────────────
    if trial_results:
        gap_fields = extract_gaps(trial_results)
        logger.info("Session %s: gap fields detected: %s", session_id, gap_fields)

        if should_ask_before_showing(trial_results, merged_profile, gap_fields):
            # Hold the trials — we'll use them when the user replies
            store_pending_trials(session_id, trial_results)
            add_asked_gaps(session_id, gap_fields)
            mark_gaps_asked(merged_profile, gap_fields)
            update_patient_profile(session_id, merged_profile)

            questions = gaps_to_questions(gap_fields, conditions_list, sex=merged_profile.get("sex"))

            reply = "I need a bit more info to match better trials:\n\n"
            reply += "\n".join([f"• {q}" for q in questions[:4]])
            reply += "\n\nAnswer whatever you can."
            add_message(session_id, "assistant", reply)

            body = ChatResponse(
                session_id=session_id,
                reply=reply,
                patient_profile=_safe_profile(merged_profile),
                trials_found=len(trial_results),   # found but not shown yet
                search_performed=True,
                clarification_needed=False,
                awaiting_gap_answers=True,          # v4 flag for frontend
            )
            if req.stream:
                return StreamingResponse(
                    _stream_json(body.model_dump()), media_type="text/event-stream"
                )
            return body

    # ── No gaps (or all definitive) — emit response now ──────────────────────
    return await _emit_response(
        req, session_id, merged_profile, trial_results,
        cache_key=cache_key,
        awaiting_gap_answers=False,
    )


# ─── Session endpoints (unchanged from v3) ────────────────────────────────────

@app.get("/chat/session")
def get_session_info(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    summary = session_summary(session_id)
    if not summary:
        raise HTTPException(404, detail="Session not found.")
    return summary


@app.delete("/chat/session")
def reset_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    delete_session(session_id)
    return {"message": "Session reset."}


@app.get("/chat/history")
def chat_history(
    session_id: str,
    last_n: int = Query(default=20, ge=1, le=100),
    current_user: UserInDB = Depends(get_current_user),
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found.")
    history = session.get("chat_history", [])[-last_n:]
    return {"session_id": session_id, "messages": history}


# ─── Legacy structured endpoint (unchanged from v3) ───────────────────────────

@app.post("/match_trials", response_model=MatchResponse)
def match_trials(
    patient: PatientQuery,
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=RETRIEVAL_K, ge=1, le=20),
    current_user: UserInDB = Depends(get_current_user),
):
    patient_text = (
        f"Patient summary:\nAge: {patient.age}\nSex: {patient.sex}\n"
        f"Conditions: {patient.conditions}\n"
        f"Medications: {patient.medications or 'none listed'}\n"
    )
    cache_params = dict(
        age=patient.age, sex=patient.sex,
        conditions=patient.conditions.lower(),
        medications=(patient.medications or "").lower(),
        page=page, page_size=page_size,
    )
    cached = query_cache.get(**cache_params)
    if cached:
        cached["cached"] = True
        return cached

    try:
        trials = retriever.retrieve_trials(
            patient_text=patient_text,
            raw_conditions=patient.conditions,
            age=patient.age,
            sex=patient.sex,
            k=page_size * page,
        )
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))

    if trials.empty:
        return MatchResponse(total_found=0, page=page, page_size=page_size, results=[])

    raw_results = rag_reason(patient_text, trials)
    total       = len(raw_results)
    start       = (page - 1) * page_size
    response    = MatchResponse(
        total_found=total, page=page, page_size=page_size,
        results=[TrialResult(**r) for r in raw_results[start: start + page_size]],
    )
    query_cache.set(response.model_dump(), **cache_params)
    return response


# ─── Admin endpoints (unchanged) ─────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats(admin: UserInDB = Depends(require_role("admin"))):
    return {"cached_entries": query_cache.size}


@app.post("/cache/clear")
def cache_clear(admin: UserInDB = Depends(require_role("admin"))):
    query_cache.clear()
    return {"message": "Cache cleared."}


# ─── Internal helpers ────────────────────────────────────────────────────────

def _safe_profile(profile: dict) -> dict:
    return {
        "age":        profile.get("age"),
        "sex":        profile.get("sex"),
        "conditions": profile.get("conditions", []),
        "symptoms":   profile.get("symptoms", []),
        "medications": profile.get("medications", []),
    }


def _make_cache_key(profile: dict) -> dict:
    return dict(
        conditions=", ".join(profile.get("conditions") or []).lower(),
        age=profile.get("age"),
        sex=(profile.get("sex") or "").lower(),
        medications=", ".join(profile.get("medications") or []).lower(),
    )


def _re_reason_pending(
    patient_text: str,
    pending: list[dict],
) -> list[dict]:
    """
    Re-run GPT reasoning on the pending trial set using the enriched
    patient profile text. This gives a fresh eligibility assessment
    now that the user has answered gap questions.

    Falls back to returning the original pending results unchanged
    if rag_reason raises (so we never lose the trials entirely).
    """
    import pandas as pd
    try:
        # Reconstruct a minimal DataFrame with the fields rag_reason expects
        rows = []
        for t in pending:
            rows.append({
                "nct_id":          t.get("trial_id", ""),
                "title":           t.get("title", ""),
                "cleaned_criteria": t.get("_criteria_text", ""),  # stored below
                "_distance":       t.get("distance", -1),
            })
        df = pd.DataFrame(rows)
        re_reasoned = rag_reason(patient_text, df)
        return re_reasoned
    except Exception as exc:
        logger.warning("Re-reasoning failed (%s), returning original pending results.", exc)
        return pending


async def _emit_response(
    req: ChatRequest,
    session_id: str,
    profile: dict,
    trial_results: list[dict],
    cache_key: dict,
    awaiting_gap_answers: bool,
) -> ChatResponse | StreamingResponse:
    """
    Final step: generate Aria's trial presentation and return/stream it.
    Shared between the direct path and the post-gap-answer path.
    """
    if req.stream:
        async def _generate_and_store():
            full_reply = []
            async for chunk in stream_response(profile, trial_results):
                full_reply.append(chunk)
                event_data = json.dumps({"type": "token", "content": chunk})
                yield f"data: {event_data}\n\n"

            complete_reply = "".join(full_reply)
            add_message(session_id, "assistant", complete_reply)
            query_cache.set(
                {"reply": complete_reply, "trials_found": len(trial_results)},
                **cache_key,
            )
            meta = json.dumps({
                "type":               "done",
                "session_id":         session_id,
                "trials_found":       len(trial_results),
                "awaiting_gap_answers": awaiting_gap_answers,
                "patient_profile":    _safe_profile(profile),
            })
            yield f"data: {meta}\n\n"

        return StreamingResponse(
            _generate_and_store(),
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )

    # Sync path
    reply = generate_response(profile, trial_results)
    add_message(session_id, "assistant", reply)
    query_cache.set(
        {"reply": reply, "trials_found": len(trial_results)},
        **cache_key,
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        patient_profile=_safe_profile(profile),
        trials_found=len(trial_results),
        search_performed=True,
        clarification_needed=False,
        awaiting_gap_answers=awaiting_gap_answers,
    )


async def _stream_json(data: dict) -> AsyncIterator[str]:
    event_data = json.dumps({"type": "done", **data})
    yield f"data: {event_data}\n\n"