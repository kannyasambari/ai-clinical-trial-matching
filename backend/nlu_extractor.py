"""
backend/nlu_extractor.py
─────────────────────────
Natural Language Understanding layer.

Converts free-text like:
  "I'm a 28-year-old woman with irregular periods and PCOS. Not on medication."

Into structured data:
  {
    "age": 28, "sex": "female",
    "conditions": ["polycystic ovary syndrome"],
    "symptoms": ["irregular periods"],
    "medications": [],
    "missing_fields": [],
    "clarification_needed": false,
    "clarification_question": null
  }

Strategy:
  1. GPT-4o-mini extracts the JSON (fast, handles paraphrase, slang, symptoms)
  2. Rule-based post-processing normalises conditions via condition_normalizer
  3. Confidence scoring — if critical fields (conditions) are missing, sets
     clarification_needed=True with a specific follow-up question.
"""

import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from backend.condition_normalizer import normalize_conditions
from backend.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Extraction prompt ────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """You are a medical data extraction assistant.

Extract structured patient information from free-text input. Return ONLY valid JSON — no explanation, no markdown, no code fences.

JSON schema (all fields required, use null if unknown):
{
  "age": <integer or null>,
  "sex": <"male" | "female" | null>,
  "conditions": [<list of medical conditions as strings>],
  "symptoms": [<list of reported symptoms as strings>],
  "medications": [<list of medications/drugs as strings>],
  "medical_history": [<list of past diagnoses, surgeries, prior treatments>],
  "missing_fields": [<list of field names that could not be inferred: "age", "sex", "conditions">],
  "clarification_needed": <true | false>,
  "clarification_question": <string question to ask user, or null>
}

Rules:
- Extract conditions even when described symptomatically (e.g. "irregular periods + weight gain + suspected by doctor" → PCOS is a likely condition; include it under conditions if the user or doctor has named it)
- Symptoms are things the patient reports feeling; conditions are diagnoses
- If sex is ambiguous, set null
- If age is given as range ("late 30s"), take midpoint (35)
- Always normalise condition names to clinical terminology (e.g. "sugar problem" → "diabetes mellitus")
- If conditions list is empty AND no clear symptoms allow inference, set clarification_needed=true
- clarification_question should be specific and natural, e.g. "Could you tell me which condition your doctor is investigating, or what symptoms concern you most?"
- medications: include supplements and OTC drugs if mentioned
- Do NOT hallucinate conditions — only extract what is stated or clearly implied"""


def extract_patient_profile(
    user_message: str,
    chat_history: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Extract a structured patient profile from *user_message*.

    *chat_history* is a list of {"role": "user"|"assistant", "content": str}
    which allows the model to resolve pronouns and follow-ups
    ("Also I have insulin resistance" → merges with prior context).

    Returns a dict matching the schema above, plus a "raw_text" key.
    """
    messages: list[dict] = [{"role": "system", "content": _EXTRACT_SYSTEM}]

    # Inject prior turns so the model can resolve follow-ups
    if chat_history:
        for turn in chat_history[-6:]:  # last 3 exchanges = 6 messages
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
            timeout=20,
        )
        raw_json = response.choices[0].message.content.strip()
        profile  = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("NLU JSON parse error: %s", exc)
        profile = _empty_profile()
        profile["clarification_needed"]   = True
        profile["clarification_question"] = (
            "I had trouble understanding that. Could you describe your "
            "age, sex, and main medical condition?"
        )
    except Exception as exc:
        logger.error("NLU extraction error: %s", exc)
        profile = _empty_profile()
        profile["clarification_needed"]   = True
        profile["clarification_question"] = (
            "Something went wrong on my end. Could you rephrase — "
            "what condition are you seeking a trial for?"
        )

    # ── Post-process: normalise conditions through synonym map ────────────────
    raw_conditions: list[str] = profile.get("conditions") or []
    if raw_conditions:
        joined    = ", ".join(raw_conditions)
        canonical = normalize_conditions(joined)
        profile["conditions"] = canonical

    # ── Add raw_text for downstream use ──────────────────────────────────────
    profile["raw_text"] = user_message
    profile.setdefault("age",                    None)
    profile.setdefault("sex",                    None)
    profile.setdefault("conditions",             [])
    profile.setdefault("symptoms",               [])
    profile.setdefault("medications",            [])
    profile.setdefault("medical_history",        [])
    profile.setdefault("missing_fields",         [])
    profile.setdefault("clarification_needed",   False)
    profile.setdefault("clarification_question", None)

    # ── Validate: if no conditions AND no symptoms, force clarification ───────
    if not profile["conditions"] and not profile["symptoms"]:
        profile["clarification_needed"]   = True
        profile["clarification_question"] = profile.get("clarification_question") or (
            "I couldn't identify a specific condition from your message. "
            "Could you tell me which condition you're looking for trial options for?"
        )

    logger.info(
        "NLU extracted: age=%s sex=%s conditions=%s clarification=%s",
        profile.get("age"),
        profile.get("sex"),
        profile.get("conditions"),
        profile.get("clarification_needed"),
    )
    return profile


def merge_profiles(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Merge an NLU update into an existing patient profile.

    Rules:
    - Scalars (age, sex): update wins if not None
    - Lists (conditions, symptoms, medications): union, no duplicates
    - Preserve raw_text as the latest message
    - Re-evaluate clarification_needed after merge
    """
    merged = existing.copy()

    if update.get("age") is not None:
        merged["age"] = update["age"]
    if update.get("sex") is not None:
        merged["sex"] = update["sex"]

    for field in ("conditions", "symptoms", "medications", "medical_history"):
        existing_list = merged.get(field) or []
        update_list   = update.get(field) or []
        combined      = list(dict.fromkeys(existing_list + update_list))  # order-preserving dedup
        merged[field] = combined

    merged["raw_text"] = update.get("raw_text", existing.get("raw_text", ""))

    # Re-evaluate completeness
    missing = []
    if merged.get("age") is None:     missing.append("age")
    if merged.get("sex") is None:     missing.append("sex")
    if not merged.get("conditions"):  missing.append("conditions")

    merged["missing_fields"]         = missing
    merged["clarification_needed"]   = len(missing) > 0
    merged["clarification_question"] = (
        update.get("clarification_question")
        or _build_clarification_question(missing)
    )

    return merged


def build_patient_text(profile: dict[str, Any]) -> str:
    """Convert a structured profile dict into the narrative text sent to the RAG pipeline."""
    parts = ["Patient profile:"]

    if profile.get("age"):
        parts.append(f"Age: {profile['age']}")
    if profile.get("sex"):
        parts.append(f"Sex: {profile['sex']}")
    if profile.get("conditions"):
        parts.append(f"Conditions: {', '.join(profile['conditions'])}")
    if profile.get("symptoms"):
        parts.append(f"Symptoms: {', '.join(profile['symptoms'])}")
    if profile.get("medications"):
        parts.append(f"Medications: {', '.join(profile['medications'])}")
    if profile.get("medical_history"):
        parts.append(f"Medical history: {', '.join(profile['medical_history'])}")

    return "\n".join(parts)


def _empty_profile() -> dict[str, Any]:
    return {
        "age": None, "sex": None,
        "conditions": [], "symptoms": [],
        "medications": [], "medical_history": [],
        "missing_fields": ["age", "sex", "conditions"],
        "clarification_needed": True,
        "clarification_question": None,
    }


def _build_clarification_question(missing: list[str]) -> Optional[str]:
    if not missing:
        return None
    if "conditions" in missing:
        return (
            "Could you tell me which medical condition you're looking "
            "for trial options for?"
        )
    if "age" in missing and "sex" in missing:
        return "Could you share your age and biological sex so I can filter trials accurately?"
    if "age" in missing:
        return "Could you tell me your age? This helps filter trials by eligibility criteria."
    if "sex" in missing:
        return "Could you share your biological sex (male/female)? Some trials are sex-specific."
    return None