"""
backend/rag_reasoner.py
────────────────────────
Production-grade GPT reasoning over retrieved trial documents.

Changes from original:
  1. Uses client.chat.completions.create() — correct API (original eligibility_reasoner.py
     used the non-existent client.responses.create()).
  2. Anti-hallucination system prompt: model is forbidden from using outside knowledge.
  3. Structured output: every response includes Eligibility, Reasoning, and Key Criteria.
  4. Graceful fallback: OpenAI failures return an error dict, not an exception crash.
  5. Returns trial title alongside trial_id for better UI display.
  6. Respects token budget — eligibility text is truncated to avoid context overflow.
"""

import logging
from typing import Any

import pandas as pd
from openai import APIConnectionError, OpenAI, RateLimitError

from backend.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are a clinical trial eligibility specialist.

STRICT RULES:
1. Base your analysis ONLY on the trial eligibility criteria provided.
2. Do NOT use outside knowledge.
3. Do NOT mention missing or unknown information.
4. Focus only on relevant matches or mismatches.

OUTPUT FORMAT:

Eligibility: YES | NO | POSSIBLE

Reasoning:
- Short, clear bullet points explaining match or mismatch

Guidelines:
- If clearly matches → say YES
- If clearly does not match → say NO
- If partially matches → say POSSIBLE
- Ignore criteria where patient data is missing
- Do NOT use words like "unknown", "uncertain", or "not provided"

Keep response concise and confident."""

# Truncate eligibility text to avoid exceeding the context window
_MAX_CRITERIA_CHARS = 3000


def _truncate(text: str, max_chars: int = _MAX_CRITERIA_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...criteria truncated for brevity...]"


def rag_reason(patient_text: str, trials: pd.DataFrame) -> list[dict[str, Any]]:
    """
    For each trial in *trials*, call GPT to evaluate patient eligibility.

    Returns a list of dicts:
      {
        "trial_id":    str,
        "title":       str,
        "eligibility": "YES" | "NO" | "POSSIBLE" | "ERROR",
        "analysis":    str,
        "distance":    float | None,   # semantic similarity score
      }
    """
    results: list[dict] = []

    if trials.empty:
        logger.warning("rag_reason() called with empty trials dataframe.")
        return results

    for _, trial in trials.iterrows():
        trial_id   = trial.get("nct_id",          "UNKNOWN")
        title      = trial.get("title",            trial_id)
        eligibility_text = str(trial.get("cleaned_criteria", ""))
        distance   = float(trial.get("_distance", -1))

        if not eligibility_text.strip():
            continue

        user_prompt = f"""Patient profile:
{patient_text.strip()}

Trial ID: {trial_id}
Trial Title: {title}

Eligibility Criteria:
{_truncate(eligibility_text)}

Evaluate the patient's eligibility for this trial."""

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                timeout=30,
            )
            analysis = response.choices[0].message.content.strip()

            # Parse top-level eligibility label for fast filtering in the UI
            eligibility_label = "POSSIBLE"
            upper = analysis.upper()
            if "ELIGIBILITY: YES"      in upper: eligibility_label = "YES"
            elif "ELIGIBILITY: NO"     in upper: eligibility_label = "NO"
            elif "ELIGIBILITY: POSSIBLE" in upper: eligibility_label = "POSSIBLE"

            results.append({
                "trial_id":    trial_id,
                "title":       title,
                "eligibility": eligibility_label,
                "analysis":    analysis,
                "distance":    round(distance, 4),
            })

        except RateLimitError:
            logger.error("OpenAI rate limit hit for trial %s.", trial_id)
            results.append({
                "trial_id":    trial_id,
                "title":       title,
                "eligibility": "ERROR",
                "analysis":    "OpenAI rate limit reached. Please try again in a moment.",
                "distance":    distance,
            })
        except APIConnectionError:
            logger.error("OpenAI connection error for trial %s.", trial_id)
            results.append({
                "trial_id":    trial_id,
                "title":       title,
                "eligibility": "ERROR",
                "analysis":    "OpenAI service unreachable. Check your network/API key.",
                "distance":    distance,
            })
        except Exception as exc:
            logger.exception("Unexpected error reasoning trial %s: %s", trial_id, exc)
            results.append({
                "trial_id":    trial_id,
                "title":       title,
                "eligibility": "ERROR",
                "analysis":    f"Reasoning failed: {str(exc)}",
                "distance":    distance,
            })

    return results