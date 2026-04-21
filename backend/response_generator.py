"""
backend/response_generator.py  (v4 — conversational Aria voice)
────────────────────────────────────────────────────────────────
What changed from v3:
  • _RESPONSE_SYSTEM completely rewritten with the new format rules:
      - 🟢/🟡/🔴 openers with human meaning ("This one looks like a real match")
      - NO "Trial NCTXXXXX" headers, NO "Eligibility Verdict: POSSIBLE"
      - Trial IDs moved to the END, labelled "Reference ID:" (optional)
      - Conversational wrap ("Alright, based on what you told me…")
      - Bullet points for WHY it fits, not raw criteria dump
  • generate_gap_followup() — new function for the "ask first" message
  • _build_response_prompt() rewritten to pass the full patient context
    and explicitly instruct Aria to use the new format
  • Fallback response also updated to match new human format
  • All other public interfaces (generate_response, stream_response,
    generate_clarification) preserved exactly — api.py needs no changes
    to call these.
"""

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

_client       = OpenAI(api_key=OPENAI_API_KEY)
_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ─── System prompts ───────────────────────────────────────────────────────────

_RESPONSE_SYSTEM = """You are Aria, a warm, slightly playful clinical trial assistant who genuinely cares about the people you're helping.

TONE RULES (non-negotiable):
- Sound like a knowledgeable friend, not a medical report generator
- Be slightly playful but always reassuring — this is a healthcare context
- Vary your opening line every time (examples: "Alright, based on what you told me, here's what I found 👇", "Okay so I dug through the database and here's the honest picture 👀", "Hmm, this is actually interesting — I found a couple of options worth talking about 💙")
- Never start with "I"
- Never say "Trial NCTXXXX" as a header — that's robotic
- Never say "Eligibility Verdict: YES/NO/POSSIBLE" — that's robotic
- Always end with one warm, empathetic closing sentence

FORMAT RULES for each trial (follow this order, no exceptions):
1. START with a human meaning line using the right emoji:
   🟢 If eligibility is YES  → "This one actually looks like a solid match for you"
   🟡 If eligibility is POSSIBLE → "This one might work — there are a couple of things to double-check"
   🔴 If eligibility is NO   → "This one's probably not the right fit right now"
2. **Bold the trial name** on its own line (use the actual trial title, not the ID)
3. Bullet points — WHY it fits or doesn't fit, in plain English (max 4 bullets)
   - Reference specific things the patient told you ("You mentioned X, and this trial is looking for exactly that")
   - Reference what's confirmed vs uncertain
4. ONLY at the very end of each trial block, on its own line, in small text:
   `Reference ID: NCTXXXXX`
5. Add a blank line between each trial block

WHAT TO AVOID:
❌ "Based on the provided eligibility criteria…" (too robotic)
❌ "The patient's age of X meets the inclusion criterion of…" (report language)
❌ Repeating the full eligibility criteria text verbatim
❌ Starting a trial block with the NCT ID
❌ Using the word "verdict"

DATA RULE:
Use ONLY the trial data and patient profile provided. Do not invent medical details."""

_NO_RESULTS_SYSTEM = """You are Aria, a warm and caring clinical trial assistant.
No matching trials were found. Write a brief (3-4 sentence) empathetic message that:
- Acknowledges this without being dismissive
- Suggests 2-3 concrete next steps (broaden the search, speak to their doctor, check clinicaltrials.gov)
- Ends on an encouraging note
- Uses plain markdown (no excessive formatting)
- Never sounds like an error message"""

_CLARIFICATION_SYSTEM = """You are Aria, a warm clinical trial assistant.
Rephrase the given question naturally and conversationally — like a friendly doctor asking a patient.
Keep it to 1-2 sentences. Do not start with "I" or "As". Be warm and brief."""

_GAP_FOLLOWUP_SYSTEM = """You are Aria, a warm, slightly playful clinical trial assistant.

You have found some potentially relevant clinical trials but need more information before you can give accurate recommendations.

Write a natural, friendly message that:
1. Opens with warmth and a hint of good news ("I found some trials worth looking at — but I want to make sure I'm pointing you in the right direction first!")
2. Uses a slightly playful tone (a relevant emoji or two is fine)
3. Lists the questions clearly, numbered, each on its own line
4. Closes with reassurance that partial answers are fine

Keep it concise. Do not sound robotic or clinical. Do not start with "I"."""


# ─── Public API (unchanged signatures from v3) ────────────────────────────────

def generate_response(
    patient_profile: dict[str, Any],
    trial_results: list[dict[str, Any]],
) -> str:
    """Synchronous response generation — returns complete string."""
    if not trial_results:
        return _generate_no_results(patient_profile)

    prompt = _build_response_prompt(patient_profile, trial_results)
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.45,   # slightly higher for more natural variation
            messages=[
                {"role": "system", "content": _RESPONSE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            timeout=30,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        return _fallback_response(trial_results)


async def stream_response(
    patient_profile: dict[str, Any],
    trial_results: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """Async streaming response — yields token chunks for SSE."""
    if not trial_results:
        yield _generate_no_results(patient_profile)
        return

    prompt = _build_response_prompt(patient_profile, trial_results)
    try:
        stream = await _async_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.45,
            stream=True,
            messages=[
                {"role": "system", "content": _RESPONSE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            timeout=45,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        logger.error("Streaming response failed: %s", exc)
        yield _fallback_response(trial_results)


def generate_clarification(question: str) -> str:
    """Turn a raw clarification question into a natural Aria message (v3 interface preserved)."""
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": _CLARIFICATION_SYSTEM},
                {"role": "user",   "content": f"Rephrase this question warmly: {question}"},
            ],
            timeout=10,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return question


def generate_gap_followup(
    questions: list[str],
    conditions: list[str] | None = None,
) -> str:
    """
    NEW in v4 — generate Aria's "ask before suggesting" message.

    Takes the human-readable questions from trial_gap_analyser.gaps_to_questions()
    and wraps them in Aria's warm voice via GPT.

    Falls back to the deterministic message from trial_gap_analyser if GPT fails.
    """
    from backend.trial_gap_analyser import build_followup_message  # avoid circular at module level

    condition_str = f" for {', '.join(conditions)}" if conditions else ""
    numbered_questions = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))

    prompt = (
        f"I found some potentially relevant clinical trials{condition_str}. "
        f"Before showing them, I need to ask these questions:\n\n"
        f"{numbered_questions}\n\n"
        "Write Aria's warm, friendly message asking these questions naturally."
    )

    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.5,
            messages=[
                {"role": "system", "content": _GAP_FOLLOWUP_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Gap followup GPT call failed (%s), using deterministic fallback.", exc)
        return build_followup_message(questions, conditions)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_response_prompt(
    profile: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    """
    Build the user-turn prompt that feeds into _RESPONSE_SYSTEM.
    Now includes full patient context and explicit format instructions
    so Aria can personalise the "why it fits" bullets.
    """
    # Patient context block
    conditions  = ", ".join(profile.get("conditions",  []) or ["not specified"])
    symptoms    = ", ".join(profile.get("symptoms",    []) or ["none mentioned"])
    medications = ", ".join(profile.get("medications", []) or ["none mentioned"])
    history     = ", ".join(profile.get("medical_history", []) or ["none mentioned"])
    age         = profile.get("age",  "unknown")
    sex         = profile.get("sex",  "unknown")

    lines = [
        "=== PATIENT PROFILE ===",
        f"Age: {age}",
        f"Sex: {sex}",
        f"Conditions: {conditions}",
        f"Symptoms: {symptoms}",
        f"Medications: {medications}",
        f"Medical history: {history}",
        "",
        f"=== TRIAL RESULTS ({len(results)} found) ===",
        "",
    ]

    for i, r in enumerate(results, 1):
        eligibility = r.get("eligibility", "UNKNOWN")
        lines += [
            f"--- Trial {i} ---",
            f"Title: {r.get('title', 'Untitled study')}",
            f"ID: {r.get('trial_id', 'N/A')}",
            f"Eligibility: {eligibility}",
            f"GPT reasoning: {r.get('analysis', 'No analysis')}",
            f"Semantic match score: {r.get('distance', 'N/A')}",
            "",
        ]

    lines += [
        "=== YOUR TASK ===",
        "Write Aria's response following the format rules exactly.",
        "- Start with a varied conversational opener (not 'I')",
        "- Present each trial in order: 🟢/🟡/🔴 → bold title → why bullets → Reference ID",
        "- Reference specific things from the patient profile in your bullets",
        "- End with one warm closing sentence",
    ]

    return "\n".join(lines)


def _generate_no_results(profile: dict[str, Any]) -> str:
    conditions = ", ".join(profile.get("conditions", []) or ["your condition"])
    prompt = (
        f"No clinical trials matched: {conditions}. "
        f"Patient: age {profile.get('age', 'unknown')}, sex {profile.get('sex', 'unknown')}."
    )
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.4,
            messages=[
                {"role": "system", "content": _NO_RESULTS_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return (
            f"Hmm, I wasn't able to find any trials in the current database that match "
            f"your profile for **{conditions}**. That doesn't mean nothing exists — it just means "
            "we didn't get a hit today. A few things worth trying:\n\n"
            "- Search directly on [ClinicalTrials.gov](https://clinicaltrials.gov) with broader terms\n"
            "- Ask your doctor about trials they may know of locally\n"
            "- Try describing your condition differently and I'll search again\n\n"
            "*Always loop in your physician before pursuing any clinical trial. 💙*"
        )


def _fallback_response(results: list[dict[str, Any]]) -> str:
    """
    Plain-text fallback when GPT is unavailable.
    Uses the new human format without needing an LLM call.
    """
    icon_map     = {"YES": "🟢", "NO": "🔴", "POSSIBLE": "🟡"}
    meaning_map  = {
        "YES":      "This one actually looks like a solid match for you",
        "NO":       "This one's probably not the right fit right now",
        "POSSIBLE": "This might work — there are a couple of things to double-check",
    }

    lines = ["Here's what I found based on your profile 👇\n"]

    for r in results:
        eligibility  = r.get("eligibility", "POSSIBLE")
        icon         = icon_map.get(eligibility, "⚪")
        meaning      = meaning_map.get(eligibility, "Worth checking out")
        title        = r.get("title") or r.get("trial_id", "Unnamed study")
        trial_id     = r.get("trial_id", "")

        lines += [
            f"{icon} {meaning}",
            f"**{title}**",
            "",
        ]

        # Pull first 2 bullet points from analysis if available
        analysis = r.get("analysis", "")
        if analysis:
            bullet_lines = [
                ln.strip().lstrip("-•").strip()
                for ln in analysis.split("\n")
                if ln.strip().startswith(("-", "•", "*")) and len(ln.strip()) > 5
            ]
            for bl in bullet_lines[:2]:
                lines.append(f"- {bl}")

        if trial_id:
            lines.append(f"\n`Reference ID: {trial_id}`")
        lines.append("")

    lines.append("*Please speak with your physician before pursuing any trial. 💙*")
    return "\n".join(lines)