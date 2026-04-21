"""
backend/trial_gap_analyser.py
──────────────────────────────
Scans the GPT reasoning output from rag_reason() to detect what important
patient information is UNKNOWN or MISSING — and converts those gaps into
warm, human follow-up questions.

This is the engine behind the "ask before suggesting" behaviour.

Key responsibilities:
  1. extract_gaps(trial_results)
       → scans each trial's analysis text for UNKNOWN/missing markers
       → returns a deduplicated list of gap field names (e.g. "pregnancy_status")

  2. gaps_to_questions(gap_fields, conditions)
       → maps each gap field to a natural, friendly question
       → returns a list of question strings

  3. should_ask_before_showing(trial_results, profile)
       → decision function: True if gaps are significant enough to ask first
       → only asks once per session (uses profile["asked_gaps"] to track)

  4. build_followup_message(questions, conditions)
       → wraps the questions into Aria's warm conversational style
       → returns the full string sent to the user
"""

from dataclasses import field
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ─── Patterns that signal an unknown/missing value in the GPT reasoning ───────
_UNKNOWN_PATTERNS = [
    r"\bunknown\b",
    r"\bnot (mentioned|provided|specified|stated|given|known|disclosed)\b",
    r"\bnot available\b",
    r"\bno information\b",
    r"\binsufficient (data|information)\b",
    r"\bcannot (confirm|determine|assess|verify)\b",
    r"\bunable to (confirm|determine|assess|verify)\b",
    r"\bmissing\b",
    r"\bnot clear\b",
    r"\bunclear\b",
    r"\bN/A\b",
    r"→\s*unknown",
    r":\s*unknown",
]
_UNKNOWN_RE = re.compile("|".join(_UNKNOWN_PATTERNS), re.IGNORECASE)

# ─── Field detector patterns ─────────────────────────────────────────────────
# Maps a canonical field key → list of regex patterns that indicate
# that field is being discussed as unknown in the reasoning text.
_FIELD_PATTERNS: dict[str, list[str]] = {
    "pregnancy_status": [
        r"pregnan",
        r"gravid",
        r"childbearing",
        r"lactati",
        r"breastfeed",
        r"nursing",
        r"postpartum",
    ],
    "heart_history": [
        r"cardiac",
        r"cardiovascular",
        r"heart (disease|condition|failure|attack|history)",
        r"coronary",
        r"arrhythmia",
        r"myocardial",
        r"ECG",
        r"EKG",
    ],
    "liver_function": [
        r"hepatic",
        r"liver (function|disease|damage|enzymes)",
        r"ALT",
        r"AST",
        r"bilirubin",
        r"cirrhosis",
        r"hepatitis",
    ],
    "kidney_function": [
        r"renal",
        r"kidney (function|disease|failure)",
        r"creatinine",
        r"GFR",
        r"eGFR",
        r"CKD",
    ],
    "hba1c_level": [
        r"HbA1c",
        r"glycated haemoglobin",
        r"glycated hemoglobin",
        r"A1C",
        r"blood sugar control",
    ],
    "bmi_weight": [
        r"\bBMI\b",
        r"body mass index",
        r"weight\b",
        r"obesity",
        r"overweight",
    ],
    "prior_cancer_treatment": [
        r"prior (chemo|chemotherapy|radiation|immunotherapy|surgery)",
        r"previous (treatment|therapy|lines of therapy)",
        r"treatment.{0,20}history",
        r"prior.{0,20}treatment",
    ],
    "smoking_status": [
        r"smok",
        r"tobacco",
        r"nicotine",
        r"pack.year",
    ],
    "alcohol_use": [
        r"alcohol",
        r"drink",
        r"ethanol",
    ],
    "autoimmune_history": [
        r"autoimmune",
        r"lupus",
        r"rheumatoid",
        r"inflammatory (condition|disease|disorder)",
        r"immune.mediated",
    ],
    "current_infections": [
        r"active infection",
        r"HIV",
        r"hepatitis (B|C)",
        r"tuberculosis",
        r"\bTB\b",
    ],
    "performance_status": [
        r"ECOG",
        r"performance status",
        r"Karnofsky",
        r"functional status",
    ],
    "recent_surgery": [
        r"recent surgery",
        r"recent (operation|procedure|surgical)",
        r"post.?operative",
    ],
    "blood_counts": [
        r"haemoglobin",
        r"hemoglobin",
        r"platelet",
        r"neutrophil",
        r"white blood cell",
        r"\bWBC\b",
        r"\bANC\b",
        r"complete blood count",
        r"\bCBC\b",
    ],
    "hormone_receptor_status": [
        r"ER[\s/]PR",
        r"oestrogen receptor",
        r"estrogen receptor",
        r"progesterone receptor",
        r"HER2",
        r"hormone receptor",
    ],
    "genetic_mutations": [
        r"BRCA",
        r"EGFR",
        r"ALK",
        r"KRAS",
        r"PD.?L1",
        r"MSI",
        r"TMB",
        r"mutation status",
        r"genomic (testing|profiling)",
    ],
    "blood_pressure": [
        r"blood pressure",
        r"hypertension.*control",
        r"BP.*level",
    ],
    "diabetes_complications": [
        r"diabetic (nephropathy|retinopathy|neuropathy|foot)",
        r"diabetes.{0,20}complication",
        r"microvascular",
        r"macrovascular",
    ],
    "washout_period": [
        r"washout",
        r"prior.{0,30}(drug|medication|treatment).{0,20}(period|weeks|days|months)",
        r"last (dose|treatment|therapy).{0,20}(date|when|how long)",
    ],
}

# Pre-compile field patterns for speed
_COMPILED_FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
    field: [re.compile(p, re.IGNORECASE) for p in patterns]
    for field, patterns in _FIELD_PATTERNS.items()
}

# ─── Human-readable question map ─────────────────────────────────────────────
_FIELD_QUESTIONS: dict[str, str] = {
    "pregnancy_status":       "Are you currently pregnant, breastfeeding, or planning to become pregnant?",
    "heart_history":          "Do you have any history of heart conditions (like heart failure, arrhythmia, or coronary artery disease)?",
    "liver_function":         "Do you have any known liver conditions, or have you had abnormal liver function tests recently?",
    "kidney_function":        "Do you have any kidney disease or known kidney function issues?",
    "hba1c_level":            "Do you know your most recent HbA1c (blood sugar control) level?",
    "bmi_weight":             "Could you share your approximate height and weight (or BMI if you know it)?",
    "prior_cancer_treatment": "Have you received any prior cancer treatments — like chemotherapy, radiation, surgery, or immunotherapy?",
    "smoking_status":         "Do you currently smoke, or have you smoked in the past?",
    "alcohol_use":            "How much alcohol do you consume on a typical week?",
    "autoimmune_history":     "Do you have any autoimmune conditions (like lupus, rheumatoid arthritis, or inflammatory bowel disease)?",
    "current_infections":     "Do you have any active infections, or have you been tested for HIV, hepatitis B, or hepatitis C?",
    "performance_status":     "How would you describe your general ability to do daily activities — are you fully active, or are there things you struggle with?",
    "recent_surgery":         "Have you had any surgery or major medical procedures in the past few months?",
    "blood_counts":           "Do you have recent blood test results showing your haemoglobin, platelet, or white blood cell counts?",
    "hormone_receptor_status":"Do you know your tumour's hormone receptor status (ER/PR/HER2)?",
    "genetic_mutations":      "Has your tumour been tested for genetic mutations (like BRCA, EGFR, or PD-L1 expression)?",
    "blood_pressure":         "Is your blood pressure currently well-controlled?",
    "diabetes_complications":  "Have you experienced any diabetes complications like eye, kidney, or nerve problems?",
    "washout_period":         "When did you last take any medication or complete a course of treatment for this condition?",
}

# ─── Priority ordering (most clinically impactful first) ─────────────────────
_GAP_PRIORITY = list(_FIELD_QUESTIONS.keys())


def extract_gaps(trial_results: list[dict[str, Any]]) -> list[str]:
    """
    Scan each trial's analysis text for sentences that contain BOTH:
      (a) a known clinical field keyword
      (b) an unknown/missing marker

    Returns a deduplicated, priority-ordered list of gap field names.

    Example output: ["pregnancy_status", "heart_history", "hba1c_level"]
    """
    found_gaps: set[str] = set()

    for result in trial_results:
        analysis = result.get("analysis", "")
        if not analysis:
            continue

        # Split into sentences for localised matching
        sentences = re.split(r"[.\n;]", analysis)

        for sentence in sentences:
            # Only check sentences that contain an unknown marker
            if not _UNKNOWN_RE.search(sentence):
                continue

            # Now check which fields are discussed in this "unknown" sentence
            for field, compiled_patterns in _COMPILED_FIELD_PATTERNS.items():
                for pattern in compiled_patterns:
                    if pattern.search(sentence):
                        found_gaps.add(field)
                        break  # one match per field per sentence is enough

    # Return in priority order
    return [f for f in _GAP_PRIORITY if f in found_gaps]


def gaps_to_questions(
    gap_fields: list[str],
    conditions: list[str] | None = None,
    sex: str | None = None,
    max_questions: int = 4,
) -> list[str]:
    """
    Convert gap field names into natural-language questions.
    Caps at max_questions to avoid overwhelming the user.
    """
    questions = []

    for field in gap_fields:
    
    # ❌ Skip pregnancy questions for males
        if field == "pregnancy_status" and sex == "male":
            continue

        q = _FIELD_QUESTIONS.get(field)
        if q:
            questions.append(q)

        if len(questions) >= max_questions:
            break

    return questions


def should_ask_before_showing(
    trial_results: list[dict[str, Any]],
    profile: dict[str, Any],
    gap_fields: list[str],
) -> bool:
    """
    Decision gate: return True if we should ask follow-up questions
    instead of showing trials immediately.

    Rules:
    - Only ask if there are meaningful gaps (≥1 gap field found)
    - Only ask ONCE per set of gaps (track via profile["asked_gaps"])
    - If the user has already answered these exact gaps, proceed to results
    - If all trials are NO, still show them (no point asking)
    - If all trials are YES, show them immediately (no gaps block good matches)
    """
    if not gap_fields:
        return False

    # Don't ask if every trial is already a definitive YES or NO
    definitive = {"YES", "NO"}
    all_definitive = all(r.get("eligibility") in definitive for r in trial_results)
    if all_definitive:
        return False

    # Check if we've already asked about these exact gaps in this session
    already_asked: set[str] = set(profile.get("asked_gaps") or [])
    new_gaps = [g for g in gap_fields if g not in already_asked]

    return len(new_gaps) > 0


def build_followup_message(
    questions: list[str],
    conditions: list[str] | None = None,
) -> str:
    """
    Wrap the list of questions in Aria's warm, slightly playful voice.
    Returns a markdown-formatted string ready to send to the user.
    """
    condition_str = (
        f" for **{', '.join(conditions)}**" if conditions else ""
    )

    openers = [
        f"Okay, I found some trials that could be relevant{condition_str} — but before I share them, I just need a couple of quick things from you 👀",
        f"Hang on just a sec! I ran your profile against the trial database and found some potential matches{condition_str}. To make sure I'm not sending you down the wrong path, I need to check a few things first 💙",
        f"Good news — there are trials worth looking at{condition_str}! Before I walk you through them, help me fill in a couple of blanks so I can be more accurate 🔍",
    ]

    # Use a deterministic opener based on number of questions so it's reproducible
    opener = openers[len(questions) % len(openers)]

    lines = [opener, ""]

    for i, q in enumerate(questions, 1):
        lines.append(f"**{i}.** {q}")

    lines += [
        "",
        "Answer whatever you're comfortable with — even partial answers help me narrow things down for you 🙂",
    ]

    return "\n".join(lines)


def mark_gaps_asked(profile: dict[str, Any], gap_fields: list[str]) -> dict[str, Any]:
    """
    Record which gaps were asked so we don't ask them again next turn.
    Mutates and returns the profile dict.
    """
    existing: list[str] = list(profile.get("asked_gaps") or [])
    combined = list(dict.fromkeys(existing + gap_fields))
    profile["asked_gaps"] = combined
    return profile