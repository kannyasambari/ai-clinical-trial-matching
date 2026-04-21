"""
backend/eligibility_reasoner.py
─────────────────────────────────
Standalone eligibility evaluator (used by legacy patient_query.py).

BUG FIX: Original used client.responses.create() which does not exist.
Replaced with client.chat.completions.create() — the correct API.
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_SYSTEM = """You are a clinical trial eligibility assistant.

Rules:
- Use ONLY the provided eligibility criteria.
- Do NOT use outside knowledge.
- Do NOT mention missing or unknown information.
- Focus only on relevant matches or mismatches.

Output format:

Eligibility: YES / NO / POSSIBLE

Reasoning:
- Short, clear bullet points
- Only include useful, relevant points
- Be confident and concise

Guidelines:
- YES → strong match
- NO → clear mismatch
- POSSIBLE → partial match
- Ignore missing data completely (do NOT mention it)
"""


def evaluate_trial(patient_text: str, eligibility_text: str) -> str:
    prompt = f"""Patient profile:
{patient_text}

Clinical trial eligibility criteria:
{eligibility_text}

Determine:
1. Whether the patient is eligible.
2. Which inclusion criteria match.
3. Which exclusion criteria may disqualify the patient.

Respond in this format:

Eligibility: YES / NO / POSSIBLE

Reasoning:
- point 1
- point 2
- point 3"""

    # ── FIXED: was client.responses.create(model=..., input=...) ──────────────
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    output = response.choices[0].message.content

    # Clean unwanted words safely
    for word in ["unknown", "UNKNOWN", "Unknown"]:
        output = output.replace(word, "")

    # Clean extra spaces / broken lines
    output = "\n".join(line.strip() for line in output.splitlines() if line.strip())

    return output
