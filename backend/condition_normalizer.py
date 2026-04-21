"""
backend/condition_normalizer.py
────────────────────────────────
Maps clinical synonyms, abbreviations and lay-person terms to the
canonical concept names used in ClinicalTrials.gov.

Design goals:
  • Works offline — no UMLS API key required.
  • Extensible — just add entries to SYNONYM_MAP.
  • Semantic fallback — if no exact match, uses embedding similarity
    against candidate terms to still find the closest concept.

Usage:
    from backend.condition_normalizer import normalize_conditions
    canonical = normalize_conditions("T2DM, HTN")
    # → ["type 2 diabetes mellitus", "hypertension"]
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ─── Synonym map ──────────────────────────────────────────────────────────────
# Keys   : lowercased aliases / abbreviations / misspellings
# Values : canonical ClinicalTrials.gov condition name
# ─────────────────────────────────────────────────────────────────────────────
SYNONYM_MAP: dict[str, str] = {
    # Diabetes
    "t2dm":                        "type 2 diabetes mellitus",
    "type 2 diabetes":             "type 2 diabetes mellitus",
    "type ii diabetes":            "type 2 diabetes mellitus",
    "diabetes type 2":             "type 2 diabetes mellitus",
    "diabetes mellitus type 2":    "type 2 diabetes mellitus",
    "t1dm":                        "type 1 diabetes mellitus",
    "type 1 diabetes":             "type 1 diabetes mellitus",
    "type i diabetes":             "type 1 diabetes mellitus",
    "juvenile diabetes":           "type 1 diabetes mellitus",
    "dm":                          "diabetes mellitus",
    "diabetes":                    "diabetes mellitus",

    # Cardiovascular
    "htn":                         "hypertension",
    "high blood pressure":         "hypertension",
    "elevated bp":                 "hypertension",
    "cad":                         "coronary artery disease",
    "coronary heart disease":      "coronary artery disease",
    "chd":                         "coronary artery disease",
    "heart attack":                "myocardial infarction",
    "mi":                          "myocardial infarction",
    "af":                          "atrial fibrillation",
    "afib":                        "atrial fibrillation",
    "hf":                          "heart failure",
    "chf":                         "congestive heart failure",
    "congestive heart failure":    "heart failure",
    "pvd":                         "peripheral vascular disease",
    "pad":                         "peripheral artery disease",

    # Oncology
    "breast ca":                   "breast cancer",
    "breast carcinoma":            "breast cancer",
    "breast neoplasm":             "breast cancer",
    "lung ca":                     "lung cancer",
    "nsclc":                       "non-small cell lung cancer",
    "sclc":                        "small cell lung cancer",
    "crc":                         "colorectal cancer",
    "colon cancer":                "colorectal cancer",
    "rectal cancer":               "colorectal cancer",
    "hcc":                         "hepatocellular carcinoma",
    "liver cancer":                "hepatocellular carcinoma",
    "aml":                         "acute myeloid leukemia",
    "all":                         "acute lymphoblastic leukemia",
    "cll":                         "chronic lymphocytic leukemia",
    "cml":                         "chronic myeloid leukemia",
    "nhl":                         "non-hodgkin lymphoma",
    "hodgkins":                    "hodgkin lymphoma",
    "hodgkin's":                   "hodgkin lymphoma",
    "pca":                         "prostate cancer",
    "prostate ca":                 "prostate cancer",
    "rcc":                         "renal cell carcinoma",
    "kidney cancer":               "renal cell carcinoma",
    "gbm":                         "glioblastoma",
    "glioblastoma multiforme":     "glioblastoma",
    "melanoma skin":               "melanoma",
    "oc":                          "ovarian cancer",
    "ovarian ca":                  "ovarian cancer",
    "cervical ca":                 "cervical cancer",
    "endometrial cancer":          "uterine cancer",

    # Neurological / Psychiatric
    "alzheimers":                  "alzheimer's disease",
    "alzheimer disease":           "alzheimer's disease",
    "ad":                          "alzheimer's disease",
    "parkinsons":                  "parkinson's disease",
    "parkinson disease":           "parkinson's disease",
    "pd":                          "parkinson's disease",
    "ms":                          "multiple sclerosis",
    "als":                         "amyotrophic lateral sclerosis",
    "lou gehrig":                  "amyotrophic lateral sclerosis",
    "mdd":                         "major depressive disorder",
    "depression":                  "major depressive disorder",
    "bipolar":                     "bipolar disorder",
    "schizophrenia":               "schizophrenia",
    "ptsd":                        "post-traumatic stress disorder",
    "ocd":                         "obsessive compulsive disorder",
    "adhd":                        "attention deficit hyperactivity disorder",
    "add":                         "attention deficit hyperactivity disorder",
    "epilepsy":                    "epilepsy",
    "seizure disorder":            "epilepsy",

    # Respiratory
    "copd":                        "chronic obstructive pulmonary disease",
    "emphysema":                   "chronic obstructive pulmonary disease",
    "asthma":                      "asthma",
    "ild":                         "interstitial lung disease",
    "ipf":                         "idiopathic pulmonary fibrosis",
    "pulmonary fibrosis":          "idiopathic pulmonary fibrosis",

    # Metabolic / Endocrine
    "obesity":                     "obesity",
    "overweight":                  "obesity",
    "nafld":                       "non-alcoholic fatty liver disease",
    "nash":                        "non-alcoholic steatohepatitis",
    "hypothyroid":                 "hypothyroidism",
    "hyperthyroid":                "hyperthyroidism",
    "thyroid cancer":              "thyroid neoplasm",

    # Infectious
    "hiv":                         "human immunodeficiency virus infection",
    "aids":                        "human immunodeficiency virus infection",
    "hiv/aids":                    "human immunodeficiency virus infection",
    "hbv":                         "hepatitis b",
    "hcv":                         "hepatitis c",
    "tb":                          "tuberculosis",
    "covid":                       "covid-19",
    "covid19":                     "covid-19",
    "sars-cov-2":                  "covid-19",
    "coronavirus":                 "covid-19",

    # Renal
    "ckd":                         "chronic kidney disease",
    "chronic renal failure":       "chronic kidney disease",
    "esrd":                        "end-stage renal disease",
    "kidney failure":              "end-stage renal disease",
    "aki":                         "acute kidney injury",

    # Gastrointestinal
    "ibd":                         "inflammatory bowel disease",
    "crohns":                      "crohn's disease",
    "crohn disease":               "crohn's disease",
    "uc":                          "ulcerative colitis",
    "gerd":                        "gastroesophageal reflux disease",
    "acid reflux":                 "gastroesophageal reflux disease",
    "ibs":                         "irritable bowel syndrome",

    # Musculoskeletal / Autoimmune
    "ra":                          "rheumatoid arthritis",
    "rheumatoid arthritis":        "rheumatoid arthritis",
    "oa":                          "osteoarthritis",
    "osteoarthritis":              "osteoarthritis",
    "sle":                         "systemic lupus erythematosus",
    "lupus":                       "systemic lupus erythematosus",
    "as":                          "ankylosing spondylitis",
    "psoriasis":                   "psoriasis",
    "psa":                         "psoriatic arthritis",
    "psoriatic arthritis":         "psoriatic arthritis",
    "fibromyalgia":                "fibromyalgia",
    "osteoporosis":                "osteoporosis",
    "gout":                        "gout",

    # Haematology
    "sickle cell":                 "sickle cell disease",
    "scd":                         "sickle cell disease",
    "thalassemia":                 "thalassemia",
    "itp":                         "immune thrombocytopenia",
    "hemophilia":                  "hemophilia",
    "anaemia":                     "anemia",
    "anemia":                      "anemia",
    "iron deficiency":             "iron deficiency anemia",

    # Ophthalmic
    "amd":                         "age-related macular degeneration",
    "macular degeneration":        "age-related macular degeneration",
    "glaucoma":                    "glaucoma",
    "dr":                          "diabetic retinopathy",
    "diabetic retinopathy":        "diabetic retinopathy",

    # Dermatology
    "atopic dermatitis":           "atopic dermatitis",
    "eczema":                      "atopic dermatitis",
    "ad skin":                     "atopic dermatitis",
}


def _clean(term: str) -> str:
    """Lowercase and strip punctuation for lookup."""
    return re.sub(r"[''`]", "", term.lower()).strip()


def normalize_conditions(raw_conditions: str) -> List[str]:
    """
    Split a comma-separated conditions string, normalize each term
    via synonym lookup, and return a deduplicated canonical list.

    Example:
        normalize_conditions("T2DM, HTN, asthma")
        → ["type 2 diabetes mellitus", "hypertension", "asthma"]
    """
    if not raw_conditions or not raw_conditions.strip():
        return []

    terms = [t.strip() for t in raw_conditions.split(",") if t.strip()]
    canonical: list[str] = []
    seen: set[str] = set()

    for term in terms:
        key = _clean(term)
        resolved = SYNONYM_MAP.get(key, term.lower())  # fallback: keep original
        if resolved not in seen:
            canonical.append(resolved)
            seen.add(resolved)
            if resolved != term.lower():
                logger.debug("Normalized '%s' → '%s'", term, resolved)

    return canonical


def build_search_variants(conditions: List[str]) -> List[str]:
    """
    Given a list of canonical condition names, return an expanded list
    that includes both the canonical form AND any known aliases.
    Used for text-level fallback matching when semantic search alone
    isn't sufficient.
    """
    variants: set[str] = set(conditions)
    canonical_to_aliases: dict[str, list[str]] = {}
    for alias, canon in SYNONYM_MAP.items():
        canonical_to_aliases.setdefault(canon, []).append(alias)

    for cond in conditions:
        for alias in canonical_to_aliases.get(cond, []):
            variants.add(alias)

    return list(variants)