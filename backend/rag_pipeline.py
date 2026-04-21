# backend/rag_pipeline.py

import re
import logging
import pickle
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from backend.config import (
    EMBEDDINGS_PKL_PATH,
    FAISS_INDEX_PATH,
    FAISS_SIMILARITY_THRESHOLD,
    RETRIEVAL_K,
    SEARCH_MULTIPLIER,
)
from backend.condition_normalizer import build_search_variants, normalize_conditions
from backend.embedding_model import get_embedding

logger = logging.getLogger(__name__)

_faiss_index: Optional[faiss.Index] = None
_trial_df: Optional[pd.DataFrame] = None


def _load_artifacts():
    global _faiss_index, _trial_df

    if _faiss_index is None:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    if _trial_df is None:
        with open(EMBEDDINGS_PKL_PATH, "rb") as f:
            _trial_df = pickle.load(f)

    return _faiss_index, _trial_df


def _parse_age_years(val):
    if val is None:
        return None
    try:
        return float(val)
    except:
        return None


class TrialRetriever:

    def retrieve_trials(
        self,
        patient_text: str,
        raw_conditions: str,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        k: int = RETRIEVAL_K,
    ) -> pd.DataFrame:

        index, df = _load_artifacts()

        # ─────────────────────────────────────────────
        # FIX: ensure raw_conditions_str is ALWAYS defined
        # ─────────────────────────────────────────────
        if isinstance(raw_conditions, list):
            raw_conditions_str = ", ".join(raw_conditions)
        elif isinstance(raw_conditions, str):
            raw_conditions_str = raw_conditions
        else:
            raw_conditions_str = ""

        # ─────────────────────────────────────────────
        # 1. BETTER EMBEDDING INPUT (KEY FIX)
        # ─────────────────────────────────────────────
        query_text = f"""
        Find clinical trials for this condition:

        Condition: {raw_conditions_str}

        Patient details:
        {patient_text}
        """

        emb = get_embedding(query_text)
        emb_vec = np.array(emb, dtype="float32").reshape(1, -1)

        search_k = min(k * SEARCH_MULTIPLIER, len(df))

        distances, indices = index.search(emb_vec, search_k)
        distances, indices = distances[0], indices[0]

        valid_mask = distances <= FAISS_SIMILARITY_THRESHOLD
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]

        if len(valid_indices) == 0:
            return df.iloc[[]]

        semantic_results = df.iloc[valid_indices].copy()
        semantic_results["_distance"] = valid_distances


        # ─────────────────────────────────────────────
        # FORCE CONDITION EXTRACTION (FINAL FIX)
        # ─────────────────────────────────────────────
        raw_lower = raw_conditions_str.lower()

        # simple keyword extraction
        keywords = []

        if "crohn" in raw_lower:
            keywords = ["crohn", "crohn disease", "inflammatory bowel disease", "ibd"]
        elif "diabetes" in raw_lower:
            keywords = ["diabetes", "type 2 diabetes", "t2dm"]
        elif "cancer" in raw_lower:
            keywords = ["cancer", "tumor", "oncology"]
        else:
            # fallback: use raw text
            keywords = [raw_lower]

        logger.info(f"Using keywords: {keywords}")

        if "cleaned_criteria" in semantic_results.columns and keywords:
            pattern = "|".join(re.escape(k) for k in keywords)

            condition_filtered = semantic_results[
                semantic_results["cleaned_criteria"].str.contains(
                    pattern, case=False, na=False, regex=True
                )
            ]

            if len(condition_filtered) > 0:
                logger.info(f"Filtered to {len(condition_filtered)} relevant trials")
                semantic_results = condition_filtered
            else:
                logger.warning("No keyword matches found — returning empty")
                return semantic_results.iloc[[]]

        # ─────────────────────────────────────────────
        # 2. STRUCTURED FILTERS (AGE / SEX)
        # ─────────────────────────────────────────────
        filtered = semantic_results.copy()

        if age is not None:
            age_f = float(age)

            if "min_age_years" in filtered.columns:
                min_ages = filtered["min_age_years"].apply(_parse_age_years)
                filtered = filtered[(min_ages.isna()) | (min_ages <= age_f)]

            if "max_age_years" in filtered.columns:
                max_ages = filtered["max_age_years"].apply(_parse_age_years)
                filtered = filtered[(max_ages.isna()) | (max_ages >= age_f)]

        if sex and "sex" in filtered.columns:
            sex_lower = sex.lower()
            filtered = filtered[
                filtered["sex"].fillna("all").str.lower().isin(["all", sex_lower])
            ]


        # ─────────────────────────────────────────────
        # 4. FALLBACK LOGIC
        # ─────────────────────────────────────────────
        if len(filtered) >= k:
            return filtered.head(k)

        return semantic_results.head(k)