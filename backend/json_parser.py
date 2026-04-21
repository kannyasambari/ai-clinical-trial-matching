"""
backend/json_parser.py
───────────────────────
Parse a single ClinicalTrials.gov JSON file into a flat dict.
Unchanged from original — already correct and well-structured.
"""

import json
from typing import Dict, Optional


def parse_trial_json(json_file_path: str) -> Optional[Dict]:
    """
    Parse a single clinical trial JSON file and extract relevant fields.
    Returns None if parsing fails.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        protocol = data.get("protocolSection", {})

        identification    = protocol.get("identificationModule",   {})
        nct_id            = identification.get("nctId")
        if not nct_id:
            return None

        eligibility       = protocol.get("eligibilityModule",      {})
        description       = protocol.get("descriptionModule",      {})
        design            = protocol.get("designModule",            {})
        status_mod        = protocol.get("statusModule",            {})
        conditions_module = protocol.get("conditionsModule",        {})
        interventions_mod = protocol.get("armsInterventionsModule", {})

        return {
            "nct_id":               nct_id,
            "title":                identification.get("briefTitle",    ""),
            "brief_summary":        description.get("briefSummary",     ""),
            "detailed_description": description.get("detailedDescription", ""),
            "study_type":           design.get("studyType",             ""),
            "phase":                (design.get("phases") or ["N/A"])[0],
            "enrollment":           design.get("enrollmentInfo", {}).get("count"),
            "start_date":           status_mod.get("startDateStruct",  {}).get("date"),
            "completion_date":      status_mod.get("completionDateStruct", {}).get("date"),
            "study_status":         status_mod.get("overallStatus",    ""),
            # ELIGIBILITY — core for RAG
            "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
            "sex":                  eligibility.get("sex",              "ALL"),
            "minimum_age":          eligibility.get("minimumAge",       "N/A"),
            "maximum_age":          eligibility.get("maximumAge",       "N/A"),
            "healthy_volunteers":   eligibility.get("healthyVolunteers", False),
            # Conditions & interventions
            "conditions":           conditions_module.get("conditions",    []),
            "interventions":        interventions_mod.get("interventions", []),
            "raw_json":             data,
        }

    except Exception as exc:
        print(f"Error parsing {json_file_path}: {exc}")
        return None