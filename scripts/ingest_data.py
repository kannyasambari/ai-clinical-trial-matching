"""
scripts/ingest_data.py
───────────────────────
Load raw ClinicalTrials.gov JSON files into PostgreSQL.

Reads config from .env via backend/config.py — no hardcoded secrets.
Run from the project root:
    python -m scripts.ingest_data
"""

import json
import os
import re
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from backend.config import BATCH_SIZE, DB_CONFIG, JSON_DATA_DIR
from backend.json_parser import parse_trial_json


def format_date(date_string) -> str | None:
    if not date_string:
        return None
    if isinstance(date_string, str):
        if re.match(r"^\d{4}-\d{2}$", date_string):
            return f"{date_string}-01"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_string):
            return date_string
    print(f"\n⚠️  Unexpected date format: '{date_string}' → NULL")
    return None


class TrialIngestion:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cur  = self.conn.cursor()
            print("✅ Connected to PostgreSQL.")
        except Exception as exc:
            print(f"❌ DB connection failed: {exc}")
            raise

        self.stats = {"processed": 0, "failed": 0, "skipped": 0, "total": 0}

    def insert_batch(self, trial_list: list[dict]) -> None:
        try:
            # trials
            trials_data = []
            for t in trial_list:
                if t["nct_id"]:
                    trials_data.append((
                        t["nct_id"], t["title"], t["brief_summary"],
                        t["detailed_description"], t["study_type"], t["phase"],
                        t["enrollment"],
                        format_date(t["start_date"]),
                        format_date(t["completion_date"]),
                        t["study_status"], json.dumps(t["raw_json"]),
                    ))
            if trials_data:
                execute_batch(self.cur, """
                    INSERT INTO trials
                      (nct_id, title, brief_summary, detailed_description,
                       study_type, phase, enrollment_count, start_date,
                       completion_date, study_status, raw_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (nct_id) DO NOTHING
                """, trials_data)

            # eligibility
            elig_data = [
                (t["nct_id"], str(t.get("eligibility_criteria", "")),
                 t.get("sex"), t.get("minimum_age"), t.get("maximum_age"),
                 str(t.get("healthy_volunteers", "")).lower() in ("true", "yes"))
                for t in trial_list
                if t.get("eligibility_criteria") is not None
            ]
            if elig_data:
                execute_batch(self.cur, """
                    INSERT INTO eligibility_criteria
                      (nct_id, full_criteria_text, sex, minimum_age, maximum_age, healthy_volunteers)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (nct_id) DO NOTHING
                """, elig_data)

            # conditions
            cond_data = [
                (t["nct_id"], cond)
                for t in trial_list
                for cond in t.get("conditions", [])
                if t["nct_id"] and cond
            ]
            if cond_data:
                execute_batch(self.cur, """
                    INSERT INTO trial_conditions (nct_id, condition_name)
                    VALUES (%s,%s)
                    ON CONFLICT (nct_id, condition_name) DO NOTHING
                """, cond_data)

            # interventions
            interv_data = [
                (t["nct_id"], iv.get("type"), iv.get("name"), iv.get("description"))
                for t in trial_list
                for iv in t.get("interventions", [])
                if t["nct_id"] and isinstance(iv, dict)
            ]
            if interv_data:
                execute_batch(self.cur, """
                    INSERT INTO trial_interventions
                      (nct_id, intervention_type, intervention_name, description)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (nct_id, intervention_name, intervention_type) DO NOTHING
                """, interv_data)

            self.conn.commit()

        except Exception as exc:
            self.conn.rollback()
            ids = [t.get("nct_id", "?") for t in trial_list[:5]]
            print(f"\n❌ Batch insert failed (IDs: {ids}…): {exc}")
            self.stats["skipped"] += len(trial_list)

    def process_all_files(self) -> None:
        json_dir = Path(JSON_DATA_DIR)
        if not json_dir.exists():
            print(f"❌ Directory not found: {json_dir}")
            return

        all_files = list(json_dir.rglob("*.json"))
        self.stats["total"] = len(all_files)
        print(f"📁 Found {len(all_files):,} JSON files")

        batch: list[dict] = []
        failed_batches    = 0

        with tqdm(total=len(all_files), unit="files") as pbar:
            for path in all_files:
                try:
                    trial = parse_trial_json(str(path))
                    if trial and trial.get("nct_id"):
                        batch.append(trial)
                        self.stats["processed"] += 1
                    else:
                        self.stats["failed"] += 1
                except Exception:
                    self.stats["failed"] += 1

                if len(batch) >= BATCH_SIZE:
                    self.insert_batch(batch)
                    batch = []

                pbar.update(1)

        if batch:
            self.insert_batch(batch)

        self._print_summary(failed_batches)

    def _print_summary(self, failed_batches: int = 0) -> None:
        s = self.stats
        print(f"\n{'='*60}")
        print(f"✅ Parsed:  {s['processed']:,} / {s['total']:,}")
        print(f"❌ Failed:  {s['failed']:,}")
        print(f"⚠️  Skipped: {s['skipped']:,} (DB errors, {failed_batches} batches)")
        print(f"{'='*60}")

    def close(self) -> None:
        if self.cur:  self.cur.close()
        if self.conn: self.conn.close()


if __name__ == "__main__":
    ing = None
    try:
        ing = TrialIngestion()
        ing.process_all_files()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    finally:
        if ing:
            ing.close()