"""Central file locations for every pipeline step, so inputs and outputs are defined in one place."""

from __future__ import annotations

import os
from pathlib import Path

REBUILD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REBUILD_DIR.parent

# Raw MIMIC-IV tables (hosp/, icu/, note/ modules). Override with ANAMNESIS_DATA_DIR or --data-dir.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "ANAMNESIS_DATA_DIR",
        PROJECT_ROOT / "data" / "mimic-iv-clinical-database-demo-2.2",
    )
)

# Every generated artifact lands here. Override with ANAMNESIS_WORK_DIR to keep the source tree clean.
WORK_DIR = Path(os.environ.get("ANAMNESIS_WORK_DIR", REBUILD_DIR))

SAMPLE_JSON = WORK_DIR / "mimic_stratified_sample.json"
SAMPLE_REPORT_MD = WORK_DIR / "mimic_stratified_sample_report.md"
PDF_DIR = WORK_DIR / "Admission_PDFs_stratified_enriched"
SINGLE_QA_PATH = WORK_DIR / "single_dataset_rulebased.jsonl"
MULTIDOC_QA_PATH = WORK_DIR / "multidoc_rulebased.jsonl"
MULTIDOC_SAME_PATIENT_PATH = WORK_DIR / "multidoc_dataset_rulebased_main_same_patient.jsonl"
MULTIDOC_CROSS_PATIENT_PATH = WORK_DIR / "multidoc_dataset_rulebased_cross_patient.jsonl"


def admission_pdf_name(subject_id: object, hadm_id: object) -> str:
    return f"Patient_{subject_id}_Admission_{hadm_id}.pdf"
