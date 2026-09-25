"""Central file locations for every pipeline step, so inputs and outputs are defined in one place."""

from __future__ import annotations

import os
from pathlib import Path

# Raw MIMIC-IV module folders (official PhysioNet releases: MIMIC-IV v3.1 and MIMIC-IV-Note v2.2).
# Edit these three values when deploying elsewhere, or override them per run with the
# ANAMNESIS_HOSP_DIR / ANAMNESIS_ICU_DIR / ANAMNESIS_NOTE_DIR environment variables or the
# --hosp-dir / --icu-dir / --note-dir flags of extract_stratified_sample.py.
MIMIC_IV_HOSP_DIR = Path(os.environ.get("ANAMNESIS_HOSP_DIR", "/home/tommaso/datasets/MIMICIV/3.1/hosp"))
MIMIC_IV_ICU_DIR = Path(os.environ.get("ANAMNESIS_ICU_DIR", "/home/tommaso/datasets/MIMICIV/3.1/icu"))
MIMIC_IV_NOTE_DIR = Path(
    os.environ.get("ANAMNESIS_NOTE_DIR", "/home/tommaso/datasets/MIMICIV/mimic-iv-note/2.2/note")
)

# Every generated artifact (JSON, report, PDFs, JSONL) lands here, outside the repository, because
# outputs derived from credentialed MIMIC data must never be committed. Override with ANAMNESIS_WORK_DIR.
WORK_DIR = Path(os.environ.get("ANAMNESIS_WORK_DIR", "/home/tommaso/anamnesis_output"))

SAMPLE_JSON = WORK_DIR / "mimic_stratified_sample.json"
SAMPLE_REPORT_MD = WORK_DIR / "mimic_stratified_sample_report.md"
PDF_DIR = WORK_DIR / "Admission_PDFs_stratified_enriched"
SINGLE_QA_PATH = WORK_DIR / "single_dataset_rulebased.jsonl"
MULTIDOC_QA_PATH = WORK_DIR / "multidoc_rulebased.jsonl"
MULTIDOC_SAME_PATIENT_PATH = WORK_DIR / "multidoc_dataset_rulebased_main_same_patient.jsonl"
MULTIDOC_CROSS_PATIENT_PATH = WORK_DIR / "multidoc_dataset_rulebased_cross_patient.jsonl"


def admission_pdf_name(subject_id: object, hadm_id: object) -> str:
    return f"Patient_{subject_id}_Admission_{hadm_id}.pdf"
