"""MIMIC-IV schema knowledge shared by the pipeline: table locations, clinical-note sources, and code-to-label mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NoteSource:
    """One MIMIC-IV-Note-schema table (note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text) and the note category it contributes."""

    table: str
    category: str
    detail_table: str


# Discharge and radiology are the two tables MIMIC-IV-Note actually ships. Nursing, physician, and
# consult tables are read whenever they are present with the same schema (e.g. a synthetic or
# locally curated extension); a missing optional table is skipped, never an error.
NOTE_SOURCES: tuple[NoteSource, ...] = (
    NoteSource("discharge", "Discharge summary", "discharge_detail"),
    NoteSource("radiology", "Radiology", "radiology_detail"),
    NoteSource("nursing", "Nursing", "nursing_detail"),
    NoteSource("physician", "Physician", "physician_detail"),
    NoteSource("consult", "Consult", "consult_detail"),
)

REQUIRED_NOTE_TABLES = frozenset({"discharge"})

NOTE_CATEGORY_ORDER: tuple[str, ...] = tuple(source.category for source in NOTE_SOURCES)

NOTE_TYPE_LABELS = {
    "DS": "Discharge summary",
    "AD": "Discharge summary addendum",
    "RR": "Radiology report",
    "AR": "Radiology report addendum",
}

NOTE_COLUMNS = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"]

ADMISSION_TYPE_LABELS = {
    "EW EMER.": "Emergency",
    "DIRECT EMER.": "Direct emergency",
    "URGENT": "Urgent",
    "ELECTIVE": "Elective",
    "SURGICAL SAME DAY ADMISSION": "Surgical same-day admission",
    "OBSERVATION ADMIT": "Observation",
    "EU OBSERVATION": "Emergency unit observation",
    "DIRECT OBSERVATION": "Direct observation",
    "AMBULATORY OBSERVATION": "Ambulatory observation",
}

SERVICE_LABELS = {
    "CMED": "Cardiac Medicine",
    "CSURG": "Cardiac Surgery",
    "DENT": "Dental",
    "ENT": "Ear Nose and Throat",
    "EYE": "Ophthalmology",
    "GU": "Urology",
    "GYN": "Gynecology",
    "MED": "General Medicine",
    "NB": "Newborn",
    "NBB": "Newborn Baby",
    "NMED": "Neurology",
    "NSURG": "Neurosurgery",
    "OBS": "Obstetrics",
    "OMED": "Oncology",
    "ORTHO": "Orthopedics",
    "PSURG": "Plastic Surgery",
    "PSYCH": "Psychiatry",
    "SURG": "General Surgery",
    "TRAUM": "Trauma",
    "TSURG": "Thoracic Surgery",
    "VSURG": "Vascular Surgery",
}

DISCHARGE_LOCATION_LABELS = {
    "AGAINST ADVICE": "Left against medical advice",
    "CHRONIC/LONG TERM ACUTE CARE": "Chronic or long-term acute care facility",
    "HOME HEALTH CARE": "Home with home health care",
}

# Discharge locations that do not describe where a living patient went after leaving the hospital.
NON_DESTINATION_DISCHARGE_LOCATIONS = frozenset({"DIED"})

DIED_IN_HOSPITAL_DISPOSITION = "Died in hospital"
UNRECORDED_DISPOSITION = "Not recorded"

UNKNOWN_RACE_VALUES = frozenset({"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"})


def find_table(directories: list[Path], table: str) -> Optional[Path]:
    for directory in directories:
        for suffix in (".csv.gz", ".csv"):
            candidate = directory / f"{table}{suffix}"
            if candidate.exists():
                return candidate
    return None


def require_table(directories: list[Path], table: str) -> Path:
    path = find_table(directories, table)
    if path is None:
        searched = ", ".join(str(d) for d in directories)
        raise FileNotFoundError(f"Could not find {table} (.csv or .csv.gz) in: {searched}")
    return path


def module_dirs(data_dir: Path, module: str) -> list[Path]:
    """Accepts both the official layout (data_dir/hosp/...) and a flat folder of tables."""
    return [data_dir / module, data_dir]


def label_admission_type(code: object) -> str:
    text = _clean(code)
    return ADMISSION_TYPE_LABELS.get(text.upper(), text.title()) if text else ""


def label_service(code: object) -> str:
    text = _clean(code)
    return SERVICE_LABELS.get(text.upper(), text) if text else ""


def label_discharge_location(code: object) -> Optional[str]:
    text = _clean(code)
    if not text or text.upper() in NON_DESTINATION_DISCHARGE_LOCATIONS:
        return None
    return DISCHARGE_LOCATION_LABELS.get(text.upper(), text.capitalize())


def resolve_disposition(admission: dict, sections_by_category: dict[str, dict[str, str]]) -> str:
    """The header's Discharge Disposition: the structured discharge_location, else the discharge-summary section."""
    if admission.get("discharge_location"):
        return admission["discharge_location"]
    if admission.get("hospital_expire_flag"):
        return DIED_IN_HOSPITAL_DISPOSITION
    parsed = sections_by_category.get("Discharge summary", {}).get("Discharge Disposition", "").strip()
    return parsed or UNRECORDED_DISPOSITION


def label_note_type(code: object) -> str:
    text = _clean(code)
    return NOTE_TYPE_LABELS.get(text.upper(), text)


def race_group(race: object) -> str:
    text = _clean(race).upper()
    if not text or text in UNKNOWN_RACE_VALUES:
        return "Unknown"
    if text.startswith("WHITE"):
        return "White"
    if text.startswith("BLACK"):
        return "Black"
    if text.startswith("HISPANIC") or text.startswith("SOUTH AMERICAN"):
        return "Hispanic or Latino"
    if text.startswith("ASIAN"):
        return "Asian"
    return "Other"


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return "" if text.casefold() in {"nan", "none", "<na>"} else text
