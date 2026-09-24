"""Splits a MIMIC-style discharge summary note into its named sections."""

from __future__ import annotations

import re

SECTION_ALIASES: dict[str, list[str]] = {
    "Chief Complaint": ["Chief Complaint"],
    "Major Surgical or Invasive Procedure": ["Major Surgical or Invasive Procedure(?:s)?"],
    "Allergies": ["Allergies"],
    "History of Present Illness": ["History of Present Illness"],
    "Past Medical History": ["Past Medical History", "PAST MEDICAL HISTORY", "PMH"],
    "Social History": ["Social History"],
    "Family History": ["Family History"],
    "Physical Exam": ["Physical Exam"],
    "Pertinent Results": ["Pertinent Results"],
    "Brief Hospital Course": ["Brief Hospital Course", "HOSPITAL COURSE"],
    "Medications on Admission": ["Medications on Admission", "MEDICATIONS ON ADMISSION"],
    "Discharge Medications": ["Discharge Medications", "DISCHARGE MEDICATIONS"],
    "Discharge Disposition": ["Discharge Disposition", "DISCHARGE DISPOSITION"],
    "Discharge Diagnosis": ["Discharge Diagnosis", "Discharge Diagnoses", "DISCHARGE DIAGNOSIS", "DISCHARGE DIAGNOSES"],
    "Discharge Condition": ["Discharge Condition"],
    "Discharge Instructions": ["Discharge Instructions"],
    "Followup Instructions": ["Followup Instructions"],
}

_BOUNDARY_ONLY_HEADERS = ["Attending", "Service", "Facility"]

_ALL_HEADER_ALTS = [alt for alts in SECTION_ALIASES.values() for alt in alts] + _BOUNDARY_ONLY_HEADERS

_HEADER_PATTERN = re.compile(
    r"^(?P<name>" + "|".join(_ALL_HEADER_ALTS) + r"):[ \t]*",
    re.MULTILINE,
)

def _canonical_name(matched_header_text: str) -> str | None:
    if matched_header_text in _BOUNDARY_ONLY_HEADERS:
        return None
    for canon, alts in SECTION_ALIASES.items():
        for alt in alts:
            if re.fullmatch(alt, matched_header_text):
                return canon
    return matched_header_text

def parse_sections(discharge_summary_text: str) -> dict[str, str]:
    first_note = re.split(r"-{0,3}\s*NEXT SUMMARY\s*-{0,3}", discharge_summary_text)[0]

    matches = list(_HEADER_PATTERN.finditer(first_note))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        canon = _canonical_name(m.group("name"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(first_note)
        content = first_note[start:end].strip()
        if canon and content and canon not in sections:
            sections[canon] = content
    return sections
