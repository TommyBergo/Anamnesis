"""Splits MIMIC clinical notes (discharge, radiology, nursing, physician, consult) into their named sections."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SectionSchema:
    """The section headers of one note category, mapped to canonical section names."""

    aliases: dict[str, list[str]]
    boundary_only_headers: list[str] = field(default_factory=list)
    ignore_case: bool = False

    def __post_init__(self) -> None:
        flags = re.MULTILINE | (re.IGNORECASE if self.ignore_case else 0)
        all_alternatives = [alt for alts in self.aliases.values() for alt in alts] + self.boundary_only_headers
        self.header_pattern = re.compile(
            r"^[ \t]*(?P<name>" + "|".join(all_alternatives) + r")[ \t]*:[ \t]*",
            flags,
        )
        self._match_flags = re.IGNORECASE if self.ignore_case else 0

    def canonical_name(self, matched_header_text: str) -> str | None:
        for boundary in self.boundary_only_headers:
            if re.fullmatch(boundary, matched_header_text, self._match_flags):
                return None
        for canon, alts in self.aliases.items():
            for alt in alts:
                if re.fullmatch(alt, matched_header_text, self._match_flags):
                    return canon
        return matched_header_text


DISCHARGE_SCHEMA = SectionSchema(
    aliases={
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
    },
    boundary_only_headers=["Attending", "Service", "Facility"],
)

RADIOLOGY_SCHEMA = SectionSchema(
    aliases={
        "Examination": ["Examination", "Exam"],
        "Indication": ["Indication", "Clinical Indication", "History"],
        "Technique": ["Technique"],
        "Comparison": ["Comparison"],
        "Findings": ["Findings"],
        "Impression": ["Impression", "Conclusion"],
    },
    ignore_case=True,
)

NURSING_SCHEMA = SectionSchema(
    aliases={
        "Situation": ["Situation"],
        "Background": ["Background"],
        "Assessment": ["Assessment"],
        "Action": ["Action"],
        "Response": ["Response"],
        "Plan": ["Plan"],
    },
    ignore_case=True,
)

PHYSICIAN_SCHEMA = SectionSchema(
    aliases={
        "Chief Complaint": ["Chief Complaint"],
        "History of Present Illness": ["History of Present Illness", "HPI"],
        "24 Hour Events": ["24 Hour Events"],
        "Physical Examination": ["Physical Examination", "Physical Exam"],
        "Labs and Radiology": ["Labs / Radiology", "Labs and Radiology"],
        "Assessment and Plan": ["Assessment and Plan", "Assessment & Plan", "A/P"],
    },
    ignore_case=True,
)

CONSULT_SCHEMA = SectionSchema(
    aliases={
        "Reason for Consultation": ["Reason for Consult(?:ation)?"],
        "History of Present Illness": ["History of Present Illness", "HPI"],
        "Impression": ["Impression", "Assessment"],
        "Recommendations": ["Recommendations?"],
    },
    ignore_case=True,
)

SECTION_SCHEMAS: dict[str, SectionSchema] = {
    "Discharge summary": DISCHARGE_SCHEMA,
    "Radiology": RADIOLOGY_SCHEMA,
    "Nursing": NURSING_SCHEMA,
    "Physician": PHYSICIAN_SCHEMA,
    "Consult": CONSULT_SCHEMA,
}


def parse_sections(note_text: str, category: str = "Discharge summary") -> dict[str, str]:
    schema = SECTION_SCHEMAS.get(category, DISCHARGE_SCHEMA)
    matches = list(schema.header_pattern.finditer(note_text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        canon = schema.canonical_name(m.group("name"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note_text)
        content = note_text[start:end].strip()
        if canon and content and canon not in sections:
            sections[canon] = content
    return sections


def parse_admission_sections(notes: list[dict]) -> dict[str, dict[str, str]]:
    """Parses every note of an admission; within a category, the first note carrying a section wins."""
    by_category: dict[str, dict[str, str]] = {}
    for note in notes:
        category = note.get("category", "Discharge summary")
        parsed = parse_sections(note.get("text", ""), category)
        merged = by_category.setdefault(category, {})
        for name, content in parsed.items():
            merged.setdefault(name, content)
    return by_category
