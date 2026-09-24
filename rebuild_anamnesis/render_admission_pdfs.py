"""Renders each admission in mimic_stratified_sample.json to one PDF: an enriched structured header followed by every clinical note of the stay."""

import hashlib

_old_md5 = hashlib.md5
def _new_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return _old_md5(*args, **kwargs)
hashlib.md5 = _new_md5

import html
import json
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.graphics.shapes import Drawing, Line
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from shared.mimic_iv import NOTE_CATEGORY_ORDER, resolve_disposition
from shared.mimic_section_parser import parse_admission_sections
from shared.pipeline_paths import PDF_DIR, SAMPLE_JSON, admission_pdf_name

# Detail-table fields worth showing in a note heading (MIMIC-IV-Note radiology_detail uses exam_name).
NOTE_HEADING_DETAIL_FIELDS = ("exam_name", "Modality", "author")

NBSP_GAP = "&nbsp;&nbsp;&nbsp;"


def esc(value: object) -> str:
    return html.escape(str(value))


def note_inventory(notes: list[dict]) -> str:
    counts = Counter(note["category"] for note in notes)
    return ", ".join(f"{category} ({counts[category]})" for category in NOTE_CATEGORY_ORDER if counts[category])


def note_heading(note: dict, index: int, total: int) -> str:
    parts = [f"<b>Clinical note {index} of {total}: {esc(note['category'])}</b>"]
    if note.get("note_type_label") and note["note_type_label"] != note["category"]:
        parts.append(esc(note["note_type_label"]))
    if note.get("charttime"):
        parts.append(f"charted {esc(note['charttime'])}")
    details = note.get("details") or {}
    for field in NOTE_HEADING_DETAIL_FIELDS:
        if details.get(field):
            parts.append(f"{esc(field.replace('_', ' ').capitalize())}: {esc(details[field])}")
    return " - ".join(parts)


def build_header(story: list, styles, info: dict, adm: dict, disposition: str) -> None:
    age = adm.get("age_at_admission")
    los_days = adm.get("length_of_stay_days")
    lines = [
        f"<b>Patient ID:</b> {esc(info['subject_id'])} {NBSP_GAP} <b>Sex:</b> {esc(info['gender'])} "
        f"{NBSP_GAP} <b>Age at Admission:</b> {age if age is not None else 'N/A'}",
        f"<b>Admission ID:</b> {esc(adm['hadm_id'])} {NBSP_GAP} <b>Type:</b> {esc(adm['admission_type'])} "
        f"{NBSP_GAP} <b>Hospital Service:</b> {esc(adm.get('service') or 'Not recorded')}",
        f"<b>Admission Date:</b> {esc(adm['admittime'].split(' ')[0])} {NBSP_GAP} <b>Discharge Date:</b> "
        f"{esc(adm['dischtime'].split(' ')[0])} {NBSP_GAP} <b>Length of Stay:</b> "
        f"{los_days if los_days is not None else 'N/A'} days",
        f"<b>Principal Diagnosis:</b> {esc(adm.get('diagnosis') or 'Not recorded')}",
        f"<b>Discharge Disposition:</b> {esc(disposition)}",
        f"<b>ICU Stay:</b> {'Yes' if adm.get('had_icu_stay') else 'No'} {NBSP_GAP} "
        f"<b>In-Hospital Mortality:</b> {'Yes' if adm.get('hospital_expire_flag') else 'No'} {NBSP_GAP} "
        f"<b>Number of Diagnoses:</b> {adm.get('num_diagnoses', 0)}",
        f"<b>Clinical Notes Included:</b> {esc(note_inventory(adm['notes']))}",
    ]
    for line in lines:
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 5))


def main():
    print(f"Loading {SAMPLE_JSON}...")
    with SAMPLE_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='MedicalText', parent=styles['Normal'], fontName='Courier',
        fontSize=9, leading=11, alignment=TA_LEFT, spaceAfter=10
    ))

    pdf_count = 0

    for patient in data:
        info = patient["patient_info"]

        for adm in patient.get("admissions", []):
            notes = adm["notes"]
            disposition = resolve_disposition(adm, parse_admission_sections(notes))

            filename = PDF_DIR / admission_pdf_name(info["subject_id"], adm["hadm_id"])
            doc = SimpleDocTemplate(str(filename), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

            story = []
            story.append(Paragraph("Clinical Record Summary", styles['Title']))
            story.append(Spacer(1, 15))
            build_header(story, styles, info, adm, disposition)

            story.append(Spacer(1, 5))
            rule = Drawing(515, 1)
            rule.add(Line(0, 0, 515, 0, strokeWidth=1, strokeColorName='black'))
            story.append(rule)
            story.append(Spacer(1, 15))

            for index, note in enumerate(notes, 1):
                story.append(Paragraph(note_heading(note, index, len(notes)), styles['Normal']))
                story.append(Spacer(1, 5))
                safe_text = esc(note["text"].strip()).replace('\n', '<br/>')
                story.append(Paragraph(safe_text, styles['MedicalText']))
                story.append(Spacer(1, 15))

            doc.build(story)
            pdf_count += 1
            if pdf_count % 20 == 0:
                print(f"  ... {pdf_count} PDFs generated")

    print(f"Created {pdf_count} PDFs in '{PDF_DIR}'.")


if __name__ == "__main__":
    main()
