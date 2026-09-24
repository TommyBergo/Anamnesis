"""Renders each admission in mimic_stratified_sample.json to an enriched multi-note PDF."""

import hashlib

_old_md5 = hashlib.md5
def _new_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return _old_md5(*args, **kwargs)
hashlib.md5 = _new_md5

import json
import re
import sys
import html
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.graphics.shapes import Drawing, Line
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from shared.mimic_section_parser import parse_sections
from shared.mimic_common import compute_age, compute_los_days

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "mimic_stratified_sample.json"
OUTPUT_DIR = BASE_DIR / "Ammissioni_PDF_stratified_enriched"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print(f"Loading {JSON_FILE.name}...")
    with JSON_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='MedicalText', parent=styles['Normal'], fontName='Courier',
        fontSize=9, leading=11, alignment=TA_LEFT, spaceAfter=10
    ))

    pdf_count = 0

    for patient in data:
        info = patient["patient_info"]
        subject_id = info["subject_id"]
        gender = info["gender"]
        dob = info["dob"].split(" ")[0]

        for adm in patient.get("admissions", []):
            hadm_id = adm["hadm_id"]
            adm_type = adm["admission_type"]
            diagnosis = adm.get("diagnosis", "N/D")
            adm_time_full = adm["admittime"]
            disch_time_full = adm["dischtime"]
            adm_time = adm_time_full.split(" ")[0]
            disch_time = disch_time_full.split(" ")[0]
            
            # MODIFICATO: Legge la nuova chiave multi-nota
            raw_text = adm.get("clinical_notes", adm.get("discharge_summary", ""))

            service = adm.get("service") or "Not recorded"
            icu_stay = "Yes" if adm.get("had_icu_stay") else "No"
            expired = "Yes" if adm.get("hospital_expire_flag") else "No"
            num_diagnoses = adm.get("num_diagnoses", 0)

            sections = parse_sections(raw_text)
            disposition = sections.get("Discharge Disposition", "").strip() or "Not stated in note"

            age = compute_age(dob, adm_time_full)
            los_days = compute_los_days(adm_time_full, disch_time_full)

            filename = OUTPUT_DIR / f"Patient_{subject_id}_Admission_{hadm_id}.pdf"
            doc = SimpleDocTemplate(str(filename), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

            Story = []

            Story.append(Paragraph("Clinical Record Summary", styles['Title']))
            Story.append(Spacer(1, 15))

            Story.append(Paragraph(
                f"<b>Patient ID:</b> {subject_id} &nbsp;&nbsp;&nbsp; <b>Sex:</b> {gender} "
                f"&nbsp;&nbsp;&nbsp; <b>Date of Birth:</b> {dob} &nbsp;&nbsp;&nbsp; "
                f"<b>Age at Admission:</b> {age if age is not None else 'N/A'}", styles['Normal']
            ))
            Story.append(Spacer(1, 5))
            Story.append(Paragraph(
                f"<b>Admission ID:</b> {hadm_id} &nbsp;&nbsp;&nbsp; <b>Type:</b> {adm_type} "
                f"&nbsp;&nbsp;&nbsp; <b>Hospital Service:</b> {html.escape(str(service))}", styles['Normal']
            ))
            Story.append(Spacer(1, 5))
            Story.append(Paragraph(
                f"<b>Admission Date:</b> {adm_time} &nbsp;&nbsp;&nbsp; <b>Discharge Date:</b> "
                f"{disch_time} &nbsp;&nbsp;&nbsp; <b>Length of Stay:</b> "
                f"{los_days if los_days is not None else 'N/A'} days", styles['Normal']
            ))
            Story.append(Spacer(1, 5))
            Story.append(Paragraph(f"<b>Principal Diagnosis:</b> {html.escape(str(diagnosis))}", styles['Normal']))
            Story.append(Spacer(1, 5))
            Story.append(Paragraph(f"<b>Discharge Disposition:</b> {html.escape(disposition)}", styles['Normal']))
            Story.append(Spacer(1, 5))
            Story.append(Paragraph(
                f"<b>ICU Stay:</b> {icu_stay} &nbsp;&nbsp;&nbsp; "
                f"<b>In-Hospital Mortality:</b> {expired} &nbsp;&nbsp;&nbsp; "
                f"<b>Number of Diagnoses:</b> {num_diagnoses}", styles['Normal']
            ))

            Story.append(Spacer(1, 10))
            d = Drawing(515, 1)
            d.add(Line(0, 0, 515, 0, strokeWidth=1, strokeColorName='black'))
            Story.append(d)
            Story.append(Spacer(1, 15))

            # MODIFICATO: Usa il separatore corretto delle note multiple
            summaries = raw_text.split("--- NEXT NOTE ---")
            for sub_idx, summary_text in enumerate(summaries, 1):
                if len(summaries) > 1:
                    Story.append(Paragraph(f"<i>Clinical note {sub_idx} of {len(summaries)}</i>", styles['Normal']))
                    Story.append(Spacer(1, 5))
                safe_text = html.escape(summary_text.strip()).replace('\n', '<br/>')
                Story.append(Paragraph(safe_text, styles['MedicalText']))
                Story.append(Spacer(1, 15))

            doc.build(Story)
            pdf_count += 1
            if pdf_count % 20 == 0:
                print(f"  ... {pdf_count} PDFs generated")

    print(f"Created {pdf_count} PDFs in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()