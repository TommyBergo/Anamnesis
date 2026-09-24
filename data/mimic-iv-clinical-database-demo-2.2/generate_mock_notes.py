import os
import uuid
import random
import pandas as pd

def generate_official_mimic_notes():
    hosp_dir = "./hosp"
    note_dir = "./note"
    
    admissions_path = os.path.join(hosp_dir, "admissions.csv.gz")
    
    if not os.path.exists(admissions_path):
        raise FileNotFoundError(f"Impossibile trovare admissions.csv.gz in {admissions_path}. Posiziona lo script nella cartella principale.")
    
    os.makedirs(note_dir, exist_ok=True)
    
    print(f"Caricamento dei ricoveri reali da: {admissions_path}")
    admissions_df = pd.read_csv(admissions_path, compression="gzip")
    
    discharge_rows = []
    discharge_detail_rows = []
    radiology_rows = []
    radiology_detail_rows = []
    
    for index, row in admissions_df.iterrows():
        subject_id = row["subject_id"]
        hadm_id = row["hadm_id"]
        dischtime = row["dischtime"]
        
        # 1. DISCHARGE TABLE
        note_id_ds = f"{subject_id}-{str(uuid.uuid4())[:8]}"
        padding_text = "The patient remained stable overnight without acute events. " * random.randint(3, 10)
        
        clinical_text = f"""
Chief Complaint:
Patient presents with acute symptoms requiring evaluation.

History of Present Illness:
{padding_text} Patient reports feeling progressively worse over the last 48 hours.

Past Medical History:
1. Hypertension
2. Type 2 Diabetes Mellitus

Discharge Medications:
1. Metformin 500mg daily
2. Lisinopril 10mg daily

Discharge Condition:
Stable and ready for discharge.

Discharge Disposition:
Home
"""
        discharge_rows.append({
            "note_id": note_id_ds,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "note_type": "DS",
            "note_seq": 1,
            "charttime": dischtime,
            "storetime": dischtime,
            "text": clinical_text.strip()
        })
        
        # 2. DISCHARGE_DETAIL TABLE (Official schema metadata)
        discharge_detail_rows.append({
            "note_id": note_id_ds,
            "field_name": "Attending Physician",
            "field_value": "Dr. Smith"
        })
        
        # 3. RADIOLOGY TABLE
        note_id_rad = f"{subject_id}-{str(uuid.uuid4())[:8]}"
        radiology_text = f"""
Indication: Evaluate for acute cardiopulmonary process.
Comparison: None available.
Findings: The lungs are clear bilaterally. No focal consolidation, effusion, or pneumothorax. Heart size is normal.
Impression: No acute cardiopulmonary abnormalities.
"""
        radiology_rows.append({
            "note_id": note_id_rad,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "note_type": "AR",
            "note_seq": 1,
            "charttime": dischtime,
            "storetime": dischtime,
            "text": radiology_text.strip()
        })
        
        # 4. RADIOLOGY_DETAIL TABLE (Official schema metadata)
        radiology_detail_rows.append({
            "note_id": note_id_rad,
            "field_name": "Modality",
            "field_value": "XR"
        })

    # Salvataggio di tutti i file compressi esattamente come richiesto da MIMIC-IV-Note
    pd.DataFrame(discharge_rows).to_csv(os.path.join(note_dir, "discharge.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(discharge_detail_rows).to_csv(os.path.join(note_dir, "discharge_detail.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(radiology_rows).to_csv(os.path.join(note_dir, "radiology.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(radiology_detail_rows).to_csv(os.path.join(note_dir, "radiology_detail.csv.gz"), index=False, compression="gzip")
    
    print("Tutte le tabelle del modulo note di MIMIC-IV sono state generate con successo in ./note/")

if __name__ == "__main__":
    generate_official_mimic_notes()