"""Selects a stratified multi-category patient sample from raw MIMIC-III CSVs and writes it to mimic_stratified_sample.json."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import pandas as pd

from shared.mimic_common import compute_age

BASE_DIR = "/gringotts/datasets/MIMIC-III"
OUTPUT_JSON = "mimic_stratified_sample.json"
REPORT_MD = "mimic_stratified_sample_report.md"

INNER_STRATUM_VARIABLES = ["gender", "age_quintile", "note_length_quintile"]

# Categorie di note cliniche incluse (non più solo Discharge summary)
TARGET_CATEGORIES = {
    "Discharge summary",
    "Radiology",
    "Nursing",
    "Physician",
    "Consult",
}

def get_file_path(base_dir: str, filename: str) -> str:
    gz_path = os.path.join(base_dir, f"{filename}.csv.gz")
    csv_path = os.path.join(base_dir, f"{filename}.csv")
    if os.path.exists(gz_path):
        return gz_path
    if os.path.exists(csv_path):
        return csv_path
    raise FileNotFoundError(f"Could not find {filename} (.csv or .csv.gz) in {base_dir}")

def quintile_bucket(value: float, sorted_values: list[float]) -> str:
    n = len(sorted_values)
    for band, label in [(1, "Q1_0-20"), (2, "Q2_20-40"), (3, "Q3_40-60"), (4, "Q4_60-80")]:
        cut = sorted_values[min(n - 1, (band * n) // 5)]
        if value <= cut:
            return label
    return "Q5_80-100"

def phase1_note_length_index(noteevents_path: str) -> pd.DataFrame:
    print(f"Phase 1: streaming {os.path.basename(noteevents_path)} for multi-category note lengths...")
    lengths: dict[tuple[int, int], int] = defaultdict(int)
    rows_seen = 0
    for chunk in pd.read_csv(
        noteevents_path, usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "CATEGORY": str, "TEXT": str},
        chunksize=100_000, low_memory=False,
    ):
        rows_seen += len(chunk)
        valid_notes = chunk[chunk["CATEGORY"].isin(TARGET_CATEGORIES) & chunk["HADM_ID"].notna()]
        for subj, hadm, text in zip(valid_notes["SUBJECT_ID"], valid_notes["HADM_ID"], valid_notes["TEXT"]):
            lengths[(int(subj), int(hadm))] += len(str(text).strip())
        if rows_seen % 1_000_000 < 100_000:
            print(f"  ... {rows_seen:,} rows scanned, {len(lengths):,} admissions with target notes so far")
    print(f"Phase 1 done: {len(lengths):,} admissions have target clinical notes")
    return pd.DataFrame(
        [(subj, hadm, length) for (subj, hadm), length in lengths.items()],
        columns=["SUBJECT_ID", "HADM_ID", "note_length"],
    )

def load_diagnoses(diagnoses_path: str) -> pd.DataFrame:
    print(f"Loading {os.path.basename(diagnoses_path)}...")
    diagnoses = pd.read_csv(diagnoses_path, usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    diagnoses = diagnoses.dropna(subset=["HADM_ID", "ICD9_CODE"])
    agg = diagnoses.groupby(["SUBJECT_ID", "HADM_ID"]).agg(
        num_diagnoses=("ICD9_CODE", "count"),
    ).reset_index()
    print(f"  {len(agg):,} admissions have >=1 diagnosis code")
    return agg

def load_icustays(icustays_path: str) -> pd.DataFrame:
    print(f"Loading {os.path.basename(icustays_path)}...")
    icu = pd.read_csv(icustays_path, usecols=["SUBJECT_ID", "HADM_ID"]).dropna(subset=["HADM_ID"])
    icu = icu.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])
    icu["had_icu_stay"] = True
    return icu[["SUBJECT_ID", "HADM_ID", "had_icu_stay"]]

def load_services(services_path: str) -> pd.DataFrame:
    print(f"Loading {os.path.basename(services_path)}...")
    services = pd.read_csv(services_path, usecols=["SUBJECT_ID", "HADM_ID", "TRANSFERTIME", "CURR_SERVICE"])
    services = services.dropna(subset=["HADM_ID"]).sort_values("TRANSFERTIME")
    first_service = services.groupby(["SUBJECT_ID", "HADM_ID"], as_index=False).first()
    return first_service[["SUBJECT_ID", "HADM_ID", "CURR_SERVICE"]].rename(columns={"CURR_SERVICE": "service"})

def select_patients_stratified(representative: pd.DataFrame, n: int, seed: int) -> tuple[list[int], dict]:
    population_n = len(representative)
    group_counts = Counter(tuple(row[v] for v in INNER_STRATUM_VARIABLES) for _, row in representative.iterrows())
    
    rng = random.Random(seed)
    order = list(representative["SUBJECT_ID"])
    rng.shuffle(order)
    by_subject = {row["SUBJECT_ID"]: row for _, row in representative.iterrows()}

    selected: list[int] = []
    current_counts: Counter = Counter()
    
    for relaxed in (False, True):
        for subj in order:
            if len(selected) >= n:
                break
            if subj in selected:
                continue
            row = by_subject[subj]
            key = tuple(row[v] for v in INNER_STRATUM_VARIABLES)
            target_share = group_counts[key] / population_n
            current_share = current_counts.get(key, 0) / max(1, len(selected) + 1)
            if relaxed or current_share <= target_share:
                selected.append(subj)
                current_counts[key] += 1
        if len(selected) >= n:
            break

    balance_info = {
        "population_n": population_n,
        "representative": representative,
    }
    return selected[:n], balance_info

def phase2_select_patients(
    note_index: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    icustays: pd.DataFrame,
    services: pd.DataFrame,
    n: int,
    seed: int,
) -> tuple[list[int], dict]:
    print("Phase 2: joining tables and selecting extended patient sample...")
    patients_named = patients.rename(columns={"GENDER": "gender"})

    merged = note_index.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    merged = merged.merge(patients_named, on="SUBJECT_ID", how="inner")
    merged = merged.merge(diagnoses, on=["SUBJECT_ID", "HADM_ID"], how="left")
    merged = merged.merge(icustays, on=["SUBJECT_ID", "HADM_ID"], how="left")
    merged = merged.merge(services, on=["SUBJECT_ID", "HADM_ID"], how="left")
    
    merged["num_diagnoses"] = merged["num_diagnoses"].fillna(0).astype(int)
    merged["had_icu_stay"] = merged["had_icu_stay"].fillna(False)
    merged["service"] = merged["service"].fillna("UNKNOWN")
    merged = merged.dropna(subset=["DOB", "ADMITTIME"])
    print(f"  {len(merged):,} admissions fully joined")

    merged["age"] = merged.apply(lambda row: compute_age(row["DOB"], row["ADMITTIME"]), axis=1)

    representative = merged.sort_values("ADMITTIME").groupby("SUBJECT_ID", as_index=False).first()
    print(f"  {len(representative):,} unique candidate patients")

    ages_sorted = sorted(representative["age"].tolist())
    note_len_sorted = sorted(representative["note_length"].tolist())
    representative = representative.copy()
    representative["age_quintile"] = representative["age"].apply(lambda a: quintile_bucket(a, ages_sorted))
    representative["note_length_quintile"] = representative["note_length"].apply(lambda x: quintile_bucket(x, note_len_sorted))

    selected_subject_ids, balance_info = select_patients_stratified(representative, n=n, seed=seed)
    print(f"Phase 2 done: selected {len(selected_subject_ids)} patients")
    return selected_subject_ids, balance_info

def phase3_pull_full_records(
    selected_subject_ids: list[int],
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    icustays: pd.DataFrame,
    services: pd.DataFrame,
    noteevents_path: str,
) -> list[dict]:
    print("Phase 3: streaming NOTEEVENTS to pull full multi-category text for selected patients...")
    selected_set = set(selected_subject_ids)

    patients = patients[patients["SUBJECT_ID"].isin(selected_set)]
    admissions = admissions[admissions["SUBJECT_ID"].isin(selected_set)]

    extras = admissions[["SUBJECT_ID", "HADM_ID"]].merge(diagnoses, on=["SUBJECT_ID", "HADM_ID"], how="left")
    extras = extras.merge(icustays, on=["SUBJECT_ID", "HADM_ID"], how="left")
    extras = extras.merge(services, on=["SUBJECT_ID", "HADM_ID"], how="left")
    extras["num_diagnoses"] = extras["num_diagnoses"].fillna(0).astype(int)
    extras["had_icu_stay"] = extras["had_icu_stay"].fillna(False)
    extras["service"] = extras["service"].fillna("UNKNOWN")
    extras_by_adm = {(r["SUBJECT_ID"], r["HADM_ID"]): r for _, r in extras.iterrows()}

    notes_by_admission: dict[tuple[int, int], list[str]] = defaultdict(list)
    rows_seen = 0
    for chunk in pd.read_csv(
        noteevents_path, usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "CATEGORY": str, "TEXT": str},
        chunksize=100_000, low_memory=False,
    ):
        rows_seen += len(chunk)
        f = chunk[
            chunk["CATEGORY"].isin(TARGET_CATEGORIES)
            & chunk["HADM_ID"].notna()
            & chunk["SUBJECT_ID"].isin(selected_set)
        ]
        for subj, hadm, cat, text in zip(f["SUBJECT_ID"], f["HADM_ID"], f["CATEGORY"], f["TEXT"]):
            formatted_note = f"[Category: {cat}]\n{str(text).strip()}"
            notes_by_admission[(int(subj), int(hadm))].append(formatted_note)
            
    print(f"  scanned {rows_seen:,} rows, found clinical notes for {len(notes_by_admission):,} admissions")

    final_data = []
    for _, p_row in patients.iterrows():
        subj_id = int(p_row["SUBJECT_ID"])
        patient_dict = {
            "patient_info": {
                "subject_id": subj_id,
                "gender": str(p_row.get("GENDER", "")),
                "dob": str(p_row.get("DOB", "")),
            },
            "admissions": [],
        }
        p_admissions = admissions[admissions["SUBJECT_ID"] == subj_id]
        for _, a_row in p_admissions.iterrows():
            if pd.isna(a_row["HADM_ID"]):
                continue
            hadm_id = int(a_row["HADM_ID"])
            texts = notes_by_admission.get((subj_id, hadm_id))
            if not texts:
                continue
            extra = extras_by_adm.get((subj_id, hadm_id))
            patient_dict["admissions"].append({
                "hadm_id": hadm_id,
                "admission_type": str(a_row.get("ADMISSION_TYPE", "")),
                "diagnosis": str(a_row.get("DIAGNOSIS", "")),
                "admittime": str(a_row.get("ADMITTIME", "")),
                "dischtime": str(a_row.get("DISCHTIME", "")),
                "clinical_notes": "\n\n--- NEXT NOTE ---\n\n".join(texts),
                "hospital_expire_flag": bool(a_row.get("HOSPITAL_EXPIRE_FLAG", 0)),
                "num_diagnoses": int(extra["num_diagnoses"]) if extra is not None else 0,
                "had_icu_stay": bool(extra["had_icu_stay"]) if extra is not None else False,
                "service": str(extra["service"]) if extra is not None else "UNKNOWN",
            })
        if patient_dict["admissions"]:
            final_data.append(patient_dict)

    print(f"Phase 3 done: {len(final_data)} patients with full records extracted")
    return final_data

def write_balance_report(balance_info: dict, n: int, seed: int, selected_subject_ids: list[int]):
    representative = balance_info["representative"]
    selected_set = set(selected_subject_ids)
    selected_df = representative[representative["SUBJECT_ID"].isin(selected_set)]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(f"# MIMIC-III stratified patient selection report (n={n}, seed={seed})\n\n")
        f.write(
            f"Selected from the full population of {balance_info['population_n']:,} MIMIC-III "
            "patients with >=1 target clinical note. Stratification variables: sex, age quintile, "
            "and total note-length quintile.\n\n"
        )

        def proportions(df: pd.DataFrame, col: str) -> dict:
            counts = df[col].value_counts()
            total = len(df)
            return {str(k): f"{v}/{total} ({100*v/total:.1f}%)" for k, v in counts.items()}

        f.write("\n## Stratum marginals: population vs selected\n\n")
        for label, col in [("Sex", "gender"), ("Age quintile", "age_quintile"), ("Note-length quintile", "note_length_quintile")]:
            f.write(f"\n### {label}\n\n| Value | Population | Selected |\n|---|---|---|\n")
            pop_props = proportions(representative, col)
            sel_props = proportions(selected_df, col)
            for value in sorted(set(pop_props) | set(sel_props)):
                f.write(f"| {value} | {pop_props.get(value, '0')} | {sel_props.get(value, '0')} |\n")

    print(f"Wrote {REPORT_MD}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of patients to select (extended sample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    args = parser.parse_args()

    noteevents_path = get_file_path(args.base_dir, "NOTEEVENTS")
    note_index = phase1_note_length_index(noteevents_path)

    print("Loading auxiliary tables once...")
    patients = pd.read_csv(get_file_path(args.base_dir, "PATIENTS"), usecols=["SUBJECT_ID", "GENDER", "DOB"])
    admissions = pd.read_csv(
        get_file_path(args.base_dir, "ADMISSIONS"),
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE", "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG"],
    )
    diagnoses = load_diagnoses(get_file_path(args.base_dir, "DIAGNOSES_ICD"))
    icustays = load_icustays(get_file_path(args.base_dir, "ICUSTAYS"))
    services = load_services(get_file_path(args.base_dir, "SERVICES"))

    selected_subject_ids, balance_info = phase2_select_patients(
        note_index, patients, admissions, diagnoses, icustays, services, n=args.n, seed=args.seed
    )
    write_balance_report(balance_info, n=args.n, seed=args.seed, selected_subject_ids=selected_subject_ids)

    final_data = phase3_pull_full_records(
        selected_subject_ids, patients, admissions, diagnoses, icustays, services, noteevents_path
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_JSON} ({len(final_data)} patients, "
          f"{sum(len(p['admissions']) for p in final_data)} admissions)")

if __name__ == "__main__":
    main()