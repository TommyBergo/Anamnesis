"""Selects a demographically stratified patient sample from MIMIC-IV (hosp, icu, and note modules) and writes it to mimic_stratified_sample.json."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from shared.mimic_common import compute_age_at_admission, compute_los_days
from shared.mimic_iv import (
    NOTE_CATEGORY_ORDER,
    NOTE_COLUMNS,
    NOTE_DETAIL_COLUMNS,
    NOTE_SOURCES,
    REQUIRED_NOTE_TABLES,
    NoteSource,
    find_table,
    label_admission_type,
    label_discharge_location,
    label_note_type,
    label_service,
    race_group,
    require_table,
)
from shared.pipeline_paths import (
    MIMIC_IV_HOSP_DIR,
    MIMIC_IV_ICU_DIR,
    MIMIC_IV_NOTE_DIR,
    SAMPLE_JSON,
    SAMPLE_REPORT_MD,
)

# Balanced marginally (one variable at a time) so that adding a variable never fragments the
# population into near-empty joint strata.
STRATUM_VARIABLES = ["gender", "age_quintile", "race_group", "note_length_quintile"]

STRATUM_LABELS = {
    "gender": "Sex",
    "age_quintile": "Age quintile",
    "race_group": "Race group",
    "note_length_quintile": "Note-length quintile",
}

# MIMIC-IV-Note v2.2 discharge summaries average ~10.5k characters, so 25k rows keep each text chunk
# near 300 MB in memory. Detail tables carry no free text and can be read in larger chunks.
NOTE_CHUNK_ROWS = 25_000
DETAIL_CHUNK_ROWS = 500_000


def quintile_bucket(value: float, sorted_values: list[float]) -> str:
    n = len(sorted_values)
    for band, label in [(1, "Q1_0-20"), (2, "Q2_20-40"), (3, "Q3_40-60"), (4, "Q4_60-80")]:
        cut = sorted_values[min(n - 1, (band * n) // 5)]
        if value <= cut:
            return label
    return "Q5_80-100"


def available_note_sources(note_dir: Path) -> list[tuple[NoteSource, Path]]:
    available = []
    for source in NOTE_SOURCES:
        path = find_table(note_dir, source.table)
        if path is None:
            if source.table in REQUIRED_NOTE_TABLES:
                require_table(note_dir, source.table)
            print(f"  note source '{source.table}' ({source.category}) not found - skipped")
            continue
        print(f"  note source '{source.table}' ({source.category}): {path}")
        available.append((source, path))
    return available


def stream_notes(path: Path):
    for chunk in pd.read_csv(
        path,
        usecols=NOTE_COLUMNS,
        dtype={"note_id": str, "subject_id": "Int64", "hadm_id": "Int64", "note_type": str, "text": str},
        chunksize=NOTE_CHUNK_ROWS,
        low_memory=False,
    ):
        yield chunk[chunk["hadm_id"].notna() & chunk["text"].notna()]


def phase1_note_length_index(note_sources: list[tuple[NoteSource, Path]]) -> pd.DataFrame:
    print("Phase 1: streaming clinical-note tables for per-admission note lengths...")
    partial_lengths: list[pd.DataFrame] = []
    for source, path in note_sources:
        rows_seen = 0
        for chunk in stream_notes(path):
            rows_seen += len(chunk)
            partial_lengths.append(
                chunk.assign(note_length=chunk["text"].str.strip().str.len())
                .groupby(["subject_id", "hadm_id"], as_index=False)["note_length"]
                .sum()
            )
        print(f"  {source.table}: {rows_seen:,} notes linked to an admission")
    lengths = (
        pd.concat(partial_lengths, ignore_index=True)
        .groupby(["subject_id", "hadm_id"], as_index=False)["note_length"]
        .sum()
        .astype("int64")
    )
    print(f"Phase 1 done: {len(lengths):,} admissions have at least one clinical note")
    return lengths


def load_principal_diagnoses(diagnoses_path: Path, dictionary_path: Path) -> pd.DataFrame:
    print(f"Loading {diagnoses_path.name} and {dictionary_path.name}...")
    diagnoses = pd.read_csv(
        diagnoses_path,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    ).dropna(subset=["hadm_id", "icd_code"])
    dictionary = pd.read_csv(dictionary_path, dtype={"icd_code": str})
    dictionary["icd_code"] = dictionary["icd_code"].str.strip()
    diagnoses["icd_code"] = diagnoses["icd_code"].str.strip()

    counts = diagnoses.groupby(["subject_id", "hadm_id"]).size().rename("num_diagnoses").reset_index()

    principal = (
        diagnoses.sort_values("seq_num")
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .first()
        .merge(dictionary, on=["icd_code", "icd_version"], how="left")
        .rename(columns={"long_title": "diagnosis", "icd_code": "principal_icd_code", "icd_version": "principal_icd_version"})
    )
    principal = principal[["subject_id", "hadm_id", "diagnosis", "principal_icd_code", "principal_icd_version"]]
    merged = counts.merge(principal, on=["subject_id", "hadm_id"], how="left")
    print(f"  {len(merged):,} admissions have >=1 diagnosis code")
    return merged


def load_icustays(icustays_path: Path) -> pd.DataFrame:
    print(f"Loading {icustays_path.name}...")
    icu = pd.read_csv(icustays_path, usecols=["subject_id", "hadm_id"]).dropna(subset=["hadm_id"])
    icu = icu.drop_duplicates(subset=["subject_id", "hadm_id"])
    icu["had_icu_stay"] = True
    return icu


def load_services(services_path: Path) -> pd.DataFrame:
    print(f"Loading {services_path.name}...")
    services = pd.read_csv(services_path, usecols=["subject_id", "hadm_id", "transfertime", "curr_service"])
    services = services.dropna(subset=["hadm_id"]).sort_values("transfertime")
    first_service = services.groupby(["subject_id", "hadm_id"], as_index=False).first()
    return first_service[["subject_id", "hadm_id", "curr_service"]].rename(columns={"curr_service": "service_code"})


def join_admission_attributes(
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    icustays: pd.DataFrame,
    services: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["subject_id", "hadm_id"]
    merged = admissions.merge(diagnoses, on=keys, how="left")
    merged = merged.merge(icustays, on=keys, how="left")
    merged = merged.merge(services, on=keys, how="left")
    merged["num_diagnoses"] = merged["num_diagnoses"].fillna(0).astype(int)
    merged["had_icu_stay"] = merged["had_icu_stay"].eq(True)
    return merged


def select_patients_stratified(representative: pd.DataFrame, n: int, seed: int) -> list[int]:
    population_n = len(representative)
    target_share = {
        variable: representative[variable].value_counts(normalize=True).to_dict()
        for variable in STRATUM_VARIABLES
    }

    rng = random.Random(seed)
    order = sorted(representative["subject_id"].tolist())
    rng.shuffle(order)
    by_subject = representative.set_index("subject_id").to_dict("index")

    selected: list[int] = []
    selected_set: set[int] = set()
    marginal_counts: dict[str, Counter] = {variable: Counter() for variable in STRATUM_VARIABLES}

    for relaxed in (False, True):
        for subj in order:
            if len(selected) >= n:
                break
            if subj in selected_set:
                continue
            row = by_subject[subj]
            within_targets = all(
                marginal_counts[variable][row[variable]] / (len(selected) + 1)
                <= target_share[variable][row[variable]]
                for variable in STRATUM_VARIABLES
            )
            if relaxed or within_targets:
                selected.append(subj)
                selected_set.add(subj)
                for variable in STRATUM_VARIABLES:
                    marginal_counts[variable][row[variable]] += 1

    if n >= population_n:
        print(f"  requested n={n} >= eligible population ({population_n}); every eligible patient is kept")
    return selected[:n]


def phase2_select_patients(
    note_index: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    n: int,
    seed: int,
) -> tuple[list[int], pd.DataFrame]:
    print("Phase 2: joining tables and selecting a stratified patient sample...")
    merged = note_index.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.merge(patients, on="subject_id", how="inner")
    merged = merged.dropna(subset=["admittime", "anchor_age", "anchor_year"])
    print(f"  {len(merged):,} admissions with notes fully joined")

    merged["age"] = [
        compute_age_at_admission(age, year, admit)
        for age, year, admit in zip(merged["anchor_age"], merged["anchor_year"], merged["admittime"])
    ]
    merged = merged.dropna(subset=["age"])
    merged["race_group"] = merged["race"].map(race_group)

    representative = merged.sort_values(["admittime", "hadm_id"]).groupby("subject_id", as_index=False).first()
    print(f"  {len(representative):,} unique candidate patients (no diagnosis-based filtering)")

    ages_sorted = sorted(representative["age"].tolist())
    note_len_sorted = sorted(representative["note_length"].tolist())
    representative = representative.copy()
    representative["age_quintile"] = representative["age"].apply(lambda a: quintile_bucket(a, ages_sorted))
    representative["note_length_quintile"] = representative["note_length"].apply(
        lambda x: quintile_bucket(x, note_len_sorted)
    )

    selected_subject_ids = select_patients_stratified(representative, n=n, seed=seed)
    print(f"Phase 2 done: selected {len(selected_subject_ids)} patients")
    return selected_subject_ids, representative


def load_note_details(detail_path: Path | None, note_ids: set[str]) -> dict[str, dict[str, str]]:
    if detail_path is None or not note_ids:
        return {}
    matches = [
        chunk[chunk["note_id"].isin(note_ids)]
        for chunk in pd.read_csv(
            detail_path,
            usecols=lambda column: column in NOTE_DETAIL_COLUMNS,
            dtype=str,
            chunksize=DETAIL_CHUNK_ROWS,
        )
    ]
    if not matches:
        return {}
    rows = pd.concat(matches, ignore_index=True).dropna(subset=["field_name", "field_value"])
    if "field_ordinal" in rows.columns:
        rows = rows.assign(field_ordinal=pd.to_numeric(rows["field_ordinal"], errors="coerce"))
        rows = rows.sort_values(["note_id", "field_name", "field_ordinal"], kind="stable")

    details: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for note_id, field_name, field_value in zip(rows["note_id"], rows["field_name"], rows["field_value"]):
        details[note_id][field_name].append(field_value)
    return {
        note_id: {name: "; ".join(values) for name, values in fields.items()}
        for note_id, fields in details.items()
    }


def phase3_pull_notes(
    selected_set: set[int],
    note_sources: list[tuple[NoteSource, Path]],
    note_dir: Path,
) -> dict[tuple[int, int], list[dict]]:
    print("Phase 3: streaming clinical-note tables for the selected patients' full text...")
    notes_by_admission: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for source, path in note_sources:
        source_notes: list[tuple[tuple[int, int], dict]] = []
        for chunk in stream_notes(path):
            chunk = chunk[chunk["subject_id"].isin(selected_set)]
            for row in chunk.itertuples(index=False):
                note = {
                    "note_id": str(row.note_id),
                    "category": source.category,
                    "note_type": str(row.note_type) if pd.notna(row.note_type) else "",
                    "note_type_label": label_note_type(row.note_type),
                    "note_seq": int(row.note_seq) if pd.notna(row.note_seq) else 0,
                    "charttime": str(row.charttime) if pd.notna(row.charttime) else "",
                    "text": row.text.strip(),
                }
                source_notes.append(((int(row.subject_id), int(row.hadm_id)), note))

        details = load_note_details(
            find_table(note_dir, source.detail_table),
            {note["note_id"] for _, note in source_notes},
        )
        for key, note in source_notes:
            note["details"] = details.get(note["note_id"], {})
            notes_by_admission[key].append(note)
        print(f"  {source.table}: {len(source_notes):,} notes for selected patients")

    category_rank = {category: rank for rank, category in enumerate(NOTE_CATEGORY_ORDER)}
    for notes in notes_by_admission.values():
        notes.sort(key=lambda note: (category_rank[note["category"]], note["charttime"], note["note_seq"], note["note_id"]))

    print(f"Phase 3 done: clinical notes found for {len(notes_by_admission):,} admissions")
    return notes_by_admission


def optional_str(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def build_records(
    selected_subject_ids: list[int],
    patients: pd.DataFrame,
    admission_attributes: pd.DataFrame,
    notes_by_admission: dict[tuple[int, int], list[dict]],
) -> list[dict]:
    patients_by_id = patients.set_index("subject_id").to_dict("index")
    selected_attributes = admission_attributes[admission_attributes["subject_id"].isin(set(selected_subject_ids))]
    attributes_by_subject = {subj_id: group for subj_id, group in selected_attributes.groupby("subject_id")}
    final_data = []

    for subj_id in sorted(selected_subject_ids):
        p_row = patients_by_id[subj_id]
        p_admissions = attributes_by_subject[subj_id].sort_values(["admittime", "hadm_id"])
        admissions = []
        for a_row in p_admissions.to_dict("records"):
            hadm_id = int(a_row["hadm_id"])
            notes = notes_by_admission.get((subj_id, hadm_id))
            if not notes:
                continue
            admissions.append({
                "hadm_id": hadm_id,
                "admittime": str(a_row["admittime"]),
                "dischtime": str(a_row["dischtime"]),
                "admission_type": label_admission_type(a_row.get("admission_type")),
                "admission_type_code": optional_str(a_row.get("admission_type")),
                "diagnosis": optional_str(a_row.get("diagnosis")) or "",
                "principal_icd_code": optional_str(a_row.get("principal_icd_code")),
                "principal_icd_version": (
                    int(a_row["principal_icd_version"]) if pd.notna(a_row.get("principal_icd_version")) else None
                ),
                "age_at_admission": compute_age_at_admission(p_row["anchor_age"], p_row["anchor_year"], a_row["admittime"]),
                "length_of_stay_days": compute_los_days(a_row["admittime"], a_row["dischtime"]),
                "discharge_location": label_discharge_location(a_row.get("discharge_location")),
                "discharge_location_code": optional_str(a_row.get("discharge_location")),
                "hospital_expire_flag": bool(a_row.get("hospital_expire_flag") or 0),
                "num_diagnoses": int(a_row["num_diagnoses"]),
                "had_icu_stay": bool(a_row["had_icu_stay"]),
                "service": label_service(a_row.get("service_code")) or None,
                "service_code": optional_str(a_row.get("service_code")),
                "insurance": optional_str(a_row.get("insurance")),
                "race": optional_str(a_row.get("race")),
                "note_categories": dict(Counter(note["category"] for note in notes)),
                "notes": notes,
            })
        if not admissions:
            continue
        final_data.append({
            "patient_info": {
                "subject_id": subj_id,
                "gender": str(p_row["gender"]),
                "anchor_age": int(p_row["anchor_age"]),
                "anchor_year": int(p_row["anchor_year"]),
                "anchor_year_group": optional_str(p_row.get("anchor_year_group")),
                "race_group": race_group(admissions[0]["race"]),
            },
            "admissions": admissions,
        })
    return final_data


def write_balance_report(
    representative: pd.DataFrame,
    n: int,
    seed: int,
    selected_subject_ids: list[int],
    final_data: list[dict],
) -> None:
    selected_df = representative[representative["subject_id"].isin(set(selected_subject_ids))]
    category_totals: Counter = Counter()
    for patient in final_data:
        for adm in patient["admissions"]:
            category_totals.update(adm["note_categories"])

    def proportions(df: pd.DataFrame, col: str) -> dict:
        counts = df[col].value_counts()
        total = len(df)
        return {str(k): f"{v}/{total} ({100 * v / total:.1f}%)" for k, v in counts.items()}

    with SAMPLE_REPORT_MD.open("w", encoding="utf-8") as f:
        f.write(f"# MIMIC-IV stratified patient selection report (n={n}, seed={seed})\n\n")
        f.write(
            f"Selected from the full population of {len(representative):,} MIMIC-IV patients with >=1 "
            "clinical note linked to an admission. No diagnosis-based inclusion or exclusion is applied. "
            "Marginally balanced variables: " + ", ".join(STRATUM_LABELS[v].lower() for v in STRATUM_VARIABLES) + ".\n\n"
        )
        f.write(
            f"Selected patients: {len(final_data)}; admissions: "
            f"{sum(len(p['admissions']) for p in final_data)}.\n\n"
        )
        f.write("## Clinical notes by category (selected admissions)\n\n| Category | Notes |\n|---|---|\n")
        for category in NOTE_CATEGORY_ORDER:
            f.write(f"| {category} | {category_totals.get(category, 0)} |\n")

        f.write("\n## Stratum marginals: population vs selected\n")
        for col in STRATUM_VARIABLES:
            f.write(f"\n### {STRATUM_LABELS[col]}\n\n| Value | Population | Selected |\n|---|---|---|\n")
            pop_props = proportions(representative, col)
            sel_props = proportions(selected_df, col)
            for value in sorted(set(pop_props) | set(sel_props)):
                f.write(f"| {value} | {pop_props.get(value, '0')} | {sel_props.get(value, '0')} |\n")

    print(f"Wrote {SAMPLE_REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="number of patients to select")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hosp-dir", type=Path, default=MIMIC_IV_HOSP_DIR, help="MIMIC-IV hosp module folder")
    parser.add_argument("--icu-dir", type=Path, default=MIMIC_IV_ICU_DIR, help="MIMIC-IV icu module folder")
    parser.add_argument("--note-dir", type=Path, default=MIMIC_IV_NOTE_DIR, help="MIMIC-IV-Note note module folder")
    args = parser.parse_args()

    hosp_dir: Path = args.hosp_dir
    icu_dir: Path = args.icu_dir
    note_dir: Path = args.note_dir
    for label, directory in (("hosp", hosp_dir), ("icu", icu_dir), ("note", note_dir)):
        if not directory.is_dir():
            raise SystemExit(f"MIMIC-IV {label} directory not found: {directory}")
        print(f"MIMIC-IV {label} directory: {directory}")

    note_sources = available_note_sources(note_dir)
    note_index = phase1_note_length_index(note_sources)

    print("Loading auxiliary tables once...")
    patients = pd.read_csv(
        require_table(hosp_dir, "patients"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
    )
    admissions = pd.read_csv(
        require_table(hosp_dir, "admissions"),
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
            "discharge_location", "insurance", "race", "hospital_expire_flag",
        ],
    )
    diagnoses = load_principal_diagnoses(
        require_table(hosp_dir, "diagnoses_icd"), require_table(hosp_dir, "d_icd_diagnoses")
    )
    icustays = load_icustays(require_table(icu_dir, "icustays"))
    services = load_services(require_table(hosp_dir, "services"))
    admission_attributes = join_admission_attributes(admissions, diagnoses, icustays, services)

    selected_subject_ids, representative = phase2_select_patients(
        note_index, patients, admissions, n=args.n, seed=args.seed
    )

    notes_by_admission = phase3_pull_notes(set(selected_subject_ids), note_sources, note_dir)
    final_data = build_records(selected_subject_ids, patients, admission_attributes, notes_by_admission)
    SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_balance_report(representative, args.n, args.seed, selected_subject_ids, final_data)

    with SAMPLE_JSON.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {SAMPLE_JSON} ({len(final_data)} patients, "
        f"{sum(len(p['admissions']) for p in final_data)} admissions)"
    )


if __name__ == "__main__":
    main()
