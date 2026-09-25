# `rebuild_anamnesis/` — corpus & QA-dataset reproducibility scripts (MIMIC-IV)

This folder holds the scripts that build the Anamnesis clinical-record corpus (one PDF per hospital
admission, grouped per patient) and its rule-based QA benchmarks. The pipeline reads the official,
credentialed **MIMIC-IV v3.1** (`hosp` and `icu` modules) and **MIMIC-IV-Note v2.2** (`note`
module) releases from three configurable folders. It renders **several clinical note types** per
admission, emits **first-person** questions only, and **caps multi-document items per patient**.
The original MIMIC-III pipeline that produced the paper's 138-admission corpus is described under
"Historical provenance" below. The "Changelog" at the bottom lists every change made during the
transition.

## Requirements

- Python 3.10+
- `pandas` (step 1), `reportlab` (step 2), `pypdf` (steps 3-4)

## Input data

Only step 1 reads raw MIMIC data. It reads these tables from three module folders:

```text
MIMIC_IV_HOSP_DIR  (MIMIC-IV v3.1 hosp/)       patients, admissions, diagnoses_icd, d_icd_diagnoses, services
MIMIC_IV_ICU_DIR   (MIMIC-IV v3.1 icu/)        icustays
MIMIC_IV_NOTE_DIR  (MIMIC-IV-Note v2.2 note/)  discharge, discharge_detail, radiology, radiology_detail
                                               + optional: nursing, physician, consult (and their *_detail tables)
```

- Every table is looked up as `<table>.csv.gz` first, then `<table>.csv`, directly inside its
  module folder. Step 1 checks that all three folders exist and stops with a clear message if one
  is missing. A missing required table raises `FileNotFoundError` with the folder it searched.
- MIMIC-IV-Note is a **separate PhysioNet release** from MIMIC-IV, which is why the note folder is
  configured independently. Both require credentialed access.
- **MIMIC-IV-Note v2.2 only ships the `discharge` and `radiology` tables.** Nursing, physician, and
  consult notes exist only in MIMIC-III's `NOTEEVENTS`. The pipeline therefore treats `nursing`,
  `physician`, and `consult` as **optional** tables. They are ingested automatically when present
  in the same schema, and skipped with a log line when absent. Only `discharge` is mandatory.
- Note tables use the schema `note_id, subject_id, hadm_id, note_type, note_seq, charttime,
  storetime, text`. The v2.2 detail tables use `note_id, subject_id, field_name, field_value,
  field_ordinal`. `field_ordinal` is optional, so three-column detail tables (`note_id,
  field_name, field_value`) also load.
- About 52% of Note v2.2 radiology reports have no `hadm_id` (mostly outpatient and emergency
  department studies). They cannot be attached to an admission and are skipped.

### Path configuration

All paths live in `shared/pipeline_paths.py`. The three input folders are defined at the top of
that file; edit them there when deploying to another machine, or override them per run:

| Setting | Variable | Default | Override |
|---|---|---|---|
| MIMIC-IV `hosp` module | `MIMIC_IV_HOSP_DIR` | `/home/tommaso/datasets/MIMICIV/3.1/hosp` | `ANAMNESIS_HOSP_DIR` env var, or `--hosp-dir` (step 1) |
| MIMIC-IV `icu` module | `MIMIC_IV_ICU_DIR` | `/home/tommaso/datasets/MIMICIV/3.1/icu` | `ANAMNESIS_ICU_DIR` env var, or `--icu-dir` (step 1) |
| MIMIC-IV-Note `note` module | `MIMIC_IV_NOTE_DIR` | `/home/tommaso/datasets/MIMICIV/mimic-iv-note/2.2/note` | `ANAMNESIS_NOTE_DIR` env var, or `--note-dir` (step 1) |
| Output ("work") directory | `WORK_DIR` | `/home/tommaso/anamnesis_output` (outside the repository) | `ANAMNESIS_WORK_DIR` env var |

Precedence is command-line flag, then environment variable, then the default in
`pipeline_paths.py`. `WORK_DIR` applies to every step. The input folders are only read by step 1.

Every generated file (sample JSON, balance report, PDFs, QA JSONL) is written to `WORK_DIR`, never
into the repository: these outputs contain credentialed MIMIC text. Step 1 creates the folder if it
does not exist. As a safety net, `rebuild_anamnesis/.gitignore` also ignores every output file name
and `__pycache__/`, in case `ANAMNESIS_WORK_DIR` is ever pointed inside the repository.

Example (server deployment: point every module at the server's copy, write outputs elsewhere):

```bash
export ANAMNESIS_HOSP_DIR=/data/mimiciv/3.1/hosp
export ANAMNESIS_ICU_DIR=/data/mimiciv/3.1/icu
export ANAMNESIS_NOTE_DIR=/data/mimic-iv-note/2.2/note
export ANAMNESIS_WORK_DIR=/scratch/anamnesis_run
```

Run steps 2-5 in the same shell, so they read step 1's output from the same `ANAMNESIS_WORK_DIR`.

## Layout

```text
rebuild_anamnesis/
├── extract_stratified_sample.py     step 1
├── render_admission_pdfs.py         step 2
├── generate_qa_single.py            step 3
├── generate_qa_multidoc.py          step 4
├── split_qa_multidoc_by_patient.py  step 5
├── .gitignore                       keeps pipeline outputs and bytecode out of git
└── shared/
    ├── pipeline_paths.py            input/output locations for every step
    ├── mimic_iv.py                  MIMIC-IV table lookup, note-source registry, code-to-label maps
    ├── mimic_common.py              age-at-admission (anchor_age/anchor_year) and length-of-stay helpers
    ├── mimic_section_parser.py      per-note-category section splitter (discharge, radiology, nursing, physician, consult)
    ├── pdf_paging.py                per-page PDF text + offset tracking (steps 3, 4)
    └── kotlin_mirror.py             Python port of the app's chunking logic (used by pdf_paging.py)
```

## Pipeline order

Run from inside `rebuild_anamnesis/` (the scripts import `shared.*`), in this order:

```bash
python extract_stratified_sample.py          # [--n 300] [--seed 42] [--hosp-dir ...] [--icu-dir ...] [--note-dir ...]
python render_admission_pdfs.py
python generate_qa_single.py
python generate_qa_multidoc.py
python split_qa_multidoc_by_patient.py
```

| Step | Reads | Writes (in the work directory) |
|---|---|---|
| 1 | MIMIC-IV CSVs | `mimic_stratified_sample.json`, `mimic_stratified_sample_report.md` |
| 2 | the JSON | `Admission_PDFs_stratified_enriched/Patient_{subject_id}_Admission_{hadm_id}.pdf` |
| 3 | the JSON + PDFs | `single_dataset_rulebased.jsonl` (two QA pairs per admission) |
| 4 | the JSON + PDFs | `multidoc_rulebased.jsonl` (three task families combined) |
| 5 | step 4's output | `multidoc_dataset_rulebased_main_same_patient.jsonl` (shipped to the app), `multidoc_dataset_rulebased_cross_patient.jsonl` (auxiliary probe) |

## Step detail

### 1. `extract_stratified_sample.py`

- **Phase 1** streams every available note table in 25,000-row chunks (a Note v2.2 discharge
  summary averages ~10.5k characters, so each chunk holds ~300 MB of text). It sums note length
  per `(subject_id, hadm_id)`, across all note categories, with vectorized per-chunk group sums.
  Only notes linked to an admission (`hadm_id` not null) with non-null text count.
- **Phase 2** joins `admissions` and `patients` to that index and computes age at admission from
  `anchor_age`/`anchor_year`. It keeps each patient's earliest noted admission as that patient's
  stratification profile. It then selects `--n` patients (default 300; every eligible patient is
  kept when the population is smaller), with **no diagnosis-based
  inclusion or exclusion**. Selection is a seeded greedy pass that keeps the sample's marginal
  shares of **sex, age quintile, race group, and note-length quintile** at or below their
  population shares, followed by a relaxed pass that fills any remaining slots.
- **Phase 3** streams the note tables again for the selected patients only, attaches each note's
  `*_detail` fields (multi-valued fields ordered by `field_ordinal` when present), and orders each
  admission's notes: discharge summary first, then radiology, nursing, physician, and consult,
  each chronologically by `charttime`, then `note_seq`.
- Every admission of a selected patient that has at least one note is kept.
- The balance report compares population and sample marginals for each stratification variable
  and counts notes per category.

JSON shape (per patient):

```text
patient_info: subject_id, gender, anchor_age, anchor_year, anchor_year_group, race_group
admissions[]: hadm_id, admittime, dischtime,
              admission_type (readable label), admission_type_code (raw MIMIC-IV value),
              diagnosis (ICD long title of seq_num 1), principal_icd_code, principal_icd_version,
              age_at_admission, length_of_stay_days,
              discharge_location (readable label, null if missing or DIED), discharge_location_code,
              hospital_expire_flag, num_diagnoses, had_icu_stay,
              service (readable label), service_code (raw curr_service of the first transfer),
              insurance, race, note_categories {category: count},
              notes[]: note_id, category, note_type, note_type_label, note_seq, charttime, text, details{}
```

### 2. `render_admission_pdfs.py`

Renders one A4 PDF per admission with reportlab.

- **Structured header**, one field group per line:
  1. Patient ID, Sex, Age at Admission
  2. Admission ID, Type, Hospital Service
  3. Admission Date, Discharge Date, Length of Stay
  4. Principal Diagnosis
  5. Discharge Disposition
  6. ICU Stay, In-Hospital Mortality, Number of Diagnoses
  7. Clinical Notes Included, e.g. "Discharge summary (1), Radiology (1)"
- **Discharge Disposition** is resolved by `shared.mimic_iv.resolve_disposition`, in order:
  1. the structured `discharge_location`
  2. "Died in hospital" when `hospital_expire_flag` is set
  3. the discharge summary's "Discharge Disposition:" section
  4. "Not recorded"
- **Body:** every note of the stay under a heading such as `Clinical note 2 of 2: Radiology -
  Radiology report addendum - charted 2196-03-04 14:02:00 - Modality: XR`, followed by the
  **verbatim** note text (Courier 9 pt). Note text is never rewritten, so gold evidence stays an
  exact substring of the PDF.

### 3. `generate_qa_single.py`

- Deterministic (non-LLM) generator: exactly two question types per admission.
- Types are balanced across the corpus by the same min-cost-flow assignment as before. The types
  are `header_fact`, `header_fact_enriched`, `section_lookup`, `specific_detail` (history of
  present illness), `negation_check` (allergies), and `multi_admission_distractor`.
- Each note is parsed with the section schema of its category.
- Discharge-summary sections keep their original names. Sections from other note types are keyed
  `"<category>: <section>"`:
  - Radiology: Impression, Findings
  - Nursing: Assessment, Plan
  - Physician: Assessment and Plan
  - Consult: Impression, Recommendations

  These are available to `section_lookup` alongside the discharge-summary sections.
- Every gold chunk must still be contained in one Android-equivalent 500/100 chunk of the
  re-extracted PDF text.

### 4. `generate_qa_multidoc.py`

Three task families:

- `same_patient_two_admissions`: **consecutive** admissions of one patient; at most 4 pairs per
  patient.
- `same_patient_three_admission_trajectory`: three consecutive admissions; at most 2 per patient.
- `cross_patient_two_admissions`: two patients who share a principal diagnosis; at most 1 item per
  patient pair.

Fields compared: ICU stay, hospital service, principal diagnosis, and number of diagnoses. The
cross-patient family does not compare principal diagnosis, because the shared diagnosis is its
anchor. Every header value used as evidence is verified by exact search in the re-extracted PDF
text.

### 5. `split_qa_multidoc_by_patient.py`

A lossless partition of step 4's output by `comparison_type`: same-patient items vs. cross-patient
items. It fails loudly on any unrecognized type.

## Question design rules

- **First person, always.** Single-document questions must contain `I`, `me`, `my`, `mine`, or
  `myself`. Multi-document questions may also use `we`, `us`, or `our`, for the cross-patient
  caregiver framing.
  - `add()` in step 3 raises on a non-first-person template.
  - `validate_items()` (step 3) and `validate_final()` (step 4) reject any such item before
    anything is written.
- **Admissions are referenced by date**, never by identifier; for example "my admission beginning
  on 2196-02-24". Admission and patient identifiers never appear in a question, and step 3 also
  rejects any 5+ digit number.
- **Cross-patient items are framed as a caregiver** who keeps a family member's records together
  with their own, the setting motivated in the paper (§3.3). The first admission is "mine" and
  the second is "my family member's".
- **Realism:** questions a patient cannot meaningfully ask about their own stay are not generated.
  In-hospital mortality is not asked, and a disposition question is skipped when the disposition
  is "Died in hospital" or "Not recorded".
- **De-identification:** items whose question or answer contains a MIMIC-III surrogate `[**...**]`
  or a MIMIC-IV placeholder `___` are rejected. The history-of-present-illness answer neutralizes
  them into "(removed)"-style placeholders.

## Why multi-document items are capped (design note)

The reference Android evaluation tests (`app/src/androidTest/.../rag/*MultiDoc*EvaluationTest.kt`)
read the dataset with `readLines()` and evaluate items one at a time against a per-patient index.

- **Memory and storage are not affected by item count.** The index size, and with it peak RSS and
  ObjectBox storage, depends only on the corpus (PDFs → chunks). The 762 multi-document items of
  the reference run are ~0.7 MB of JSON.
- **Evaluation time grows linearly with item count.** Every item costs one retrieval plus one full
  generation, at ~4.4 tokens/s on the reference device, for each evaluated retriever × generator ×
  evidence-condition configuration.
- **Scores need statistical balance.** All-pairs enumeration is quadratic in a patient's number of
  admissions: one patient with 20 admissions alone would produce 190 pairs, so the
  multi-document score would largely measure that one record.

Pairs are therefore restricted to consecutive admissions, the same rule the paper already applies
to trajectories so that the temporal reading stays unambiguous. Each patient contributes a bounded
number of windows. When a patient has more windows than the cap, `spread_priorities()` prefers
windows spread evenly over the timeline, e.g. pairs (0,1), (6,7), (12,13), (18,19) of a
20-admission history. It falls back to the remaining windows only if a preferred one fails
grounding. The caps are the constants `MAX_PAIRS_PER_PATIENT`, `MAX_TRAJECTORIES_PER_PATIENT`, and
`MAX_CROSS_ITEMS_PER_PATIENT_PAIR` at the top of `generate_qa_multidoc.py`.

## Reference run on MIMIC-IV v3.1 + MIMIC-IV-Note v2.2 (2026-09-25)

Defaults (`--n 300 --seed 42`, default `WORK_DIR`), on a 15 GB-RAM workstation.

| Output | Count |
|---|---|
| Eligible population (patients with ≥1 note linked to an admission) | 161,180 patients; 374,285 admissions |
| Selected patients / admissions / PDFs | 300 / 685 / 685 |
| Notes rendered | 614 discharge summaries + 2,076 radiology reports (Note v2.2 has no nursing/physician/consult tables) |
| Single-document items | 1,370 (2 per admission): header_fact 229, header_fact_enriched 229, section_lookup 228, specific_detail 228, negation_check 228, multi_admission_distractor 228 |
| Multi-document, same-patient pairs | 289 (385 consecutive pairs before grounding and caps) |
| Multi-document, trajectories | 130 (256 before grounding and caps) |
| Multi-document, cross-patient probe | 343 (490 before grounding and the one-item-per-patient-pair cap) |
| Patients / admissions covered by multi-document items | 183 / 550 |
| Grounding failures | 0 |
| Step 1 run time / peak memory | ~3 min / 3.5 GB RSS |

- The selected sample matches the population shares of every balanced variable to within 0.3
  percentage points (see `mimic_stratified_sample_report.md`).
- 71 of the 685 selected admissions have radiology reports but no discharge summary. Their PDFs
  contain radiology notes only.

The optional nursing, physician, and consult path was verified separately, with test tables in the
MIMIC-IV-Note schema kept outside the repository. All three categories were ingested, rendered, and
turned into grounded first-person `section_lookup` items with 0 grounding failures.

## Historical provenance (original MIMIC-III release)

The paper's corpus (100 patients, 138 admissions, discharge summaries only) was built by the
MIMIC-III predecessors of these scripts. They were copied from a historical, read-only research repo
(`depression_rag`, last checked 2026-09-16) into this flat folder. That copy flattened the layout,
fixed imports, produced English filenames directly, and trimmed comments, all verified
token-for-token logic-preserving.

- `split_qa_multidoc_by_patient.py` did not exist in the original pipeline. It reproduces a manual
  split whose criterion was reverse-engineered and verified as a clean, lossless partition: 81
  same-patient items in, 35 cross-patient items out.
- Evidence that the copied scripts were the ones actually used:
  - `ENRICHED_CORPUS_METHODOLOGY.md` in `depression_rag` documents the same two-script corpus
    pipeline by name.
  - The renderer's timestamp and output bytes match the PDFs the app ships.
  - `generate_qa_single.py`'s output was byte-identical (923,851 bytes) to the app's shipped
    `single_dataset_rulebased.jsonl`.

Candidates that were read in full and rejected at the time:

| Script | Why rejected |
|---|---|
| `extract_mimic.py` | Earliest draft: plain `random.sample()`, no stratification; produced the older 118-admission corpus. |
| `json_to_pdf.py` | First PDF renderer - non-stratified source, basic Italian header, 5 fields. |
| `json_to_pdf_stratified.py` | Stratified source but the old basic Italian rendering (no enrichment). |
| `json_to_pdf_enriched.py` | Enrichment added, but sourced from the old non-stratified JSON with broken CSV joins. |
| `json_to_pdf_stratified_enriched_json_only.py` | Non-functional draft requiring JSON fields that did not exist. |
| `generate_dataset_single_rulebased_v11.py` / `_v12.py` / `_v13.py` | Superseded same-day drafts of `generate_qa_single.py`. |
| `generate_multidoc_rulebased_v2.py` | Earlier draft of `generate_qa_multidoc.py`. |
| ~15 `dump_*`/`extract_*`/`build_batch*_items.py` files | A separate, older multidoc candidate-mining pipeline; out of scope. |

To reproduce the paper's exact MIMIC-III corpus, check out the scripts at commit `1610baf`
("Repo created"), which predates this refactoring.

## What is *not* included here

- **Raw data and outputs:** no MIMIC tables or derived JSON/PDF/JSONL files are committed. They
  live in `WORK_DIR`, outside the repository, and are excluded by `.gitignore`. Credentialed data
  cannot be redistributed under the PhysioNet DUA.
- **Real MIMIC-IV / MIMIC-IV-Note tables:** they are read from the folders configured in
  `shared/pipeline_paths.py`, outside the repository. Regenerating the benchmark requires
  credentialed PhysioNet access to MIMIC-IV v3.1 and MIMIC-IV-Note v2.2.

---

## Changelog

### Overview

Every change since the MIMIC-III baseline (commit `1610baf`), by theme:

| Theme | What changed | Section |
|---|---|---|
| **MIMIC-IV migration** | MIMIC-III → MIMIC-IV v3.1 (`hosp`, `icu`) and MIMIC-IV-Note v2.2 (`note`), read from three configurable folders | §A |
| **Broader cohort** | Mental-health filtering and stratification removed; marginal balancing on sex, age, race group, and note length | §B |
| **Multi-note support** | Discharge, radiology, nursing, physician, and consult notes ingested, parsed, rendered, and queried (nursing, physician, and consult are optional because Note v2.2 does not ship them) | §C |
| **First-person questions** | Every single-document, multi-document, and trajectory question is phrased from the patient's point of view and validated before writing | §D |
| **Strict multi-document capping** | Consecutive admissions only; at most 4 pairs and 2 trajectories per patient and 1 cross-patient item per patient pair, asserted at validation | §E |

### 2026-09-24 – 2026-09-25 — MIMIC-III → MIMIC-IV v3.1 + MIMIC-IV-Note v2.2, multi-note corpus, first-person questions

The baseline for everything below is commit `1610baf`. At that commit the tree already contained
partial, uncommitted-at-the-time edits toward a multi-category corpus:

- a `TARGET_CATEGORIES` set of MIMIC-III `NOTEEVENTS` category names
- a `clinical_notes` JSON key joined with `--- NEXT NOTE ---`
- a default `--n 300`
- no mental-health stratum in step 1
- a few Italian inline comments

The README at that commit still described the original MIMIC-III, mental-health-stratified
pipeline. Changes are grouped by theme, then by file.

#### A. Data source: MIMIC-III → MIMIC-IV v3.1 and MIMIC-IV-Note v2.2

1. **New `shared/pipeline_paths.py`.** A single place for every input and output path:
   - Three input-folder settings, one per module, since MIMIC-IV-Note is distributed separately
     from MIMIC-IV and usually lives elsewhere on disk:
     - `MIMIC_IV_HOSP_DIR` = `/home/tommaso/datasets/MIMICIV/3.1/hosp`, overridable with
       `ANAMNESIS_HOSP_DIR`
     - `MIMIC_IV_ICU_DIR` = `/home/tommaso/datasets/MIMICIV/3.1/icu`, overridable with
       `ANAMNESIS_ICU_DIR`
     - `MIMIC_IV_NOTE_DIR` = `/home/tommaso/datasets/MIMICIV/mimic-iv-note/2.2/note`,
       overridable with `ANAMNESIS_NOTE_DIR`
   - `WORK_DIR` = `/home/tommaso/anamnesis_output`, outside the repository, overridable with
     `ANAMNESIS_WORK_DIR`. Generated outputs previously tracked in `rebuild_anamnesis/` (the
     sample JSON and report, the QA JSONL files, and the PDF folder) were deleted from the
     repository.
   - Derived paths: `SAMPLE_JSON`, `SAMPLE_REPORT_MD`, `PDF_DIR`, `SINGLE_QA_PATH`,
     `MULTIDOC_QA_PATH`, `MULTIDOC_SAME_PATIENT_PATH`, `MULTIDOC_CROSS_PATIENT_PATH`.
   - `admission_pdf_name()` builds `Patient_{subject_id}_Admission_{hadm_id}.pdf`.

   Before this, each script defined its own `BASE_DIR`-relative paths, and step 1 defaulted to
   `/gringotts/datasets/MIMIC-III`.
2. **New `shared/mimic_iv.py`.** MIMIC-IV schema knowledge:
   - `find_table` / `require_table`: `<table>.csv.gz`, then `<table>.csv`, inside one module
     folder. `require_table` raises `FileNotFoundError` naming the folder it searched.
   - `ADMISSION_TYPE_LABELS`: the nine MIMIC-IV admission types become readable labels, e.g.
     `EW EMER.` → "Emergency", `SURGICAL SAME DAY ADMISSION` → "Surgical same-day admission".
   - `SERVICE_LABELS`: the 21 `services.curr_service` codes become readable names, e.g. `CMED` →
     "Cardiac Medicine", `MED` → "General Medicine", `OMED` → "Oncology".
   - `DISCHARGE_LOCATION_LABELS` and `label_discharge_location()`: sentence-case labels, with
     overrides such as `HOME HEALTH CARE` → "Home with home health care" and `AGAINST ADVICE` →
     "Left against medical advice". `DIED` maps to no destination.
   - All admission types, services, and discharge locations present in MIMIC-IV v3.1 are covered
     by these maps.
   - `resolve_disposition()`, shared by steps 2 and 3 so the header and the QA evidence always
     agree.
   - `race_group()`: collapses MIMIC-IV `race` into White / Black / Hispanic or Latino / Asian /
     Other / Unknown.
   - The note-source registry (see C) and `NOTE_DETAIL_COLUMNS`, the detail-table columns step 1
     reads.
3. **`extract_stratified_sample.py` rewritten for the MIMIC-IV schema.**
   - Lowercase MIMIC-IV column names throughout (`subject_id`, `hadm_id`, …) instead of MIMIC-III
     uppercase.
   - Tables read:
     - `hosp/patients`: `gender`, `anchor_age`, `anchor_year`, `anchor_year_group`
     - `hosp/admissions`: `admittime`, `dischtime`, `admission_type`, `discharge_location`,
       `insurance`, `race`, `hospital_expire_flag`
     - `hosp/diagnoses_icd` + `hosp/d_icd_diagnoses`
     - `hosp/services`
     - `icu/icustays`
     - note tables from `note/`
   - **Principal diagnosis:** MIMIC-IV `admissions` has no free-text `DIAGNOSIS` column. The
     principal diagnosis is now the `long_title` of the `seq_num`-first code in `diagnoses_icd`,
     joined on (`icd_code`, `icd_version`) so ICD-9 and ICD-10 are both handled.
     `principal_icd_code` and `principal_icd_version` are stored alongside it.
   - **Number of diagnoses:** the count of `diagnoses_icd` rows per admission (was
     `ICD9_CODE`-only).
   - **Hospital service:** the first `curr_service` by `transfertime`, as before. It is now stored
     as a readable `service` plus the raw `service_code`.
   - **Admission type:** stored as a readable label plus the raw `admission_type_code`.
   - **Discharge disposition:** taken from the structured `admissions.discharge_location`
     (readable label plus raw code). It was previously parsed from the note body only.
   - New per-admission fields: `insurance`, `race`, `note_categories`.
   - **Note detail tables:** MIMIC-IV-Note v2.2 detail tables are `note_id, subject_id,
     field_name, field_value, field_ordinal`. Only `NOTE_DETAIL_COLUMNS` are read, and
     `field_ordinal` orders multi-valued fields (e.g. several `exam_code` rows). `field_ordinal`
     is optional, so three-column detail tables also load.
   - New CLI: `--hosp-dir`, `--icu-dir`, and `--note-dir` (replacing `--base-dir`), each
     defaulting to the matching `pipeline_paths` setting. Step 1 prints the three folders and
     exits with a clear message when one does not exist.
   - **Full-scale memory and speed:**
     - Note text is streamed in 25,000-row chunks (`NOTE_CHUNK_ROWS`); a 100,000-row chunk of
       Note v2.2 discharge summaries would hold ~1.1 GB of text.
     - Detail tables are streamed in 500,000-row chunks (`DETAIL_CHUNK_ROWS`);
       `radiology_detail` alone has ~6 million rows.
     - Phase 1 sums note lengths with vectorized per-chunk `groupby`.
     - `build_records()` filters admission attributes to the selected patients once, instead of
       scanning all ~546,000 admissions per patient.

     The full run takes ~3 minutes with a 3.5 GB peak (see "Reference run").
   - `patients` is loaded once and indexed by `subject_id`. Patients and admissions are emitted in
     deterministic order: by `subject_id`, then by `admittime` and `hadm_id`.
   - Fixed a pandas `FutureWarning`: `had_icu_stay` is now computed with `.eq(True)` instead of a
     downcasting `.fillna(False)`.
   - The output directory is created before writing, so `ANAMNESIS_WORK_DIR` may point to a
     folder that does not exist yet.
4. **Age at admission (`shared/mimic_common.py`).** MIMIC-IV has no date of birth.
   - `compute_age(dob, admittime)` and the MIMIC-III constants `MIMIC_90_PLUS_SENTINEL_AGE=90` /
     `MIMIC_90_PLUS_RAW_THRESHOLD=150` were replaced.
   - New: `compute_age_at_admission(anchor_age, anchor_year, admittime)` = `anchor_age +
     (admission year − anchor_year)`.
   - Patients top-coded by MIMIC-IV (`anchor_age` ≥ 91) are reported as
     `MIMIC_IV_TOP_CODED_AGE = 91`.
   - `compute_los_days` now also returns `None` for negative stays.
   - Age and length of stay are computed **once in step 1** and stored as `age_at_admission` and
     `length_of_stay_days`.
   - The duplicate DOB-based `compute_age` / `compute_los_days` copies inside
     `generate_qa_single.py` were deleted, together with its `MIMIC_90_*` constants and the unused
     `datetime`, `sys`, and `Path` imports.
5. **Date of birth removed from the PDF header** (step 2), since MIMIC-IV does not provide it.
   `patient_info.dob` no longer exists. It is replaced by `anchor_age`, `anchor_year`,
   `anchor_year_group`, and `race_group`.

#### B. Mental-health focus removed, broader cohort

6. **No diagnosis-based inclusion, exclusion, or stratification.** There is no mental-health
   (ICD-9 290–319) stratum, filter, or flag anywhere in the pipeline, and the balance report
   states this explicitly.
7. **Mental-health question removed** from `generate_qa_single.py`. The
   `has_mental_health_diagnosis` → "Was any mental health condition recorded for me…" template,
   subkey `mental_health`, is gone.
8. **Broader demographic stratification** (`extract_stratified_sample.py`):
   - Stratification variables are now `gender`, `age_quintile`, **`race_group`** (new), and
     `note_length_quintile`.
   - **Algorithm changed from joint-stratum to marginal balancing:**
     - The old greedy pass compared a patient's share within the full joint stratum (sex × age ×
       note length).
     - With a fourth variable, joint strata become near-empty.
     - The new `select_patients_stratified()` accepts a patient only if, for every variable
       independently, the sample's current count of that patient's value divided by (sample size
       + 1) does not exceed the population share of that value.
     - A relaxed second pass fills any remaining slots.
     - The candidate order is `sorted(subject_id)` shuffled with `random.Random(seed)`, so it is
       deterministic.
   - Default sample size is `--n 300`. When the eligible population is smaller, all eligible
     patients are kept and a message is logged.
   - The balance report (`mimic_stratified_sample_report.md`) now covers MIMIC-IV, all four
     marginals, and the note count per category.

#### C. Multiple clinical-note types

9. **Note-source registry** (`shared/mimic_iv.NOTE_SOURCES`). Each entry maps a MIMIC-IV-Note
   schema table to a category and its detail table:

   | Table | Category | Detail table |
   |---|---|---|
   | `discharge` | Discharge summary | `discharge_detail` |
   | `radiology` | Radiology | `radiology_detail` |
   | `nursing` | Nursing | `nursing_detail` |
   | `physician` | Physician | `physician_detail` |
   | `consult` | Consult | `consult_detail` |

   - `discharge` is required (`REQUIRED_NOTE_TABLES`). The others are optional and are skipped
     with a log line when absent. MIMIC-IV-Note itself only ships discharge and radiology.
   - `NOTE_TYPE_LABELS` maps `DS`/`AD`/`RR`/`AR` to "Discharge summary", "Discharge summary
     addendum", "Radiology report", and "Radiology report addendum".
   - This replaces the baseline's MIMIC-III `TARGET_CATEGORIES` filter on `NOTEEVENTS.CATEGORY`.
10. **Structured note storage** (step 1). The baseline's single `clinical_notes` string (notes
    joined with `--- NEXT NOTE ---` and prefixed `[Category: …]`) is replaced by an ordered
    `notes[]` list.
    - Each note has `note_id`, `category`, `note_type`, `note_type_label`, `note_seq`,
      `charttime`, `text`, and `details`. `details` holds the `field_name → field_value` pairs
      from the `*_detail` table, with multiple values joined by `; `.
    - Detail tables are streamed in chunks and filtered to the selected notes.
    - Order: category (discharge → radiology → nursing → physician → consult), then `charttime`,
      `note_seq`, `note_id`.
11. **Category-aware section parser** (`shared/mimic_section_parser.py` rewritten).
    - A `SectionSchema` dataclass holds aliases, boundary-only headers, and case sensitivity.
    - Five schemas are provided:
      - `DISCHARGE_SCHEMA`: the original aliases, unchanged, still case-sensitive.
      - `RADIOLOGY_SCHEMA`: Examination, Indication, Technique, Comparison, Findings, Impression.
      - `NURSING_SCHEMA`: Situation, Background, Assessment, Action, Response, Plan.
      - `PHYSICIAN_SCHEMA`: Chief Complaint, History of Present Illness/HPI, 24 Hour Events,
        Physical Examination, Labs and Radiology, Assessment and Plan / A/P.
      - `CONSULT_SCHEMA`: Reason for Consultation, History of Present Illness, Impression,
        Recommendations.
    - The four new schemas are case-insensitive.
    - `parse_sections(text, category="Discharge summary")` keeps the old call signature.
    - New: `parse_admission_sections(notes)` returns `{category: {section: content}}`. Within a
      category, the first note carrying a section wins.
    - Header matching now tolerates leading spaces or tabs and whitespace before the colon.
    - The legacy `NEXT SUMMARY` pre-split was removed, since it no longer matched any separator.
12. **PDF rendering of every note** (`render_admission_pdfs.py` rewritten).
    - The header gains a **"Clinical Notes Included"** line, e.g. "Discharge summary (1),
      Radiology (1)".
    - Each note gets its own heading: category, note-type label (omitted when identical to the
      category), chart time, and the detail fields `exam_name` / `Modality` / `author` when
      present.
    - Note text is rendered verbatim. The old split on the `--- NEXT NOTE ---` string and the
      "*Clinical note i of n*" italic caption are gone.
    - Header construction moved into `build_header()`, and all dynamic values are HTML-escaped.
    - **Output folder renamed** from the Italian `Ammissioni_PDF_stratified_enriched/` to
      `Admission_PDFs_stratified_enriched/`.
13. **QA generation over the new note types** (`generate_qa_single.py`).
    - `NOTE_SECTION_LOOKUPS` exposes seven non-discharge sections to `section_lookup`:
      - Radiology: Impression, Findings
      - Nursing: Assessment, Plan
      - Physician: Assessment and Plan
      - Consult: Impression, Recommendations
    - Each has a full-section question template and a first-item template (e.g. "What was the
      overall conclusion of my imaging report during my admission beginning on …?").
    - All seven were appended to `SECTION_LOOKUP_PRIORITY`.
    - The six question types and the min-cost-flow balancing are unchanged, so the benchmark
      taxonomy of the paper is preserved.
    - The `Service:` fallback now searches discharge-summary notes only.
    - The disposition question now uses `resolve_disposition()`, i.e. the structured discharge
      location first.

#### D. First-person perspective

14. **Enforced in `generate_qa_single.py`.**
    - New `FIRST_PERSON_RE` (`I|me|my|mine|myself`).
    - `add()` raises `ValueError` on a non-first-person template, and `validate_items()` raises
      `RuntimeError` on any non-first-person item.
    - The admission-reference fallback changed from "this admission" to "my admission".
    - The Chief Complaint template changed from "…when I was admitted on {date or 'this
      admission'}?" to "…at the start of my admission beginning on {date}?".
    - The Brief Hospital Course first-item template changed from "How does the hospital course
      summary begin for …" to "How does the summary of my hospital course begin for …".
    - The unused `admission_ref_by_discharge` variable was deleted.
15. **Enforced in `generate_qa_multidoc.py`.**
    - `FIRST_PERSON_RE` (also `we|us|our`) plus a new patient-ID-leak check, both in
      `validate_final()`.
    - Cross-patient questions previously named "Patient {subject_id} … and patient {subject_id}".
      They are now framed as a caregiver: "I keep my family member's hospital records together
      with mine. We were both hospitalized with the principal diagnosis "…": my stay began on …
      and theirs began on …. What was the … for each of us…?"
    - Cross-patient answers read "My hospitalization beginning … ; my family member's
      hospitalization beginning …".
    - Cross-patient summaries read "…was the same for both of us" / "…differed between us" /
      "My admission had more recorded diagnoses than my family member's".
    - Same-patient summaries are unchanged ("My recorded … was the same/differed").
    - Trajectory answers now start with a capital letter ("My hospitalization beginning …").
    - The ICU field label changed from "ICU-stay status" to "intensive care (ICU) status".
16. **Unrealistic self-referential questions removed.**
    - The single-document "Did I experience in-hospital mortality…?" question was deleted.
    - `mortality` was removed from `PAIR_FIELDS`, `CROSS_FIELDS`, and `FIELD_SPECS` in the
      multi-document generator.
    - The disposition question is skipped when the resolved disposition is "Died in hospital" or
      "Not recorded".
    - The "In-Hospital Mortality" field **remains in the PDF header** as a document fact.
17. **MIMIC-IV de-identification handling** (`generate_qa_single.py`).
    - `DEID_SURROGATE_RE` now matches MIMIC-IV's `___` placeholders as well as MIMIC-III's
      `[**`.
    - `_neutralize_deid()` replaces `___` with "(removed)".

#### E. Multi-document pairing choices (`generate_qa_multidoc.py`)

18. **Consecutive pairs only.** `make_same_patient_candidates()` now pairs admission *i* with *i+1*
    in chronological order, instead of `itertools.combinations(admissions, 2)`.
19. **Per-patient caps.** New constants:
    - `MAX_PAIRS_PER_PATIENT = 4`
    - `MAX_TRAJECTORIES_PER_PATIENT = 2`
    - `MAX_CROSS_ITEMS_PER_PATIENT_PAIR = 1`

    Supporting code:
    - `PairCandidate` and `TrajectoryCandidate` gained a `priority` field and a `cap_key`
      property: the patient, or the sorted patient pair for cross-patient items.
    - New `spread_priorities()` ranks a patient's windows so that the first `cap` ranks are spread
      evenly across the timeline.
    - New `ordered_source_sets()` orders source sets by (priority, seeded `stable_rank`).
    - `build_pair_items()` takes a `max_items_per_group` argument, and both builders skip a group
      once its cap is reached.
    - A preferred window that fails grounding is replaced by the next-ranked window of the same
      patient.
    - `validate_final()` asserts every cap.
    - The run log prints the caps and the pre-cap candidate counts.
20. **Effect on the reference run** (300 patients, see "Reference run"), eligible source sets →
    final items after grounding and caps:
    - same-patient pairs: 385 → 289
    - trajectories: 256 → 130
    - cross-patient items: 490 → 343
    - total: 1,131 → 762

    All 129 multi-admission patients are still covered by same-patient items, and per-field counts
    are within one item of each other. Rationale: see "Why multi-document items are capped" above.

#### F. Other script updates

21. **`split_qa_multidoc_by_patient.py`:** paths now come from `shared.pipeline_paths`; the logic
    is unchanged.
22. **`generate_qa_single.py` / `generate_qa_multidoc.py`:** source JSON, PDF folder, and output
    paths now come from `shared.pipeline_paths`. Hand-built PDF filenames were replaced by
    `admission_pdf_name()`, and the start-up log prints the work directory instead of the script
    directory.
23. **Language:** every identifier, docstring, and comment in `rebuild_anamnesis/` is in American
    English. The Italian comments present at the baseline ("MODIFICATO: …", "Nota: …",
    "Categorie di note cliniche incluse…") were removed or rewritten.
24. **Unchanged files:** `shared/kotlin_mirror.py` (the Android chunking port) and
    `shared/pdf_paging.py`, so gold-chunk boundaries remain identical to the on-device app.
25. **Repository hygiene:** new `rebuild_anamnesis/.gitignore` ignores every pipeline output file
    name, `*.pdf`, `*.jsonl`, and `__pycache__/`. The previously tracked `shared/__pycache__/*.pyc`
    files were removed from version control.
26. **Scope:** only files inside `rebuild_anamnesis/` were modified or added. The Android
    application and all other project files were left untouched.
