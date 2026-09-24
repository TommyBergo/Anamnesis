# `rebuild_anamnesis/` — corpus & QA-dataset reproducibility scripts

This folder holds the verified scripts that built the Anamnesis paper's clinical-record corpus
(138 admission PDFs) and its rule-based QA benchmarks, based on a historical, read-only research
repo containing the original MIMIC-III extraction/rendering work (last checked 2026-09-16 - nothing
there was changed).
The five scripts below are working copies, adapted to run standalone from this flat folder (see
"What changed from the originals" at the bottom); `shared/` holds the small helper modules they
import.

`depression_rag\modello\patients_clinical_records\` and `rag_benchmark\QA\` contain many more `.py`
files than what's reproduced here - draft/superseded generators, an abandoned experiment, and an
unrelated multidoc-QA candidate-mining pipeline. Every one of them was read (not just named) before
being ruled out; see "Rejected candidates" below for exactly which ones and why.

## Layout

```text
rebuild_anamnesis/
├── extract_stratified_sample.py     step 1
├── render_admission_pdfs.py         step 2
├── generate_qa_single.py            step 3
├── generate_qa_multidoc.py          step 4
├── split_qa_multidoc_by_patient.py  step 5 (new - see below)
└── shared/
    ├── mimic_common.py              date/age helpers (steps 1, 2, 3)
    ├── mimic_section_parser.py      discharge-summary section splitter (steps 2, 3)
    ├── pdf_paging.py                per-page PDF text + offset tracking (steps 3, 4)
    └── kotlin_mirror.py             Python port of the app's chunking logic (used by pdf_paging.py)
```

## Pipeline order

Run in this order. Each step's output is the next step's input; `mimic_stratified_sample.json` and
the raw MIMIC-III CSVs are not included here (see "What is *not* included" below) - place your own
copy of `mimic_stratified_sample.json` directly in this folder before running steps 2-5.

1. **`extract_stratified_sample.py`** - raw MIMIC-III CSVs -> `mimic_stratified_sample.json`
   (100 patients / 138 admissions) + `mimic_stratified_sample_report.md`.
2. **`render_admission_pdfs.py`** - `mimic_stratified_sample.json` -> 138 PDFs in
   `Ammissioni_PDF_stratified_enriched/`.
3. **`generate_qa_single.py`** - the JSON + those PDFs -> `single_dataset_rulebased.jsonl`
   (two rule-based QA pairs per admission).
4. **`generate_qa_multidoc.py`** - the JSON + those PDFs -> `multidoc_rulebased.jsonl` (three
   task families combined: same-patient x2, same-patient x3-trajectory, cross-patient).
5. **`split_qa_multidoc_by_patient.py`** - reads step 4's output, writes
   `multidoc_dataset_rulebased_main_same_patient.jsonl` (what's actually shipped to the app) and
   `multidoc_dataset_rulebased_cross_patient.jsonl` (not currently used, produced for completeness).

## Step detail

**`extract_stratified_sample.py`** reads raw MIMIC-III CSVs (`PATIENTS`, `ADMISSIONS`,
`DIAGNOSES_ICD`, `ICUSTAYS`, `SERVICES`, `NOTEEVENTS` - PhysioNet-credentialed access required) from
a `--base-dir` argument (defaults to `/gringotts/datasets/MIMIC-III`). Selects 100 patients via
nested stratification (outer: `has_mental_health_diagnosis`, forced proportional to the true
population share; inner: gender x age-quintile x note-length-quintile, greedy-balanced), then pulls
every admission for each selected patient (138 total).

**`render_admission_pdfs.py`** renders each admission to a one-PDF-per-admission discharge summary
(reportlab), with an 8-field enriched English header (Age at Admission, Hospital Service, Length of
Stay, Discharge Disposition - parsed from the note body - plus Mental Health Diagnosis / ICU Stay /
In-Hospital Mortality / Number of Diagnoses read directly from the JSON), followed by the verbatim
note text. Output filenames: `Patient_{subject_id}_Admission_{hadm_id}.pdf`.

**`generate_qa_single.py`** is a deterministic (non-LLM) QA generator: exactly two question types
per admission, macro-types balanced as evenly as corpus availability allows, concrete QA pairs
preferring non-redundant answers and diverse evidence.

**`generate_qa_multidoc.py`** generates three task families: `same_patient_two_admissions` (exactly
2 admissions, same patient), `same_patient_three_admission_trajectory` (exactly 3 consecutive
admissions, same patient), and `cross_patient_two_admissions` (2 admissions from 2 different
patients sharing the same principal diagnosis) - 116 items total (65 + 16 + 35).

**`split_qa_multidoc_by_patient.py` was not part of the original depression_rag pipeline** - no
script producing the same-patient/cross-patient split existed anywhere there (searched thoroughly:
by filename pattern, by re-reading `generate_multidoc_rulebased.py` in full for a hidden CLI flag -
there is none - and by checking git history, which is empty). It must have been a manual/interactive
step at the time. The filter criterion was reverse-engineered and verified empirically (counting
every item's `comparison_type`): a clean, lossless partition - all 81 same-patient items in, all 35
cross-patient items out, nothing dropped or duplicated. This script reproduces exactly that split.

## Rejected candidates (read, not just named, before ruling out)

| Script | Why rejected |
|---|---|
| `extract_mimic.py` | Earliest draft: plain `random.sample()`, no stratification. Produces the older 118-admission corpus (`mimic_100_discharge_summaries.json`), not the current one. |
| `json_to_pdf.py` | First PDF renderer - non-stratified source, basic Italian header, only 5 fields. |
| `json_to_pdf_stratified.py` | Stratified source, but still the old basic/Italian rendering (no enrichment). Writes to `Ammissioni_PDF_stratified/`, not `_enriched`. |
| `json_to_pdf_enriched.py` | Enrichment added, but sourced from the *old* non-stratified JSON, where `has_mental_health_diagnosis`/`had_icu_stay`/`hospital_expire_flag` are `None` for all 118 admissions (broken CSV joins) - this is exactly why the stratified+enriched script exists. |
| `json_to_pdf_stratified_enriched_json_only.py` | Near-duplicate of the winning script, same output folder/source JSON, but requires fields (`age_at_admission`, `length_of_stay`, `discharge_disposition`) that don't exist in the real `mimic_stratified_sample.json` - would crash immediately if run. Also the single latest-modified file in the directory, three days after the real corpus was already generated and shipped - a non-functional draft. |
| `generate_dataset_single_rulebased_v11.py` / `_v12.py` / `_v13.py` | Earlier same-day drafts of `generate_qa_single.py`, superseded by the version with no suffix. |
| `generate_multidoc_rulebased_v2.py` | Earlier draft of `generate_qa_multidoc.py` (2026-08-28, three days before the final version). |
| ~15 `dump_*`/`extract_*`/`build_batch*_items.py` files | A separate, older multidoc-QA *candidate-mining* pipeline feeding `new_multidoc_items*.jsonl` - out of scope; not the current rulebased generator. |

## Evidence this is the right pipeline, not a guess

- `ENRICHED_CORPUS_METHODOLOGY.md` (in `depression_rag`) independently documents this exact
  two-script corpus pipeline by name, including why the enriched-JSON-only variant was abandoned.
- Timestamp/byte correlation: the original `json_to_pdf_stratified_enriched.py` was last edited
  2026-08-28 10:23:40; all 138 output PDFs cluster 10:34:07-10:34:46 the same day. Spot-checked one
  PDF: identical file size (6793 bytes) and identical timestamp between the `depression_rag` source
  and the Android app's shipped copy (pre-rename) - proof the app's corpus is this script's direct,
  unmodified output.
- `generate_qa_single.py`'s original output was byte-for-byte identical (923,851 bytes) to the app's
  shipped `single_dataset_rulebased.jsonl`, before the Paziente/Ricovero rename touched it.

## What changed from the originals

These are working copies, not byte-identical snapshots - three kinds of changes were made, all
logic-preserving (verified by comparing every non-comment, non-string source token before and after
each edit - identical in every file):

1. **Flattened layout + fixed imports.** The originals were split across `patients_clinical_records/`
   and `rag_benchmark/QA/` in `depression_rag`, with `sys.path.insert(...)` calls and a duplicated
   copy of `mimic_section_parser.py` to bridge the two directories. Here everything is flat, with one
   `shared/` folder and plain `from shared.x import y` imports - no path hacking, one copy of each
   helper. Verified by importing every script in isolation (a fresh Python process per file) and
   confirming every import resolves.
2. **English filenames produced directly.** The originals wrote `Paziente_{id}_Ricovero_{hadm}.pdf`
   and read/wrote several dataset files under their pre-shortening names
   (`synthetic_dataset_single_rulebased.jsonl` instead of `single_dataset_rulebased.jsonl`, and two
   more `Paziente_`/`Ricovero_` references inside `generate_qa_multidoc.py` that were missed by the
   rename applied to the live app repo, since this script was still sitting untouched in
   `depression_rag` at the time). All of these now produce the same English names the Android app
   actually ships, closing what was previously a documented "rename it yourself" gap.
3. **Comments trimmed to one sentence per file/class.** Every inline `#` comment and multi-line
   docstring was stripped; each file now carries exactly one explanatory sentence at the top
   (module docstring), and each class carries one sentence describing what it represents. Function
   bodies are unchanged. This was done mechanically (Python's own `tokenize`/`ast` modules identify
   comment and docstring spans precisely - not a regex pass that could mangle a string literal
   containing `#`), and verified by diffing every file's non-comment, non-string token stream before
   and after: identical in all nine files.

## What is *not* included here

`mimic_stratified_sample.json` and the raw MIMIC-III CSVs are not copied into this folder. The CSVs
require PhysioNet credentialing and cannot be freely redistributed; the JSON derived from them
remains at its original location,
`depression_rag\modello\patients_clinical_records\mimic_stratified_sample.json`, for anyone who
already has the appropriate access. Running steps 2-5 from scratch requires that file in place
directly inside this folder (i.e. run step 1 first, or supply your own copy of that exact JSON).
`pandas` is required for step 1 and is not bundled - install it separately.
