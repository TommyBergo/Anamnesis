"""Generates two deterministic, rule-based QA pairs per admission from the enriched MIMIC-III corpus."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
SOURCE_JSON = BASE_DIR / "mimic_stratified_sample.json"
PDF_DIR = BASE_DIR / "Ammissioni_PDF_stratified_enriched"
OUTPUT_PATH = BASE_DIR / "single_dataset_rulebased.jsonl"

from shared.mimic_section_parser import parse_sections
from shared.pdf_paging import extract_paged_document, find_pages_for_text, PagedDocument
from shared.kotlin_mirror import Chunk, clean_text, create_clinical_chunks, _WHITESPACE_RUN

QUESTIONS_PER_ADMISSION = 2

STRICT_CORPUS_COMPLETENESS = True

BENCHMARK_QUESTION_TYPES = (
    "header_fact",
    "header_fact_enriched",
    "section_lookup",
    "specific_detail",
    "negation_check",
    "multi_admission_distractor",
)

TYPE_TARGET_WEIGHTS = {
    "section_lookup": 0.30,
    "specific_detail": 0.28,
    "header_fact_enriched": 0.14,
    "header_fact": 0.12,
    "multi_admission_distractor": 0.10,
    "negation_check": 0.06,
}

DEID_SURROGATE_RE = re.compile(r"\[\*\*")
REJECT_DEID_SURROGATE_QA = True

MAX_YES_NO_SHARE = 0.12
BALANCE_NEGATION_POLARITY = True

ASSERT_ANSWER_INSIDE_SECTION = True

REQUIRE_DISTINCT_EVIDENCE_PER_ADMISSION = True
PREFER_DISTINCT_PAGES_PER_ADMISSION = True

SUPPRESS_DUPLICATE_DIAGNOSIS_QUESTION = True

INTERNAL_FIELDS = (
    "_subkey",
    "_yes_no",
    "_evidence",
)

DIVERSIFY_SECTION_LOOKUP = True

SECTION_LOOKUP_PRIORITY = (
    "Chief Complaint",
    "Discharge Condition",
    "Medications on Admission",
    "Discharge Medications",
    "Past Medical History",
    "Brief Hospital Course",
)

SECTION_LOOKUP_RANK = {
    section_name: rank
    for rank, section_name
    in enumerate(SECTION_LOOKUP_PRIORITY)
}

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 100

MIMIC_90_PLUS_SENTINEL_AGE = 90
MIMIC_90_PLUS_RAW_THRESHOLD = 150

SERVICE_RE = re.compile(
    r"^Service:[ \t]*(\S.*?)\s*$",
    re.MULTILINE,
)

HPI_PROVENANCE_PREFIXES = (
    "history obtained",
    "history was obtained",
    "information obtained",
    "information was obtained",
    "source of history",
    "source of information",
    "history provided",
    "history is provided",
    "history limited",
    "history was limited",
    "unable to obtain history",
    "the history was obtained",
    "per medical records",
    "per records",
    "per chart",
)

NO_KNOWN_ALLERGY_PATTERNS = (
    r"\bno known allergies\b",
    r"\bno known drug allergies\b",
    r"\bnkda\b",
    r"\bnka\b",
    r"^\s*none\s*$",
    r"\bno allergies\b",
)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _has_deid_surrogate(text: Optional[str]) -> bool:
    if not text:
        return False

    return bool(DEID_SURROGATE_RE.search(text))

SENTENCE_ABBREVIATIONS = {
    "dr", "drs", "mr", "mrs", "ms", "mx", "prof", "rev", "sr", "jr",
    "st", "mt", "no", "vs", "approx", "etc", "eg", "ie", "cf",
    "am", "pm", "hrs", "hr", "min", "sec", "wk", "wks", "mo", "mos",
    "yr", "yrs", "y.o", "yo", "q", "qd", "qhs", "bid", "tid", "qid",
    "prn", "po", "pr", "iv", "im", "sq", "subq", "gtt", "tab", "tabs",
    "cap", "caps", "mg", "mcg", "kg", "gm", "ml", "meq", "mmol", "cc",
    "u", "units", "inj", "sol", "susp", "ext", "rel", "d/c", "dx", "hx",
    "tx", "rx", "sx", "fx", "pt", "pts", "wt", "ht", "temp", "bp", "hr",
    "o2", "co2", "h2o", "b.i.d", "t.i.d", "q.d", "p.o", "i.v", "n.p.o",
    "npo", "a.m", "p.m", "e.g", "i.e", "approx", "dept", "hosp", "univ",
    "inc", "ltd", "co", "corp", "fig", "vol", "ref", "max", "min",
    "sat", "sats", "eval", "cont", "adm", "disch",
}

_ABBREV_TAIL_RE = re.compile(r"([A-Za-z][A-Za-z\.]*)\.$")
_NUMBER_TAIL_RE = re.compile(r"\d\.$")

def _is_false_sentence_boundary(
    left: str,
    right: str,
) -> bool:
    left = left.rstrip()

    if not left.endswith((".",)):
        return False

    if _NUMBER_TAIL_RE.search(left) and re.match(r"^\d", right):
        return True

    if re.search(r"(?:^|\s)[A-Z]\.$", left):
        return True

    match = _ABBREV_TAIL_RE.search(left)

    if match:
        token = match.group(1).rstrip(".").casefold()

        if token in SENTENCE_ABBREVIATIONS:
            return True

        if "." in token and len(token.replace(".", "")) <= 5:
            return True

    return False

def _sentence_candidates(text: str) -> list[str]:
    flat = _normalize_ws(text)

    if not flat:
        return []

    sentences: list[str] = []
    buffer = ""

    for piece in re.split(
        r"(?<=[.!?])\s+(?=(?:\[\*\*|[A-Za-z0-9]))",
        flat,
    ):
        piece = piece.strip()

        if not piece:
            continue

        if buffer and _is_false_sentence_boundary(buffer, piece):
            buffer = f"{buffer} {piece}"
            continue

        if buffer:
            sentences.append(buffer)

        buffer = piece

    if buffer:
        sentences.append(buffer)

    return sentences

_DEID_TOKEN_RE = re.compile(r"\[\*\*(.*?)\*\*\]", re.DOTALL)

def _deid_placeholder(inner: str) -> str:
    lowered = inner.casefold()

    if re.search(r"\d{4}-\d{1,2}(-\d{1,2})?", inner) or "date" in lowered:
        return "(date removed)"

    if "telephone" in lowered or "fax" in lowered or "pager" in lowered:
        return "(contact number removed)"

    if "hospital" in lowered or "clinic" in lowered or "ward" in lowered:
        return "(hospital name removed)"

    if "age over 90" in lowered or "age" in lowered:
        return "(age removed)"

    if "name" in lowered or "initial" in lowered or "doctor" in lowered:
        return "(name removed)"

    if (
        "location" in lowered
        or "state" in lowered
        or "country" in lowered
        or "address" in lowered
        or "university" in lowered
    ):
        return "(location removed)"

    if "number" in lowered or "numeric" in lowered or "id" in lowered:
        return "(identifier removed)"

    return "(removed)"

def _neutralize_deid(text: str) -> str:
    if not text:
        return ""

    neutral = _DEID_TOKEN_RE.sub(
        lambda m: _deid_placeholder(m.group(1)),
        text,
    )

    neutral = re.sub(r"\(\s*", "(", neutral)
    neutral = re.sub(r"\s+([,.;:])", r"\1", neutral)

    return _normalize_ws(neutral)

def _is_clinically_informative(
    text: str,
    min_alpha: int = 25,
    min_words: int = 5,
) -> bool:
    stripped = re.sub(r"\((?:[a-z ]+?) removed\)", " ", text or "")
    stripped = re.sub(r"\(removed\)", " ", stripped)
    stripped = _normalize_ws(stripped)

    alpha = sum(1 for ch in stripped if ch.isalpha())
    words = [w for w in stripped.split() if any(c.isalpha() for c in w)]

    return alpha >= min_alpha and len(words) >= min_words

def _first_complete_sentence(
    text: str,
    min_len: int = 4,
    max_len: int = 450,
) -> Optional[str]:
    flat = _normalize_ws(text)

    if not flat:
        return None

    sentences = _sentence_candidates(flat)

    if sentences:
        first = _normalize_ws(sentences[0])

        if len(first) < min_len:
            return None

        if len(first) > max_len:
            return None

        return first

    if min_len <= len(flat) <= max_len:
        return flat

    return None

def _raw_nonempty_lines(
    text: str,
) -> list[str]:
    return [
        _normalize_ws(line)
        for line in text.splitlines()
        if _normalize_ws(line)
    ]

_NUMBERED_ITEM_RE = re.compile(
    r"^\s*\d+[\.\)]\s+"
)

def _first_numbered_item(
    text: str,
    max_len: int = 450,
) -> Optional[str]:
    lines = _raw_nonempty_lines(text)

    if not lines:
        return None

    start_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if _NUMBERED_ITEM_RE.match(line)
        ),
        None,
    )

    if start_idx is None:
        return None

    item_lines = []

    for idx in range(start_idx, len(lines)):
        line = lines[idx]

        if (
            idx > start_idx
            and _NUMBERED_ITEM_RE.match(line)
        ):
            break

        item_lines.append(line)

    item = _normalize_ws(
        " ".join(item_lines)
    )

    inline_next = re.search(
        r"\s+\d+[\.\)]\s+",
        item[2:],
    )

    if inline_next:
        cut = 2 + inline_next.start()
        item = item[:cut].strip()

    if 4 <= len(item) <= max_len:
        return item

    return None

def _first_hash_item(
    text: str,
    max_len: int = 450,
) -> Optional[str]:
    lines = _raw_nonempty_lines(text)

    if not lines:
        return None

    start_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.lstrip().startswith("#")
        ),
        None,
    )

    if start_idx is None:
        return None

    item_lines = []

    for idx in range(start_idx, len(lines)):
        line = lines[idx]

        if (
            idx > start_idx
            and line.lstrip().startswith("#")
        ):
            break

        item_lines.append(line)

    item = _normalize_ws(
        " ".join(item_lines)
    )

    if 4 <= len(item) <= max_len:
        return item

    return None

_DASH_ITEM_RE = re.compile(
    r"^\s*-\s*\S"
)

def _first_dash_item(
    text: str,
    max_len: int = 450,
) -> Optional[str]:
    lines = _raw_nonempty_lines(text)

    bullet_indexes = [
        idx
        for idx, line in enumerate(lines)
        if _DASH_ITEM_RE.match(line)
    ]

    if len(bullet_indexes) < 2:
        return None

    start_idx = bullet_indexes[0]
    item_lines: list[str] = []

    for idx in range(start_idx, len(lines)):
        line = lines[idx]

        if (
            idx > start_idx
            and _DASH_ITEM_RE.match(line)
        ):
            break

        item_lines.append(line)

    item = _normalize_ws(
        " ".join(item_lines)
    )

    if 4 <= len(item) <= max_len:
        return item

    return None

def _first_medication_entry(
    text: str,
    max_len: int = 450,
) -> Optional[str]:
    return _first_numbered_item(
        text,
        max_len=max_len,
    )

def _first_pmh_item(
    text: str,
    max_len: int = 450,
) -> Optional[str]:
    numbered = _first_numbered_item(
        text,
        max_len=max_len,
    )

    if numbered:
        return numbered

    hash_item = _first_hash_item(
        text,
        max_len=max_len,
    )

    if hash_item:
        return hash_item

    dash_item = _first_dash_item(
        text,
        max_len=max_len,
    )

    if dash_item:
        return dash_item

    return _first_complete_sentence(
        text,
        max_len=max_len,
    )

def _first_complete_unit(
    text: str,
    min_len: int = 4,
    max_len: int = 320,
) -> Optional[str]:
    return _first_complete_sentence(
        text,
        min_len=min_len,
        max_len=max_len,
    )

def _extract_section_reference(
    section_name: str,
    content: str,
    max_full_len: int = 320,
    max_unit_len: int = 450,
) -> tuple[Optional[str], str]:
    flat = _normalize_ws(content)

    if not flat:
        return None, "none"

    if len(flat) <= max_full_len:
        return flat, "full"

    if section_name in {
        "Medications on Admission",
        "Discharge Medications",
    }:
        entry = _first_medication_entry(
            content,
            max_len=max_unit_len,
        )

        if entry:
            return entry, "medication"

        if len(flat) <= max_unit_len:
            return flat, "full"

        return None, "none"

    if section_name == "Past Medical History":
        item = _first_pmh_item(
            content,
            max_len=max_unit_len,
        )

        if item:
            return item, "pmh_item"

        if len(flat) <= max_unit_len:
            return flat, "full"

        return None, "none"

    statement = _first_complete_sentence(
        content,
        max_len=max_unit_len,
    )

    if statement:
        return statement, "statement"

    if len(flat) <= max_unit_len:
        return flat, "full"

    return None, "none"

def _strip_leading_hpi_provenance(
    text: str,
) -> Optional[str]:
    flat = _normalize_ws(text)

    if not flat:
        return None

    flat = re.sub(
        r"^(?:HPI|History of Present Illness)\s*:\s*",
        "",
        flat,
        flags=re.IGNORECASE,
    ).strip()

    while flat:
        lowered = flat.casefold()

        if not any(
            lowered.startswith(prefix)
            for prefix in HPI_PROVENANCE_PREFIXES
        ):
            break

        match = re.match(
            r"^.*?[.!?](?:\s+|$)",
            flat,
            flags=re.DOTALL,
        )

        if not match:
            return None

        flat = flat[match.end():].strip()

        flat = re.sub(
            r"^(?:HPI|History of Present Illness)\s*:\s*",
            "",
            flat,
            flags=re.IGNORECASE,
        ).strip()

    return flat or None

def _first_clinical_hpi_statement(
    text: str,
    max_len: int = 450,
) -> Optional[tuple[str, str]]:
    remaining = _strip_leading_hpi_provenance(
        text
    )

    if not remaining:
        return None

    remaining = re.sub(
        r"\s+\.\s+",
        " ",
        remaining,
    ).strip()

    sentences = _sentence_candidates(remaining)

    if not sentences:
        single = _first_complete_sentence(
            remaining,
            min_len=12,
            max_len=max_len,
        )
        sentences = [single] if single else []

    for sentence in sentences:
        evidence = _normalize_ws(sentence)

        if len(evidence) < 12 or len(evidence) > max_len:
            continue

        answer = (
            _neutralize_deid(evidence)
            if _has_deid_surrogate(evidence)
            else evidence
        )

        if not answer or len(answer) < 12:
            continue

        if not _is_clinically_informative(answer):
            continue

        return answer, evidence

    return None

def _contains_no_known_allergy_statement(text: str) -> bool:
    flat = _normalize_ws(text)

    if not flat:
        return False

    return any(
        re.search(
            pattern,
            flat,
            flags=re.IGNORECASE,
        ) is not None
        for pattern in NO_KNOWN_ALLERGY_PATTERNS
    )

def _extract_allergy_negation_reference(
    text: str,
    max_len: int = 320,
) -> Optional[tuple[str, str]]:
    flat = _normalize_ws(text)

    if not flat:
        return None

    has_no_known = _contains_no_known_allergy_statement(flat)

    if has_no_known:
        for sentence in _sentence_candidates(text):
            sentence_norm = _normalize_ws(sentence)
            if (
                len(sentence_norm) <= max_len
                and _contains_no_known_allergy_statement(sentence_norm)
            ):
                return "Yes", sentence_norm

        if len(flat) <= max_len:
            return "Yes", flat

        return None

    if len(flat) <= max_len:
        return "No", flat

    return None

def _page_text(
    paged_doc: PagedDocument,
    page_number: int,
) -> str:
    if not (1 <= page_number <= len(paged_doc.page_ranges)):
        raise IndexError(f"Invalid page number: {page_number}")

    start, end = paged_doc.page_ranges[page_number - 1]
    return paged_doc.clean_text[start:end]

def _contains_normalized_text(
    container_text: str,
    evidence_text: str,
) -> bool:
    evidence_norm = _normalize_ws(evidence_text).casefold()

    if not evidence_norm:
        return False

    return evidence_norm in _normalize_ws(container_text).casefold()

def _page_contains_text(
    paged_doc: PagedDocument,
    page_number: int,
    evidence_text: str,
) -> bool:
    if not (1 <= page_number <= len(paged_doc.page_ranges)):
        return False

    return _contains_normalized_text(
        _page_text(paged_doc, page_number),
        evidence_text,
    )

def _chunk_clean_offset(
    paged_doc: PagedDocument,
    raw_offset: int,
) -> int:
    raw_text = paged_doc.clean_text
    raw_offset = max(0, min(raw_offset, len(raw_text)))

    collapsed_full = _WHITESPACE_RUN.sub(" ", raw_text)
    collapsed_prefix = _WHITESPACE_RUN.sub(" ", raw_text[:raw_offset])

    leading_trim = (
        len(collapsed_full)
        - len(collapsed_full.lstrip())
    )

    return max(
        0,
        min(
            len(clean_text(raw_text)),
            len(collapsed_prefix) - leading_trim,
        ),
    )

def _find_gold_chunk(
    paged_doc: PagedDocument,
    page_number: int,
    evidence_text: str,
    chunks: Optional[list[Chunk]] = None,
) -> Optional[str]:
    if not _page_contains_text(
        paged_doc,
        page_number,
        evidence_text,
    ):
        return None

    raw_page_start, raw_page_end = paged_doc.page_ranges[page_number - 1]
    page_start = _chunk_clean_offset(
        paged_doc,
        raw_page_start,
    )
    page_end = _chunk_clean_offset(
        paged_doc,
        raw_page_end,
    )

    if chunks is None:
        chunks = create_clinical_chunks(
            paged_doc.clean_text,
        )

    exact = [
        chunk
        for chunk in chunks
        if (
            chunk.start < page_end
            and chunk.end > page_start
            and _contains_normalized_text(
                chunk.text,
                evidence_text,
            )
        )
    ]

    if not exact:
        return None

    def sort_key(chunk: Chunk) -> tuple:
        overlap = max(
            0,
            min(chunk.end, page_end)
            - max(chunk.start, page_start),
        )
        return (-overlap, len(chunk.text), chunk.start)

    exact.sort(key=sort_key)
    return exact[0].text

def _extract_header_value(
    paged_doc: PagedDocument,
    label: str,
) -> Optional[str]:
    if not paged_doc.page_ranges:
        return None

    page_one = _page_text(paged_doc, 1)
    match = re.search(
        rf"^{re.escape(label)}:[ \t]*(.+?)\s*$",
        page_one,
        flags=re.MULTILINE,
    )

    if not match:
        return None

    value = _normalize_ws(match.group(1))
    return value or None

def compute_age(
    dob: str,
    admittime: str,
) -> Optional[int]:
    try:
        dob_dt = datetime.strptime(
            dob.split(" ")[0],
            "%Y-%m-%d",
        )
        adm_dt = datetime.strptime(
            admittime.split(" ")[0],
            "%Y-%m-%d",
        )
    except (ValueError, IndexError):
        return None

    raw_year_gap = adm_dt.year - dob_dt.year

    if raw_year_gap >= MIMIC_90_PLUS_RAW_THRESHOLD:
        return MIMIC_90_PLUS_SENTINEL_AGE

    age = raw_year_gap - (
        (adm_dt.month, adm_dt.day)
        < (dob_dt.month, dob_dt.day)
    )

    return age if age >= 0 else None

def compute_los_days(
    admittime: str,
    dischtime: str,
) -> Optional[int]:
    try:
        t_in = datetime.strptime(
            admittime.split(" ")[0],
            "%Y-%m-%d",
        )
        t_out = datetime.strptime(
            dischtime.split(" ")[0],
            "%Y-%m-%d",
        )
    except (ValueError, IndexError):
        return None

    days = (t_out - t_in).days
    return days if days >= 0 else None

def build_questions_for_admission(
    patient_info: dict,
    admission: dict,
    sibling_admissions: list[dict],
    document_name: str,
    paged_doc: PagedDocument,
) -> list[dict]:

    pid = patient_info["subject_id"]
    hadm = admission["hadm_id"]
    dob = patient_info.get(
        "dob",
        "",
    ).split(" ")[0]

    adm_date_raw = admission.get(
        "admittime",
        "",
    ).split(" ")[0]

    admission_ref = (
        f"my admission beginning on {adm_date_raw}"
        if adm_date_raw
        else "this admission"
    )

    disch_date_raw = admission.get(
        "dischtime",
        "",
    ).split(" ")[0]

    admission_ref_by_discharge = (
        f"my admission that ended on {disch_date_raw}"
        if disch_date_raw
        else "this admission"
    )

    # MODIFICATO: Legge la nuova chiave multi-nota (clinical_notes) con fallback su discharge_summary
    sections = parse_sections(
        admission.get("clinical_notes", admission.get("discharge_summary", ""))
    )

    items: list[dict] = []
    counter = [0]

    android_chunks = create_clinical_chunks(
        paged_doc.clean_text
    )

    def add(
        question: str,
        answer: str,
        section: str,
        qtype: str,
        page_override: Optional[int] = None,
        subkey: Optional[str] = None,
        evidence_text: Optional[str] = None,
        section_text: Optional[str] = None,
    ) -> None:
        counter[0] += 1

        if REJECT_DEID_SURROGATE_QA and (
            _has_deid_surrogate(question)
            or _has_deid_surrogate(answer)
        ):
            return

        lookup_text = (
            evidence_text
            if evidence_text is not None
            else answer
        )

        if (
            ASSERT_ANSWER_INSIDE_SECTION
            and section_text
            and not _contains_normalized_text(
                section_text,
                lookup_text,
            )
        ):
            return

        if page_override is not None:
            page = page_override
        else:
            page, _candidates = find_pages_for_text(
                paged_doc,
                lookup_text,
            )

        if page is None:
            return

        gold_chunk = _find_gold_chunk(
            paged_doc,
            page,
            lookup_text,
            chunks=android_chunks,
        )

        if gold_chunk is None:
            return

        items.append(
            {
                "id": f"P{pid}_H{hadm}_Q{counter[0]}",
                "patient_id": str(pid),
                "document_name": document_name,
                "page_number": page,
                "ground_truth_context": gold_chunk,
                "ground_truth_page_context": _page_text(
                    paged_doc,
                    page,
                ),
                "question": question,
                "ground_truth_answer": answer,
                "section": section,
                "question_type": qtype,
                "_yes_no": _normalize_ws(answer).casefold() in ("yes", "no"),
                "_evidence": lookup_text,
                "_subkey": (
                    subkey
                    if subkey is not None
                    else section
                ),
            }
        )

    diagnosis = admission.get(
        "diagnosis",
        "",
    ).strip()

    if diagnosis and diagnosis not in (
        "N/D",
        "N/A",
        "",
    ):
        add(
            (
                f"What was the main problem I was admitted for "
                f"in {admission_ref}?"
            ),
            diagnosis,
            "header",
            "header_fact",
            page_override=1,
            subkey="diagnosis",
            evidence_text=f"Principal Diagnosis: {diagnosis}",
        )

    adm_type = admission.get(
        "admission_type",
        "",
    ).strip()

    if adm_type:
        add(
            (
                f"Was I admitted as an emergency, as a planned "
                f"admission, or in another way in {admission_ref}?"
            ),
            adm_type,
            "header",
            "header_fact",
            page_override=1,
            subkey="admission_type",
            evidence_text=f"Type: {adm_type}",
        )

    gender = patient_info.get(
        "gender",
        "",
    ).strip()

    if gender:
        add(
            f"Am I recorded as male or female in {admission_ref}?",
            gender,
            "header",
            "header_fact",
            page_override=1,
            subkey="sex",
            evidence_text=f"Sex: {gender}",
        )

    adm_date = admission.get(
        "admittime",
        "",
    ).split(" ")[0]

    if adm_date:
        add(
            (
                f"When was I admitted for the hospital stay that "
                f"ended on {disch_date_raw or 'discharge'}?"
            ),
            adm_date,
            "header",
            "header_fact",
            page_override=1,
            subkey="admission_date",
            evidence_text=f"Admission Date: {adm_date}",
        )

    age_value = _extract_header_value(
        paged_doc,
        "Age at Admission",
    )

    if age_value is None:
        computed_age = compute_age(
            dob,
            admission.get(
                "admittime",
                "",
            ),
        )
        age_value = (
            str(computed_age)
            if computed_age is not None
            else None
        )

    if age_value is not None:
        add(
            (
                f"How old was I when I was admitted in "
                f"{admission_ref}?"
            ),
            age_value,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="age",
            evidence_text=f"Age at Admission: {age_value}",
        )

    los_value = _extract_header_value(
        paged_doc,
        "Length of Stay",
    )

    if los_value is None:
        los_days = compute_los_days(
            admission.get(
                "admittime",
                "",
            ),
            admission.get(
                "dischtime",
                "",
            ),
        )
        los_value = (
            f"{los_days} days"
            if los_days is not None
            else None
        )

    if los_value is not None:
        add(
            (
                f"How many days did I stay in the hospital during "
                f"{admission_ref}?"
            ),
            los_value,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="los",
            evidence_text=f"Length of Stay: {los_value}",
        )

    # Nota: Usiamo il campo clinical_notes o discharge_summary per cercare il service
    note_text_for_service = admission.get("clinical_notes", admission.get("discharge_summary", ""))
    service_match = SERVICE_RE.search(note_text_for_service)

    service = (
        admission.get("service")
        or (
            service_match.group(1).strip()
            if service_match
            else None
        )
    )

    if service:
        add(
            (
                f"Which medical team or specialty looked after me during "
                f"{admission_ref}?"
            ),
            str(service),
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="service",
            evidence_text=f"Hospital Service: {service}",
        )

    disposition = sections.get(
        "Discharge Disposition",
        "",
    ).strip()

    if disposition:
        add(
            (
                f"Where did I go after leaving the hospital at the "
                f"end of {admission_ref}?"
            ),
            disposition,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="disposition",
            evidence_text=f"Discharge Disposition: {disposition}",
        )

    if "has_mental_health_diagnosis" in admission:
        answer = (
            "Yes"
            if admission["has_mental_health_diagnosis"]
            else "No"
        )

        add(
            (
                f"Was any mental health condition recorded for me "
                f"in {admission_ref}?"
            ),
            answer,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="mental_health",
            evidence_text=f"Mental Health Diagnosis: {answer}",
        )

    if "had_icu_stay" in admission:
        answer = (
            "Yes"
            if admission["had_icu_stay"]
            else "No"
        )

        add(
            (
                f"Was I treated in intensive care at any point "
                f"during {admission_ref}?"
            ),
            answer,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="icu",
            evidence_text=f"ICU Stay: {answer}",
        )

    if "hospital_expire_flag" in admission:
        answer = (
            "Yes"
            if admission["hospital_expire_flag"]
            else "No"
        )

        add(
            (
                f"Did I experience in-hospital mortality during {admission_ref}?"
            ),
            answer,
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="mortality",
            evidence_text=f"In-Hospital Mortality: {answer}",
        )

    if admission.get("num_diagnoses") is not None:
        n_diag = admission["num_diagnoses"]

        add(
            (
                f"How many separate diagnoses were listed in my medical record "
                f"for {admission_ref}?"
            ),
            str(n_diag),
            "header",
            "header_fact_enriched",
            page_override=1,
            subkey="num_diagnoses",
            evidence_text=f"Number of Diagnoses: {n_diag}",
        )

    section_questions = {
        "Chief Complaint": (
            f"What symptoms or problems did I present with when I was "
            f"admitted on {adm_date_raw or 'this admission'}?"
        ),
        "Discharge Condition": (
            f"How was my health condition described when I was sent home after "
            f"{admission_ref}?"
        ),
        "Medications on Admission": (
            f"Which medicines was I already taking when I arrived for "
            f"{admission_ref}?"
        ),
        "Discharge Medications": (
            f"Which medicines was I instructed to take after leaving "
            f"the hospital following {admission_ref}?"
        ),
        "Past Medical History": (
            f"What health problems or medical history did I have before "
            f"{admission_ref}?"
        ),
        "Brief Hospital Course": (
            f"What procedures or clinical events happened to me during "
            f"{admission_ref}?"
        ),
    }

    section_item_questions = {
        "Medications on Admission": (
            f"Which medicine is listed first among the drugs I "
            f"was already taking at {admission_ref}?"
        ),
        "Discharge Medications": (
            f"Which medicine is listed first in the treatment I "
            f"was given to continue after {admission_ref}?"
        ),
        "Past Medical History": (
            f"Which pre-existing condition is recorded first in "
            f"my medical history for {admission_ref}?"
        ),
        "Chief Complaint": (
            f"What is my first complaint recorded at "
            f"{admission_ref}?"
        ),
        "Discharge Condition": (
            f"What is the first detail noted about my health condition "
            f"at discharge for {admission_ref}?"
        ),
        "Brief Hospital Course": (
            f"How does the hospital course summary begin for "
            f"{admission_ref}?"
        ),
    }

    for section_name, full_question in section_questions.items():
        content = sections.get(section_name)

        if not content:
            continue

        answer, extraction_mode = _extract_section_reference(
            section_name,
            content,
            max_full_len=320,
            max_unit_len=450,
        )

        if not answer:
            continue

        if extraction_mode == "full":
            question = full_question

        else:
            question = section_item_questions.get(
                section_name,
                (
                    f"What is recorded first under {section_name.lower()} "
                    f"for {admission_ref}?"
                ),
            )

        add(
            question,
            answer,
            section_name,
            "section_lookup",
            subkey=section_name,
            evidence_text=answer,
            section_text=content,
        )

    hpi = sections.get(
        "History of Present Illness"
    )

    if hpi:
        hpi_statement = _first_clinical_hpi_statement(
            hpi
        )

        if hpi_statement:
            hpi_answer, hpi_evidence = hpi_statement

            add(
                (
                    f"How does the clinical narrative of my illness "
                    f"begin in the record of {admission_ref}?"
                ),
                hpi_answer,
                "History of Present Illness",
                "specific_detail",
                evidence_text=hpi_evidence,
                section_text=hpi,
            )

    allergies = sections.get("Allergies")

    if allergies:
        allergy_reference = (
            _extract_allergy_negation_reference(
                allergies,
                max_len=320,
            )
        )

        if allergy_reference is not None:
            answer, allergy_evidence = allergy_reference

            add(
                (
                    f"Does my medical record for {admission_ref} state that I "
                    f"have no known allergies?"
                ),
                answer,
                "Allergies",
                "negation_check",
                evidence_text=allergy_evidence,
                section_text=allergies,
            )

    target_diagnosis_norm = _normalize_ws(
        diagnosis
    ).casefold()

    usable_siblings = [
        sibling
        for sibling in sibling_admissions
        if sibling.get("admittime")
    ]

    diagnosis_distinguishing_siblings = [
        sibling
        for sibling in usable_siblings
        if (
            _normalize_ws(
                str(sibling.get("diagnosis", ""))
            ).casefold()
            not in {"", "n/d", "n/a", target_diagnosis_norm}
        )
    ]

    if (
        diagnosis_distinguishing_siblings
        and target_diagnosis_norm
        not in {"", "n/d", "n/a"}
    ):

        before = len(items)

        add(
            (
                f"I was admitted to the hospital more than once. "
                f"Looking only at {admission_ref}, what was my main "
                f"problem or diagnosis?"
            ),
            diagnosis,
            "header",
            "multi_admission_distractor",
            page_override=1,
            subkey="diagnosis_disambiguation",
            evidence_text=f"Principal Diagnosis: {diagnosis}",
        )

        distractor_added = len(items) > before

        if distractor_added and SUPPRESS_DUPLICATE_DIAGNOSIS_QUESTION:

            items[:] = [
                item
                for item in items
                if not (
                    item["question_type"] == "header_fact"
                    and item["_subkey"] == "diagnosis"
                )
            ]

    return items

def _section_lookup_candidate_key(
    candidate: dict,
) -> tuple:
    return (
        SECTION_LOOKUP_RANK.get(
            candidate["_subkey"],
            len(SECTION_LOOKUP_PRIORITY),
        ),
        candidate["id"],
    )

def _type_availability(
    admissions_candidates: list[list[dict]],
) -> Counter:
    availability: Counter = Counter()

    for candidates in admissions_candidates:
        for category in {
            candidate["question_type"]
            for candidate in candidates
        }:
            availability[category] += 1

    return availability

def _normalized_answer_key(
    answer: object,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        _normalize_ws(str(answer)).casefold(),
    ).strip()

@dataclass
class _CostEdge:
    """One directed edge in the min-cost-flow graph used to balance question types across admissions."""

    to: int
    rev: int
    capacity: int
    cost: int

def _add_cost_edge(
    graph: list[list[_CostEdge]],
    src: int,
    dst: int,
    capacity: int,
    cost: int,
) -> _CostEdge:
    forward = _CostEdge(
        to=dst,
        rev=len(graph[dst]),
        capacity=capacity,
        cost=cost,
    )

    backward = _CostEdge(
        to=src,
        rev=len(graph[src]),
        capacity=0,
        cost=-cost,
    )

    graph[src].append(forward)
    graph[dst].append(backward)

    return forward

def _balanced_type_assignment(
    admissions_candidates: list[list[dict]],
    k: int,
) -> tuple[list[list[str]], Counter, Counter]:
    from collections import deque

    n_admissions = len(admissions_candidates)
    total_required = n_admissions * k
    availability = _type_availability(
        admissions_candidates
    )

    source = 0
    admission_base = 1
    category_base = admission_base + n_admissions
    sink = category_base + len(BENCHMARK_QUESTION_TYPES)

    graph: list[list[_CostEdge]] = [
        []
        for _ in range(sink + 1)
    ]

    category_node = {
        category: category_base + idx
        for idx, category
        in enumerate(BENCHMARK_QUESTION_TYPES)
    }

    assignment_edges: dict[
        tuple[int, str],
        _CostEdge,
    ] = {}

    for admission_idx, candidates in enumerate(
        admissions_candidates
    ):
        admission_node = admission_base + admission_idx

        _add_cost_edge(
            graph,
            source,
            admission_node,
            k,
            0,
        )

        available_types = [
            category
            for category in BENCHMARK_QUESTION_TYPES
            if any(
                candidate["question_type"] == category
                for candidate in candidates
            )
        ]

        if len(available_types) < k:
            raise RuntimeError(
                f"Admission index {admission_idx} has only "
                f"{len(available_types)} grounded question type(s); "
                f"{k} are required."
            )

        for category in available_types:
            assignment_edges[
                (admission_idx, category)
            ] = _add_cost_edge(
                graph,
                admission_node,
                category_node[category],
                1,
                0,
            )

    for category in BENCHMARK_QUESTION_TYPES:
        cap = availability.get(category, 0)

        for marginal_rank in range(cap):
            _add_cost_edge(
                graph,
                category_node[category],
                sink,
                1,
                marginal_rank,
            )

    flow = 0
    total_cost = 0

    while flow < total_required:
        dist = [float("inf")] * len(graph)
        parent: list[
            Optional[tuple[int, int]]
        ] = [None] * len(graph)
        in_queue = [False] * len(graph)

        dist[source] = 0
        queue = deque([source])
        in_queue[source] = True

        while queue:
            node = queue.popleft()
            in_queue[node] = False

            for edge_idx, edge in enumerate(
                graph[node]
            ):
                if edge.capacity <= 0:
                    continue

                new_dist = dist[node] + edge.cost

                if new_dist < dist[edge.to]:
                    dist[edge.to] = new_dist
                    parent[edge.to] = (
                        node,
                        edge_idx,
                    )

                    if not in_queue[edge.to]:
                        queue.append(edge.to)
                        in_queue[edge.to] = True

        if parent[sink] is None:
            break

        node = sink

        while node != source:
            prev, edge_idx = parent[node]
            edge = graph[prev][edge_idx]

            edge.capacity -= 1
            graph[node][edge.rev].capacity += 1
            node = prev

        flow += 1
        total_cost += int(dist[sink])

    if flow != total_required:
        raise RuntimeError(
            "Unable to assign two distinct grounded question types to every "
            f"admission. Required flow={total_required}, achieved={flow}. "
            f"Type availability={dict(availability)}"
        )

    assignment: list[list[str]] = []
    achieved_counts: Counter = Counter()

    for admission_idx in range(n_admissions):
        assigned_types = [
            category
            for category in BENCHMARK_QUESTION_TYPES
            if (
                (admission_idx, category) in assignment_edges
                and assignment_edges[
                    (admission_idx, category)
                ].capacity == 0
            )
        ]

        if len(assigned_types) != k:
            raise RuntimeError(
                f"Admission index {admission_idx}: expected {k} assigned "
                f"types, got {assigned_types}."
            )

        assignment.append(assigned_types)
        achieved_counts.update(assigned_types)

    print(
        "Question-type availability:",
        {
            category: availability.get(category, 0)
            for category in BENCHMARK_QUESTION_TYPES
        },
    )

    print(
        "Capability-balanced assignment:",
        {
            category: achieved_counts.get(category, 0)
            for category in BENCHMARK_QUESTION_TYPES
        },
    )

    print(
        "Challenge categories are availability-capped; "
        "no synthetic padding is performed."
    )

    return assignment, achieved_counts, availability

def _negation_candidate(
    candidates: list[dict],
) -> Optional[dict]:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate["question_type"] == "negation_check"
        ),
        None,
    )

def _rebalance_negation_assignment(
    assignment: list[list[str]],
    admissions_candidates: list[list[dict]],
) -> None:
    if not BALANCE_NEGATION_POLARITY:
        return

    def polarity_for_admission(
        admission_idx: int,
    ) -> Optional[str]:
        candidate = _negation_candidate(
            admissions_candidates[admission_idx]
        )

        if candidate is None:
            return None

        value = _normalize_ws(
            str(candidate["ground_truth_answer"])
        ).casefold()

        return value if value in ("yes", "no") else None

    for _ in range(len(assignment)):
        negation_admissions = [
            idx
            for idx, types in enumerate(assignment)
            if "negation_check" in types
        ]

        counts = Counter(
            polarity_for_admission(idx)
            for idx in negation_admissions
        )

        counts.pop(None, None)

        if abs(
            counts.get("yes", 0)
            - counts.get("no", 0)
        ) <= 1:
            break

        over = (
            "yes"
            if counts.get("yes", 0)
            > counts.get("no", 0)
            else "no"
        )
        under = "no" if over == "yes" else "yes"

        donors = [
            idx
            for idx in negation_admissions
            if polarity_for_admission(idx) == over
        ]

        receivers = [
            idx
            for idx, types in enumerate(assignment)
            if (
                "negation_check" not in types
                and polarity_for_admission(idx) == under
            )
        ]

        swapped = False

        for donor_idx in donors:
            donor_other_types = [
                category
                for category in assignment[donor_idx]
                if category != "negation_check"
            ]

            for receiver_idx in receivers:
                receiver_types = list(
                    assignment[receiver_idx]
                )

                for transfer_type in receiver_types:

                    if transfer_type in donor_other_types:
                        continue

                    donor_has_transfer = any(
                        candidate["question_type"]
                        == transfer_type
                        for candidate
                        in admissions_candidates[donor_idx]
                    )

                    if not donor_has_transfer:
                        continue

                    donor_new = [
                        transfer_type
                        if category == "negation_check"
                        else category
                        for category in assignment[donor_idx]
                    ]

                    receiver_new = [
                        "negation_check"
                        if category == transfer_type
                        else category
                        for category in assignment[receiver_idx]
                    ]

                    if (
                        len(set(donor_new)) != len(donor_new)
                        or len(set(receiver_new))
                        != len(receiver_new)
                    ):
                        continue

                    assignment[donor_idx] = donor_new
                    assignment[receiver_idx] = receiver_new
                    swapped = True
                    break

                if swapped:
                    break

            if swapped:
                break

        if not swapped:
            break

    final_polarity = Counter(
        _normalize_ws(
            str(
                _negation_candidate(
                    admissions_candidates[idx]
                )["ground_truth_answer"]
            )
        ).casefold()
        for idx, types in enumerate(assignment)
        if (
            "negation_check" in types
            and _negation_candidate(
                admissions_candidates[idx]
            ) is not None
        )
    )

    if final_polarity:
        print(
            "negation_check polarity after feasible balancing:",
            dict(final_polarity),
        )

def _pair_candidate_score(
    first: dict,
    second: dict,
    subkey_counts: Counter,
    section_usage: Counter,
    yes_no_count: int,
    total: int,
) -> tuple:
    duplicate_answer = int(
        _normalized_answer_key(
            first["ground_truth_answer"]
        )
        == _normalized_answer_key(
            second["ground_truth_answer"]
        )
    )

    duplicate_chunk = int(
        first["ground_truth_context"]
        == second["ground_truth_context"]
    )

    same_section = int(
        first["section"] == second["section"]
    )

    projected_nonchallenge_yes_no = sum(
        1
        for candidate in (first, second)
        if (
            candidate["_yes_no"]
            and candidate["question_type"]
            != "negation_check"
        )
    )

    yes_no_overflow = int(
        projected_nonchallenge_yes_no > 0
        and yes_no_count
        >= MAX_YES_NO_SHARE * total
    )

    subtype_load = sum(
        subkey_counts[
            (
                candidate["question_type"],
                candidate["_subkey"],
            )
        ]
        for candidate in (first, second)
    )

    section_lookup_load = sum(
        section_usage[candidate["_subkey"]]
        for candidate in (first, second)
        if candidate["question_type"] == "section_lookup"
    )

    same_page = int(
        first["page_number"]
        == second["page_number"]
    )

    return (
        duplicate_answer,
        duplicate_chunk,
        same_section,
        yes_no_overflow,
        subtype_load,
        section_lookup_load,
        same_page,
        first["id"],
        second["id"],
    )

def _choose_concrete_candidates(
    admissions_candidates: list[list[dict]],
    assignment: list[list[str]],
    k: int,
) -> list[dict]:
    if k != 2:
        raise ValueError(
            "Concrete pair selection currently expects k=2."
        )

    total = len(admissions_candidates) * k
    chosen: list[dict] = []

    subkey_counts: Counter = Counter()
    section_usage: Counter = Counter()
    yes_no_count = 0

    for candidates, assigned_types in zip(
        admissions_candidates,
        assignment,
    ):
        if len(assigned_types) != 2:
            raise RuntimeError(
                f"Expected two assigned types, got {assigned_types}."
            )

        first_type, second_type = assigned_types

        first_candidates = [
            candidate
            for candidate in candidates
            if candidate["question_type"] == first_type
        ]

        second_candidates = [
            candidate
            for candidate in candidates
            if candidate["question_type"] == second_type
        ]

        if not first_candidates or not second_candidates:
            raise RuntimeError(
                "Assigned type has no concrete grounded candidate."
            )

        pair_options = [
            (first, second)
            for first in first_candidates
            for second in second_candidates
        ]

        pair_options.sort(
            key=lambda pair: _pair_candidate_score(
                pair[0],
                pair[1],
                subkey_counts,
                section_usage,
                yes_no_count,
                total,
            )
        )

        first, second = pair_options[0]

        selected_pair = sorted(
            (first, second),
            key=lambda item: (
                BENCHMARK_QUESTION_TYPES.index(
                    item["question_type"]
                ),
                item["id"],
            ),
        )

        for selected in selected_pair:
            chosen.append(selected)

            subkey_counts[
                (
                    selected["question_type"],
                    selected["_subkey"],
                )
            ] += 1

            if selected["question_type"] == "section_lookup":
                section_usage[
                    selected["_subkey"]
                ] += 1

            if selected["_yes_no"]:
                yes_no_count += 1

    return chosen

def choose_balanced_selection_k(
    admissions_candidates: list[list[dict]],
    k: int,
) -> list[dict]:
    assignment, achieved_counts, availability = (
        _balanced_type_assignment(
            admissions_candidates,
            k,
        )
    )

    _rebalance_negation_assignment(
        assignment,
        admissions_candidates,
    )

    post_balance_counts = Counter(
        category
        for assigned_types in assignment
        for category in assigned_types
    )

    if post_balance_counts != achieved_counts:
        raise RuntimeError(
            "Negation polarity balancing changed macro question-type counts."
        )

    chosen = _choose_concrete_candidates(
        admissions_candidates,
        assignment,
        k,
    )

    final_counts = Counter(
        item["question_type"]
        for item in chosen
    )

    if final_counts != achieved_counts:
        raise RuntimeError(
            "Concrete candidate selection changed macro question-type counts."
        )

    print(
        "Final macro-type counts:",
        {
            category: final_counts.get(category, 0)
            for category in BENCHMARK_QUESTION_TYPES
        },
    )

    print(
        "Final macro-type balance is corpus-constrained, not quota-padded."
    )

    return chosen

def validate_items(
    items: list[dict],
    expected_admissions: set[tuple[str, str]],
) -> None:

    ids = [
        item["id"]
        for item in items
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate item IDs detected."
        )

    by_admission: dict[
        tuple[str, str],
        list[dict],
    ] = {}

    for item in items:
        key = (
            item["patient_id"],
            item["document_name"],
        )

        by_admission.setdefault(
            key,
            [],
        ).append(item)

        context = item[
            "ground_truth_context"
        ]

        if not context:
            raise RuntimeError(
                f"{item['id']}: empty gold chunk."
            )

        if len(context) > CHUNK_MAX_CHARS:
            raise RuntimeError(
                f"{item['id']}: gold chunk exceeds "
                f"{CHUNK_MAX_CHARS} characters."
            )

        if item["question_type"] not in BENCHMARK_QUESTION_TYPES:
            raise RuntimeError(
                f"{item['id']}: unknown question_type "
                f"{item['question_type']!r}."
            )

        if not item.get("ground_truth_page_context"):
            raise RuntimeError(
                f"{item['id']}: empty source-page context."
            )

        if REJECT_DEID_SURROGATE_QA and (
            _has_deid_surrogate(item["question"])
            or _has_deid_surrogate(
                str(item["ground_truth_answer"])
            )
        ):
            raise RuntimeError(
                f"{item['id']}: de-identification surrogate "
                f"(surrogate id) leaked into the question or answer."
            )

        if re.search(
            r"\b\d{5,}\b",
            f"{item['question']} {item['ground_truth_answer']}",
        ):
            raise RuntimeError(
                f"{item['id']}: id-like 5+ digit number in "
                f"the question or answer."
            )

        if str(item["patient_id"]) in item["question"]:
            raise RuntimeError(
                f"{item['id']}: patient id leaked into the question."
            )

    selected_admissions = set(by_admission)

    missing_admissions = sorted(
        expected_admissions - selected_admissions
    )
    unexpected_admissions = sorted(
        selected_admissions - expected_admissions
    )

    if missing_admissions:
        raise RuntimeError(
            "Admissions disappeared during selection: "
            f"{missing_admissions[:10]}"
        )

    if unexpected_admissions:
        raise RuntimeError(
            "Unexpected admissions in selected benchmark: "
            f"{unexpected_admissions[:10]}"
        )

    expected_total = (
        len(expected_admissions)
        * QUESTIONS_PER_ADMISSION
    )

    if len(items) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} benchmark items "
            f"({len(expected_admissions)} admissions x "
            f"{QUESTIONS_PER_ADMISSION}), got {len(items)}."
        )

    for key, admission_items in by_admission.items():

        if (
            len(admission_items)
            != QUESTIONS_PER_ADMISSION
        ):
            raise RuntimeError(
                f"{key}: expected "
                f"{QUESTIONS_PER_ADMISSION} questions, "
                f"got {len(admission_items)}."
            )

        types = [
            item["question_type"]
            for item in admission_items
        ]

        if len(types) != len(set(types)):
            raise RuntimeError(
                f"{key}: repeated question_type "
                f"within the same admission."
            )

    by_type = Counter(
        item["question_type"]
        for item in items
    )

    total = len(items)

    print(
        "Category mix:",
        {
            category: (
                by_type[category],
                f"{by_type[category] / max(1, total):.0%}",
            )
            for category in BENCHMARK_QUESTION_TYPES
        },
    )

    yes_no_items = [
        item
        for item in items
        if _normalize_ws(
            str(item["ground_truth_answer"])
        ).casefold()
        in ("yes", "no")
    ]

    yes_no_share = len(yes_no_items) / max(1, total)

    print(
        "Yes/No items:",
        len(yes_no_items),
        f"({yes_no_share:.0%})",
        "polarity:",
        dict(
            Counter(
                _normalize_ws(
                    str(item["ground_truth_answer"])
                ).casefold()
                for item in yes_no_items
            )
        ),
    )

    non_negation_items = [
        item
        for item in items
        if item["question_type"] != "negation_check"
    ]

    non_negation_yes_no_items = [
        item
        for item in non_negation_items
        if _normalize_ws(
            str(item["ground_truth_answer"])
        ).casefold()
        in ("yes", "no")
    ]

    non_negation_yes_no_share = (
        len(non_negation_yes_no_items)
        / max(1, len(non_negation_items))
    )

    print(
        "Non-negation Yes/No items:",
        len(non_negation_yes_no_items),
        f"({non_negation_yes_no_share:.0%})",
    )

    if non_negation_yes_no_share > MAX_YES_NO_SHARE + 0.02:
        print(
            "WARNING: non-negation Yes/No share exceeds the configured cap of "
            f"{MAX_YES_NO_SHARE:.0%}; guessing baseline may be inflated."
        )

    page_counts = Counter(
        item["page_number"]
        for item in items
    )

    page_one_share = page_counts[1] / max(1, total)

    print(
        "Gold evidence pages:",
        dict(sorted(page_counts.items())),
        f"- page 1 share: {page_one_share:.0%}",
    )

    if page_one_share > 0.85:
        print(
            "WARNING: most gold evidence is on page 1; retrieval is "
            "close to trivial for this corpus."
        )

    print(
        "Sections covered:",
        dict(
            sorted(
                Counter(
                    item["section"]
                    for item in items
                ).items()
            )
        ),
    )

    short_answers = sum(
        1
        for item in items
        if len(str(item["ground_truth_answer"])) <= 12
    )

    print(
        "Very short answers (<=12 chars):",
        short_answers,
        f"({short_answers / max(1, total):.0%})",
    )

    section_lookup_counts = Counter(
        item["section"]
        for item in items
        if item["question_type"] == "section_lookup"
    )

    print(
        "section_lookup sections:",
        {
            section_name: section_lookup_counts[section_name]
            for section_name in SECTION_LOOKUP_PRIORITY
            if section_lookup_counts[section_name] > 0
        },
    )

def _enforce_distinct_evidence_per_admission(
    all_items: list[dict],
    admissions_candidates: list[list[dict]],
) -> list[tuple[str, str, str]]:
    if not REQUIRE_DISTINCT_EVIDENCE_PER_ADMISSION:
        return []

    positions_by_admission: dict[
        tuple[str, str],
        list[int],
    ] = {}

    for index, item in enumerate(all_items):
        positions_by_admission.setdefault(
            (
                item["patient_id"],
                item["document_name"],
            ),
            [],
        ).append(index)

    unresolved: list[tuple[str, str, str]] = []

    for key, positions in positions_by_admission.items():
        if len(positions) < 2:
            continue

        seen_chunks: set[str] = set()

        for position in positions:
            item = all_items[position]
            chunk = item["ground_truth_context"]

            if chunk in seen_chunks:
                unresolved.append(
                    (
                        key[0],
                        key[1],
                        item["question_type"],
                    )
                )
            else:
                seen_chunks.add(chunk)

    return unresolved

def main() -> None:

    print(f"Script directory: {BASE_DIR}")
    print(f"Source JSON: {SOURCE_JSON}")
    print(f"PDF directory: {PDF_DIR}")

    if not SOURCE_JSON.exists():
        raise FileNotFoundError(
            f"Source JSON not found: {SOURCE_JSON}"
        )

    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR}"
        )

    with SOURCE_JSON.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    admissions_candidates: list[
        list[dict]
    ] = []

    expected_admissions: set[
        tuple[str, str]
    ] = set()

    missing_pdfs: list[str] = []
    insufficient_candidates: list[tuple[str, str, list[str]]] = []

    for patient in data:

        info = patient[
            "patient_info"
        ]

        admissions = patient.get(
            "admissions",
            [],
        )

        for admission in admissions:

            document_name = (
                f"Patient_"
                f"{info['subject_id']}_"
                f"Admission_"
                f"{admission['hadm_id']}.pdf"
            )

            pdf_path = (
                PDF_DIR
                / document_name
            )

            if not pdf_path.exists():
                missing_pdfs.append(
                    document_name
                )
                continue

            paged_doc = (
                extract_paged_document(
                    pdf_path
                )
            )

            sibling_admissions = []

            for other in admissions:
                if other["hadm_id"] == admission["hadm_id"]:
                    continue

                sibling_document_name = (
                    f"Patient_"
                    f"{info['subject_id']}_"
                    f"Admission_"
                    f"{other['hadm_id']}.pdf"
                )

                if (PDF_DIR / sibling_document_name).exists():
                    sibling_admissions.append(other)

            candidates = (
                build_questions_for_admission(
                    info,
                    admission,
                    sibling_admissions,
                    document_name,
                    paged_doc,
                )
            )

            admission_key = (
                str(info["subject_id"]),
                document_name,
            )
            expected_admissions.add(admission_key)

            available_types = sorted(
                {
                    candidate["question_type"]
                    for candidate in candidates
                }
            )

            if len(available_types) < QUESTIONS_PER_ADMISSION:
                insufficient_candidates.append(
                    (
                        str(info["subject_id"]),
                        document_name,
                        available_types,
                    )
                )

            admissions_candidates.append(
                candidates
            )

    if missing_pdfs and STRICT_CORPUS_COMPLETENESS:
        raise FileNotFoundError(
            "Benchmark corpus is incomplete: "
            f"{len(missing_pdfs)} expected PDF(s) are missing. "
            f"Examples: {missing_pdfs[:10]}"
        )

    if insufficient_candidates:
        raise RuntimeError(
            "At least one admission has fewer than "
            f"{QUESTIONS_PER_ADMISSION} distinct grounded question types. "
            f"Examples: {insufficient_candidates[:10]}"
        )

    all_items = (
        choose_balanced_selection_k(
            admissions_candidates,
            k=QUESTIONS_PER_ADMISSION,
        )
    )

    duplicate_evidence_report = _enforce_distinct_evidence_per_admission(
        all_items,
        admissions_candidates,
    )

    validate_items(
        all_items,
        expected_admissions=expected_admissions,
    )

    subkey_by_type: dict[
        str,
        Counter,
    ] = {}

    for item in all_items:

        subkey_by_type.setdefault(
            item["question_type"],
            Counter(),
        )[item["_subkey"]] += 1

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in all_items:

            item_to_write = {
                key: value
                for key, value
                in item.items()
                if key not in INTERNAL_FIELDS
                and not key.startswith("_")
            }

            f.write(
                json.dumps(
                    item_to_write,
                    ensure_ascii=False,
                )
                + "\n"
            )

    by_type = Counter(
        item["question_type"]
        for item in all_items
    )

    by_section = Counter(
        item["section"]
        for item in all_items
    )

    unique_admissions = len(
        {
            (
                item["patient_id"],
                item["document_name"],
            )
            for item in all_items
        }
    )

    counts_by_admission: dict[
        tuple,
        int,
    ] = {}

    for item in all_items:

        key = (
            item["patient_id"],
            item["document_name"],
        )

        counts_by_admission[
            key
        ] = (
            counts_by_admission.get(
                key,
                0,
            )
            + 1
        )

    per_admission_counts = Counter(
        counts_by_admission.values()
    )

    gold_lengths = [
        len(
            item["ground_truth_context"]
        )
        for item in all_items
    ]

    print(
        f"Wrote {len(all_items)} "
        f"benchmark items to "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Missing PDFs (skipped): "
        f"{len(missing_pdfs)}"
    )

    if missing_pdfs:
        print(
            "  ",
            missing_pdfs[:10],
        )

    print(
        f"Unique admissions covered: "
        f"{unique_admissions}"
    )

    print(
        "By question_type:",
        dict(by_type),
    )

    print(
        "By section:",
        dict(by_section),
    )

    print(
        "Admissions contributing "
        "exactly N items:",
        dict(
            sorted(
                per_admission_counts.items()
            )
        ),
    )

    print(
        "Gold policy: strict full-evidence containment "
        "inside one Android-equivalent 500/100 chunk"
    )

    print(
        "Section-boundary assertion:",
        "on" if ASSERT_ANSWER_INSIDE_SECTION else "off",
    )

    print(
        "Admissions with unresolved duplicate gold evidence:",
        len(duplicate_evidence_report),
    )

    if duplicate_evidence_report:
        print("  ", duplicate_evidence_report[:10])

    print(
        "Gold chunk length "
        "(min/avg/max chars):",
        (
            min(gold_lengths)
            if gold_lengths
            else 0
        ),
        (
            round(
                sum(gold_lengths)
                / len(gold_lengths),
                1,
            )
            if gold_lengths
            else 0
        ),
        (
            max(gold_lengths)
            if gold_lengths
            else 0
        ),
    )

    print(
        "Within-category "
        "sub-field diversity:"
    )

    for question_type, counter in sorted(
        subkey_by_type.items()
    ):
        print(
            f"  {question_type}: "
            f"{dict(counter)}"
        )

if __name__ == "__main__":
    main()