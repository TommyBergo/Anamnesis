"""Generates rule-based multi-document QA pairs (same-patient and cross-patient admission comparisons) from MIMIC-III."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from shared.pdf_paging import extract_paged_document, find_pages_for_text

BASE_DIR = Path(__file__).resolve().parent
SOURCE_JSON = BASE_DIR / "mimic_stratified_sample.json"
PDF_DIR = BASE_DIR / "Ammissioni_PDF_stratified_enriched"
OUTPUT_PATH = BASE_DIR / "multidoc_rulebased.jsonl"

SEED = 42

PAIR_FIELDS = (
    "icu",
    "service",
    "diagnosis",
    "num_diagnoses",
    "mortality",
)

TRAJECTORY_FIELDS = (
    "icu",
    "service",
    "diagnosis",
    "num_diagnoses",
)

CROSS_FIELDS = (
    "icu",
    "service",
    "num_diagnoses",
    "mortality",
)

FIELD_SPECS = {
    "icu": {
        "json_key": "had_icu_stay",
        "label": "ICU-stay status",
        "pdf_label": "ICU Stay",
        "kind": "bool",
    },
    "service": {
        "json_key": "service",
        "label": "hospital service",
        "pdf_label": "Hospital Service",
        "kind": "text",
    },
    "diagnosis": {
        "json_key": "diagnosis",
        "label": "principal diagnosis",
        "pdf_label": "Principal Diagnosis",
        "kind": "text",
    },
    "num_diagnoses": {
        "json_key": "num_diagnoses",
        "label": "number of recorded diagnoses",
        "pdf_label": "Number of Diagnoses",
        "kind": "number",
    },
    "mortality": {
        "json_key": "hospital_expire_flag",
        "label": "in-hospital mortality outcome",
        "pdf_label": "In-Hospital Mortality",
        "kind": "bool",
    },
}

EXCLUDED_CROSS_DIAGNOSES = {"NEWBORN"}

@dataclass(frozen=True)
class Admission:
    """One patient admission, identified by patient and admission id."""

    patient_id: str
    admission_id: str
    admission: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return self.patient_id, self.admission_id

    @property
    def admit_date(self) -> str:
        raw = str(self.admission.get("admittime", "")).strip()
        return raw.split(" ")[0] if raw else self.admission_id

@dataclass(frozen=True)
class PairCandidate:
    """A candidate two-admission comparison question, grounded in one field that differs (or matches) between the two admissions."""

    task: str
    field: str
    left: Admission
    right: Admission
    value_left: Any
    value_right: Any
    anchor_diagnosis: str | None = None

    @property
    def source_set(self) -> frozenset[tuple[str, str]]:
        return frozenset((self.left.key, self.right.key))

    @property
    def relation(self) -> str:
        return (
            "same"
            if normalize_field(self.field, self.value_left)
            == normalize_field(self.field, self.value_right)
            else "different"
        )

@dataclass(frozen=True)
class TrajectoryCandidate:
    """A candidate three-admission trajectory question tracking how one field changed across a patient's consecutive admissions."""

    field: str
    patient_id: str
    admissions: tuple[Admission, Admission, Admission]
    values: tuple[Any, Any, Any]

    @property
    def source_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(adm.key for adm in self.admissions)

    @property
    def relation(self) -> str:
        normalized = [normalize_field(self.field, value) for value in self.values]
        return "same" if len(set(normalized)) == 1 else "different"

def stable_rank(*parts: object) -> int:
    raw = "|".join(map(str, (SEED, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")

def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text if text else None

def normalize_bool(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    text = normalize_text(value)
    if text is None:
        return None

    lowered = text.casefold()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return None

def normalize_field(field: str, value: Any) -> Any | None:
    kind = FIELD_SPECS[field]["kind"]

    if kind == "bool":
        return normalize_bool(value)

    if kind == "number":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    text = normalize_text(value)
    return text.casefold() if text else None

def render_field(field: str, value: Any) -> str:
    normalized = normalize_field(field, value)
    if normalized is None:
        raise ValueError(f"Invalid value for {field}: {value!r}")

    kind = FIELD_SPECS[field]["kind"]

    if kind == "bool":
        return "Yes" if normalized else "No"
    if kind == "number":
        return str(normalized)

    return normalize_text(value) or ""

def field_value(adm: Admission, field: str) -> Any | None:
    value = adm.admission.get(FIELD_SPECS[field]["json_key"])
    return value if normalize_field(field, value) is not None else None

def load_patients() -> dict[str, list[Admission]]:
    with SOURCE_JSON.open(encoding="utf-8") as f:
        data = json.load(f)

    patients = data if isinstance(data, list) else data.get("patients", data)

    by_patient: dict[str, list[Admission]] = {}

    for patient in patients:
        pid = str(patient["patient_info"]["subject_id"])
        admissions = [
            Admission(pid, str(adm["hadm_id"]), adm)
            for adm in sorted(
                patient.get("admissions", []),
                key=lambda x: x["admittime"],
            )
        ]
        by_patient[pid] = admissions

    return by_patient

class GroundingVerifier:
    """Confirms a candidate answer's evidence text actually appears in its source admission's PDF."""

    def __init__(self) -> None:
        self.cache: dict[Path, Any] = {}
        self.failures: list[tuple[Any, ...]] = []

    def pdf_path(self, adm: Admission) -> Path:
        return PDF_DIR / f"Patient_{adm.patient_id}_Admission_{adm.admission_id}.pdf"

    def contains(self, adm: Admission, text: str) -> bool:
        path = self.pdf_path(adm)

        if not path.exists():
            self.failures.append((*adm.key, "missing pdf"))
            return False

        if path not in self.cache:
            self.cache[path] = extract_paged_document(path)

        page, _ = find_pages_for_text(self.cache[path], text)

        if page is None:
            self.failures.append((*adm.key, f"not found: {text!r}"))
            return False

        return True

def evidence_chunk(field: str, value: Any) -> str:
    return f"{FIELD_SPECS[field]['pdf_label']}: {render_field(field, value)}"

def source_entry(adm: Admission, chunk: str) -> dict[str, str]:
    return {
        "patient_id": adm.patient_id,
        "admission_id": adm.admission_id,
        "source_pdf": f"Patient_{adm.patient_id}_Admission_{adm.admission_id}.pdf",
        "source_chunk": chunk,
    }

def render_pair(
    candidate: PairCandidate,
) -> tuple[str, str, list[tuple[Admission, str]]]:
    spec = FIELD_SPECS[candidate.field]
    left, right = candidate.left, candidate.right

    left_value = render_field(candidate.field, candidate.value_left)
    right_value = render_field(candidate.field, candidate.value_right)

    if candidate.field == "num_diagnoses":
        n_left, n_right = int(left_value), int(right_value)

        if candidate.task == "same_patient_two_admissions":
            if n_left == n_right:
                summary = "The number of my recorded diagnoses remained unchanged."
            elif n_right > n_left:
                summary = (
                    f"The number of my recorded diagnoses increased from {n_left} to {n_right}."
                )
            else:
                summary = (
                    f"The number of my recorded diagnoses decreased from {n_left} to {n_right}."
                )
        else:
            if n_left == n_right:
                summary = "The two admissions had the same number of recorded diagnoses."
            elif n_left > n_right:
                summary = f"Patient {left.patient_id}'s admission had more recorded diagnoses."
            else:
                summary = f"Patient {right.patient_id}'s admission had more recorded diagnoses."
    else:
        summary = (
            f"My recorded {spec['label']} was the same."
            if candidate.relation == "same"
            else f"My recorded {spec['label']} differed."
        )

    if candidate.task == "same_patient_two_admissions":
        question = (
            f"During my hospitalizations beginning on {left.admit_date} and "
            f"{right.admit_date}, what was my {spec['label']} in each, "
            f"and was it the same or different?"
        )
        answer = (
            f"My hospitalization beginning {left.admit_date}: "
            f"{spec['pdf_label']}: {left_value}; my hospitalization beginning "
            f"{right.admit_date}: {spec['pdf_label']}: {right_value}. {summary}"
        )
    else:
        diagnosis = candidate.anchor_diagnosis or ""
        question = (
            f"Patient {left.patient_id} (hospitalization beginning {left.admit_date}) "
            f"and patient {right.patient_id} (hospitalization beginning "
            f"{right.admit_date}) shared the principal diagnosis \"{diagnosis}\". "
            f"What was the {spec['label']} for each, and was it the same or different?"
        )
        answer = (
            f"Patient {left.patient_id}, hospitalization beginning {left.admit_date}: "
            f"{spec['pdf_label']}: {left_value}; patient {right.patient_id}, "
            f"hospitalization beginning {right.admit_date}: "
            f"{spec['pdf_label']}: {right_value}. {summary}"
        )

    evidence = [
        (left, evidence_chunk(candidate.field, candidate.value_left)),
        (right, evidence_chunk(candidate.field, candidate.value_right)),
    ]

    return question, answer, evidence

def render_trajectory(
    candidate: TrajectoryCandidate,
) -> tuple[str, str, list[tuple[Admission, str]]]:
    spec = FIELD_SPECS[candidate.field]
    admissions = candidate.admissions
    rendered_values = [
        render_field(candidate.field, value)
        for value in candidate.values
    ]

    dates = [adm.admit_date for adm in admissions]

    question = (
        f"Across my hospitalizations beginning on {dates[0]}, {dates[1]}, and "
        f"{dates[2]}, what was my {spec['label']} at each hospitalization, "
        f"and did it remain consistent or vary across the three?"
    )

    details = "; ".join(
        f"my hospitalization beginning {adm.admit_date}: "
        f"{spec['pdf_label']}: {value}"
        for adm, value in zip(admissions, rendered_values)
    )

    if candidate.relation == "same":
        summary = (
            f"My {spec['label']} remained consistent across all three hospitalizations."
        )
    else:
        summary = f"My {spec['label']} varied across the three hospitalizations."

    answer = f"{details}. {summary}"

    evidence = [
        (adm, evidence_chunk(candidate.field, raw_value))
        for adm, raw_value in zip(admissions, candidate.values)
    ]

    return question, answer, evidence

def make_same_patient_candidates(
    by_patient: dict[str, list[Admission]],
) -> dict[frozenset[tuple[str, str]], list[PairCandidate]]:
    grouped: dict[frozenset[tuple[str, str]], list[PairCandidate]] = defaultdict(list)

    for admissions in by_patient.values():
        if len(admissions) < 2:
            continue

        for left, right in combinations(admissions, 2):
            for field in PAIR_FIELDS:
                left_value = field_value(left, field)
                right_value = field_value(right, field)

                if left_value is None or right_value is None:
                    continue

                candidate = PairCandidate(
                    task="same_patient_two_admissions",
                    field=field,
                    left=left,
                    right=right,
                    value_left=left_value,
                    value_right=right_value,
                )
                grouped[candidate.source_set].append(candidate)

    return grouped

def make_trajectory_candidates(
    by_patient: dict[str, list[Admission]],
) -> dict[frozenset[tuple[str, str]], list[TrajectoryCandidate]]:
    grouped: dict[frozenset[tuple[str, str]], list[TrajectoryCandidate]] = defaultdict(list)

    for pid, admissions in by_patient.items():
        if len(admissions) < 3:
            continue

        for i in range(len(admissions) - 2):
            triple = (admissions[i], admissions[i + 1], admissions[i + 2])

            for field in TRAJECTORY_FIELDS:
                values = tuple(field_value(adm, field) for adm in triple)

                if any(value is None for value in values):
                    continue

                candidate = TrajectoryCandidate(
                    field=field,
                    patient_id=pid,
                    admissions=triple,
                    values=values,
                )
                grouped[candidate.source_set].append(candidate)

    return grouped

def make_cross_candidates(
    by_patient: dict[str, list[Admission]],
) -> dict[frozenset[tuple[str, str]], list[PairCandidate]]:
    by_diagnosis: dict[str, list[tuple[str, Admission, str]]] = defaultdict(list)

    for pid, admissions in by_patient.items():
        for adm in admissions:
            diagnosis = normalize_text(adm.admission.get("diagnosis"))

            if not diagnosis or diagnosis.upper() in EXCLUDED_CROSS_DIAGNOSES:
                continue

            by_diagnosis[diagnosis.casefold()].append((pid, adm, diagnosis))

    grouped: dict[frozenset[tuple[str, str]], list[PairCandidate]] = defaultdict(list)

    for entries in by_diagnosis.values():
        for (pid_a, adm_a, diagnosis_a), (pid_b, adm_b, _) in combinations(entries, 2):
            if pid_a == pid_b:
                continue

            if adm_b.key < adm_a.key:
                pid_a, pid_b = pid_b, pid_a
                adm_a, adm_b = adm_b, adm_a
                diagnosis_a = normalize_text(adm_a.admission.get("diagnosis")) or diagnosis_a

            for field in CROSS_FIELDS:
                value_a = field_value(adm_a, field)
                value_b = field_value(adm_b, field)

                if value_a is None or value_b is None:
                    continue

                candidate = PairCandidate(
                    task="cross_patient_two_admissions",
                    field=field,
                    left=adm_a,
                    right=adm_b,
                    value_left=value_a,
                    value_right=value_b,
                    anchor_diagnosis=diagnosis_a,
                )
                grouped[candidate.source_set].append(candidate)

    return grouped

def verify_cross_anchor(
    candidate: PairCandidate,
    verifier: GroundingVerifier,
) -> bool:
    if candidate.task != "cross_patient_two_admissions":
        return True

    left_diag = field_value(candidate.left, "diagnosis")
    right_diag = field_value(candidate.right, "diagnosis")

    if left_diag is None or right_diag is None:
        return False

    return (
        verifier.contains(candidate.left, evidence_chunk("diagnosis", left_diag))
        and verifier.contains(candidate.right, evidence_chunk("diagnosis", right_diag))
    )

def choose_pair_candidate(
    candidates: list[PairCandidate],
    field_use: Counter[str],
    relation_use: Counter[tuple[str, str]],
) -> list[PairCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            field_use[candidate.field],
            relation_use[(candidate.field, candidate.relation)],
            stable_rank(
                candidate.task,
                candidate.field,
                candidate.left.patient_id,
                candidate.left.admission_id,
                candidate.right.patient_id,
                candidate.right.admission_id,
            ),
        ),
    )

def choose_trajectory_candidate(
    candidates: list[TrajectoryCandidate],
    field_use: Counter[str],
    relation_use: Counter[tuple[str, str]],
) -> list[TrajectoryCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            field_use[candidate.field],
            relation_use[(candidate.field, candidate.relation)],
            stable_rank(
                candidate.patient_id,
                candidate.field,
                *(adm.admission_id for adm in candidate.admissions),
            ),
        ),
    )

def build_pair_items(
    grouped_candidates: dict[frozenset[tuple[str, str]], list[PairCandidate]],
    verifier: GroundingVerifier,
) -> list[dict[str, Any]]:
    field_use: Counter[str] = Counter()
    relation_use: Counter[tuple[str, str]] = Counter()
    items: list[dict[str, Any]] = []

    source_sets = sorted(
        grouped_candidates.keys(),
        key=lambda source_set: stable_rank(*sorted(source_set)),
    )

    for source_set in source_sets:
        ordered = choose_pair_candidate(
            grouped_candidates[source_set],
            field_use,
            relation_use,
        )

        accepted: PairCandidate | None = None
        accepted_question = ""
        accepted_answer = ""
        accepted_evidence: list[tuple[Admission, str]] = []

        for candidate in ordered:
            question, answer, evidence = render_pair(candidate)

            if not all(verifier.contains(adm, chunk) for adm, chunk in evidence):
                continue

            if not verify_cross_anchor(candidate, verifier):
                continue

            accepted = candidate
            accepted_question = question
            accepted_answer = answer
            accepted_evidence = evidence
            break

        if accepted is None:
            continue

        prefix = (
            "PAIR"
            if accepted.task == "same_patient_two_admissions"
            else "CROSS"
        )

        item_id = (
            f"RB_{prefix}_{accepted.field.upper()}_"
            f"P{accepted.left.patient_id}_{accepted.left.admission_id}_"
            f"P{accepted.right.patient_id}_{accepted.right.admission_id}"
        )

        items.append(
            {
                "id": item_id,
                "comparison_type": accepted.task,
                "question": accepted_question,
                "reference_answer": accepted_answer,
                "sources": [
                    source_entry(adm, chunk)
                    for adm, chunk in accepted_evidence
                ],
            }
        )

        field_use[accepted.field] += 1
        relation_use[(accepted.field, accepted.relation)] += 1

    label = items[0]["comparison_type"] if items else "pair"
    print(f"{label}: {len(items)} items")
    print("  fields:", dict(field_use))
    print(
        "  relation:",
        {
            f"{field}:{relation}": count
            for (field, relation), count in relation_use.items()
        },
    )

    return items

def build_trajectory_items(
    grouped_candidates: dict[
        frozenset[tuple[str, str]],
        list[TrajectoryCandidate],
    ],
    verifier: GroundingVerifier,
) -> list[dict[str, Any]]:
    field_use: Counter[str] = Counter()
    relation_use: Counter[tuple[str, str]] = Counter()
    items: list[dict[str, Any]] = []

    source_sets = sorted(
        grouped_candidates.keys(),
        key=lambda source_set: stable_rank(*sorted(source_set)),
    )

    for source_set in source_sets:
        ordered = choose_trajectory_candidate(
            grouped_candidates[source_set],
            field_use,
            relation_use,
        )

        accepted: TrajectoryCandidate | None = None
        accepted_question = ""
        accepted_answer = ""
        accepted_evidence: list[tuple[Admission, str]] = []

        for candidate in ordered:
            question, answer, evidence = render_trajectory(candidate)

            if not all(verifier.contains(adm, chunk) for adm, chunk in evidence):
                continue

            accepted = candidate
            accepted_question = question
            accepted_answer = answer
            accepted_evidence = evidence
            break

        if accepted is None:
            continue

        admission_ids = "_".join(
            adm.admission_id
            for adm in accepted.admissions
        )

        item_id = (
            f"RB_TRAJ_{accepted.field.upper()}_"
            f"P{accepted.patient_id}_{admission_ids}"
        )

        items.append(
            {
                "id": item_id,
                "comparison_type": "same_patient_three_admission_trajectory",
                "question": accepted_question,
                "reference_answer": accepted_answer,
                "sources": [
                    source_entry(adm, chunk)
                    for adm, chunk in accepted_evidence
                ],
            }
        )

        field_use[accepted.field] += 1
        relation_use[(accepted.field, accepted.relation)] += 1

    print(f"same_patient_three_admission_trajectory: {len(items)} items")
    print("  fields:", dict(field_use))
    print(
        "  relation:",
        {
            f"{field}:{relation}": count
            for (field, relation), count in relation_use.items()
        },
    )

    return items

def validate_final(items: list[dict[str, Any]]) -> None:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item IDs detected.")

    source_sets: set[tuple[str, frozenset[tuple[str, str]]]] = set()

    for item in items:
        sources = item["sources"]
        ctype = item["comparison_type"]

        source_set = frozenset(
            (
                str(source["patient_id"]),
                str(source["admission_id"]),
            )
            for source in sources
        )

        key = (ctype, source_set)
        if key in source_sets:
            raise RuntimeError(f"{item['id']}: duplicate admission set detected.")
        source_sets.add(key)

        if ctype == "same_patient_two_admissions":
            if len(sources) != 2:
                raise RuntimeError(f"{item['id']}: pair must have 2 sources.")
            if len({str(s["patient_id"]) for s in sources}) != 1:
                raise RuntimeError(f"{item['id']}: expected one patient.")
            if len({str(s["admission_id"]) for s in sources}) != 2:
                raise RuntimeError(f"{item['id']}: expected two admissions.")

        elif ctype == "cross_patient_two_admissions":
            if len(sources) != 2:
                raise RuntimeError(f"{item['id']}: cross-patient item must have 2 sources.")
            if len({str(s["patient_id"]) for s in sources}) != 2:
                raise RuntimeError(f"{item['id']}: expected two different patients.")

        elif ctype == "same_patient_three_admission_trajectory":
            if len(sources) != 3:
                raise RuntimeError(f"{item['id']}: trajectory must have 3 sources.")
            if len({str(s["patient_id"]) for s in sources}) != 1:
                raise RuntimeError(f"{item['id']}: expected one patient.")
            if len({str(s["admission_id"]) for s in sources}) != 3:
                raise RuntimeError(f"{item['id']}: expected three admissions.")

        else:
            raise RuntimeError(
                f"{item['id']}: unknown comparison_type {ctype!r}."
            )

def main() -> None:
    by_patient = load_patients()
    verifier = GroundingVerifier()

    same_patient_candidates = make_same_patient_candidates(by_patient)
    trajectory_candidates = make_trajectory_candidates(by_patient)
    cross_patient_candidates = make_cross_candidates(by_patient)

    print("Eligible unique source sets before grounding:")
    print("  same-patient pairs:", len(same_patient_candidates))
    print("  three-admission trajectories:", len(trajectory_candidates))
    print("  cross-patient pairs:", len(cross_patient_candidates))
    print()

    pair_items = build_pair_items(
        same_patient_candidates,
        verifier,
    )
    print()

    trajectory_items = build_trajectory_items(
        trajectory_candidates,
        verifier,
    )
    print()

    cross_items = build_pair_items(
        cross_patient_candidates,
        verifier,
    )

    items = pair_items + trajectory_items + cross_items
    validate_final(items)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_patients = {
        str(source["patient_id"])
        for item in items
        for source in item["sources"]
    }
    all_admissions = {
        (str(source["patient_id"]), str(source["admission_id"]))
        for item in items
        for source in item["sources"]
    }

    print()
    print(f"Wrote {len(items)} items to {OUTPUT_PATH}")
    print(
        "Final composition:",
        {
            "same_patient_two_admissions": len(pair_items),
            "same_patient_three_admission_trajectory": len(trajectory_items),
            "cross_patient_two_admissions": len(cross_items),
        },
    )
    print(f"Patients covered: {len(all_patients)}")
    print(f"Admissions covered: {len(all_admissions)}")
    print(f"Grounding failures encountered: {len(verifier.failures)}")

if __name__ == "__main__":
    main()