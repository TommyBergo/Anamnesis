"""Splits generate_qa_multidoc.py's combined output into same-patient and cross-patient files."""

import json

from shared.pipeline_paths import (
    MULTIDOC_CROSS_PATIENT_PATH as CROSS_PATIENT_OUTPUT,
    MULTIDOC_QA_PATH as INPUT_PATH,
    MULTIDOC_SAME_PATIENT_PATH as SAME_PATIENT_OUTPUT,
)

SAME_PATIENT_TYPES = {"same_patient_two_admissions", "same_patient_three_admission_trajectory"}


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH.name} not found - run generate_qa_multidoc.py first.")

    same_patient, cross_patient, other = [], [], []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ctype = item.get("comparison_type")
            if ctype in SAME_PATIENT_TYPES:
                same_patient.append(item)
            elif ctype == "cross_patient_two_admissions":
                cross_patient.append(item)
            else:
                other.append(item)

    if other:
        raise SystemExit(
            f"{len(other)} item(s) had an unrecognized comparison_type - filter criteria may be "
            f"stale. First one: {other[0].get('id')} / {other[0].get('comparison_type')}"
        )

    with SAME_PATIENT_OUTPUT.open("w", encoding="utf-8") as f:
        for item in same_patient:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with CROSS_PATIENT_OUTPUT.open("w", encoding="utf-8") as f:
        for item in cross_patient:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"same-patient ({SAME_PATIENT_OUTPUT.name}): {len(same_patient)} items")
    print(f"cross-patient ({CROSS_PATIENT_OUTPUT.name}): {len(cross_patient)} items")


if __name__ == "__main__":
    main()