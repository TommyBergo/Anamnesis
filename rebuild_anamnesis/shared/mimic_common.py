"""Date-arithmetic helpers (age at admission, length of stay) for MIMIC-IV's anchor-year date shifting."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# MIMIC-IV top-codes every patient older than 89 at their anchor year to an anchor_age of 91.
MIMIC_IV_TOP_CODED_AGE = 91


def compute_age_at_admission(anchor_age: object, anchor_year: object, admittime: object) -> Optional[int]:
    try:
        anchor_age_int = int(float(str(anchor_age)))
        anchor_year_int = int(float(str(anchor_year)))
        admit_year = int(str(admittime)[:4])
    except (ValueError, TypeError):
        return None
    if anchor_age_int >= MIMIC_IV_TOP_CODED_AGE:
        return MIMIC_IV_TOP_CODED_AGE
    age = anchor_age_int + (admit_year - anchor_year_int)
    return age if age >= 0 else None


def compute_los_days(admittime: object, dischtime: object) -> Optional[int]:
    try:
        t_in = datetime.strptime(str(admittime).split(" ")[0], "%Y-%m-%d")
        t_out = datetime.strptime(str(dischtime).split(" ")[0], "%Y-%m-%d")
    except ValueError:
        return None
    days = (t_out - t_in).days
    return days if days >= 0 else None
