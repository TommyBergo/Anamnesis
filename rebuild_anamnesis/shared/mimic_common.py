"""Date-arithmetic helpers (age at admission, length of stay) shared by the extraction and PDF-rendering scripts."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

MIMIC_90_PLUS_SENTINEL_AGE = 90
MIMIC_90_PLUS_RAW_THRESHOLD = 150

def compute_age(dob: str, admittime: str) -> Optional[int]:
    try:
        raw_age = int(str(admittime)[:4]) - int(str(dob)[:4])
    except (ValueError, IndexError):
        return None
    return MIMIC_90_PLUS_SENTINEL_AGE if raw_age >= MIMIC_90_PLUS_RAW_THRESHOLD else raw_age

def compute_los_days(admittime: str, dischtime: str) -> Optional[int]:
    try:
        t_in = datetime.strptime(str(admittime).split(" ")[0], "%Y-%m-%d")
        t_out = datetime.strptime(str(dischtime).split(" ")[0], "%Y-%m-%d")
        return (t_out - t_in).days
    except ValueError:
        return None
