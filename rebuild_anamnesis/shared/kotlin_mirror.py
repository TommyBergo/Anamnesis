"""Line-for-line Python port of the Android app's ClinicalPdfProcessor.createClinicalChunks (Kotlin), so retrieval evaluation uses identical chunk boundaries to the on-device app."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE_RUN = re.compile(r"[ \t]+")

def clean_text(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip()

@dataclass
class Chunk:
    """One character-offset span of chunked document text."""

    text: str
    start: int
    end: int

def create_clinical_chunks(text: str, max_chars: int = 500, overlap_chars: int = 100) -> list[Chunk]:
    if not text.strip():
        return []

    clean = clean_text(text)
    result: list[Chunk] = []
    start_index = 0
    n = len(clean)

    while start_index < n:
        end_index = start_index + max_chars
        if end_index >= n:
            last_chunk = clean[start_index:].strip()
            if last_chunk:

                lstrip_amount = len(clean[start_index:]) - len(clean[start_index:].lstrip())
                result.append(Chunk(last_chunk, start_index + lstrip_amount, n))
            break

        cut_index = end_index

        last_paragraph = clean.rfind("\n\n", 0, min(end_index + 2, n))
        if last_paragraph > start_index + (max_chars // 2):
            cut_index = last_paragraph + 2
        else:
            last_line = clean.rfind("\n", 0, min(end_index + 1, n))
            if last_line > start_index + (max_chars // 2):
                cut_index = last_line + 1
            else:
                last_period = clean.rfind(". ", 0, min(end_index + 2, n))
                if last_period > start_index + (max_chars // 2):
                    cut_index = last_period + 2
                else:
                    last_space = clean.rfind(" ", 0, min(end_index + 1, n))
                    if last_space > start_index:
                        cut_index = last_space + 1

        if cut_index <= start_index:
            cut_index = end_index

        raw_chunk = clean[start_index:cut_index]
        chunk_text = raw_chunk.strip()
        if chunk_text:
            lstrip_amount = len(raw_chunk) - len(raw_chunk.lstrip())
            rstrip_amount = len(raw_chunk) - len(raw_chunk.rstrip())
            result.append(Chunk(chunk_text, start_index + lstrip_amount, cut_index - rstrip_amount))

        next_start_index = cut_index - overlap_chars
        if next_start_index > start_index:
            next_space = clean.find(" ", next_start_index, cut_index)
            if next_space != -1:
                start_index = next_space + 1
            else:
                start_index = next_start_index
        else:
            start_index = cut_index

    return result
