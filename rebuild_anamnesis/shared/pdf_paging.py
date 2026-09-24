"""Extracts PDF text per-page and tracks character offsets, so chunks can be traced back to the page(s) they came from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from shared.kotlin_mirror import Chunk, clean_text, create_clinical_chunks

PAGE_JOIN_SEP = "\n"

@dataclass
class PagedDocument:
    """A PDF's cleaned full text alongside the character range each source page maps to."""

    pdf_path: Path
    num_pages: int
    clean_text: str

    page_ranges: list[tuple[int, int]]

def extract_paged_document(pdf_path: Path) -> PagedDocument:
    reader = PdfReader(str(pdf_path))
    raw_pages = [page.extract_text() or "" for page in reader.pages]

    cleaned_pages = [clean_text(p) for p in raw_pages]

    ranges: list[tuple[int, int]] = []
    cursor = 0
    parts: list[str] = []
    for i, cp in enumerate(cleaned_pages):
        if i > 0:
            parts.append(PAGE_JOIN_SEP)
            cursor += len(PAGE_JOIN_SEP)
        start = cursor
        parts.append(cp)
        cursor += len(cp)
        ranges.append((start, cursor))

    full_clean = "".join(parts)

    leading_ws = len(full_clean) - len(full_clean.lstrip())
    if leading_ws:
        full_clean = full_clean[leading_ws:]
        ranges = [(max(0, s - leading_ws), max(0, e - leading_ws)) for s, e in ranges]
    full_clean = full_clean.rstrip()

    return PagedDocument(
        pdf_path=pdf_path,
        num_pages=len(raw_pages),
        clean_text=full_clean,
        page_ranges=ranges,
    )

def _offset_range_to_pages(doc: PagedDocument, start: int, end: int) -> list[int]:
    pages = []
    for i, (p_start, p_end) in enumerate(doc.page_ranges):
        if start < p_end and end > p_start:
            pages.append(i + 1)
    return pages or [doc.num_pages]

def pages_for_chunk(doc: PagedDocument, chunk: Chunk) -> list[int]:
    return _offset_range_to_pages(doc, chunk.start, chunk.end)

def _flatten_with_offset_map(text: str) -> tuple[str, list[int]]:
    out_chars: list[str] = []
    mapping: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_was_space:
                out_chars.append(" ")
                mapping.append(i)
            prev_was_space = True
        else:
            out_chars.append(ch.lower())
            mapping.append(i)
            prev_was_space = False
    return "".join(out_chars), mapping

def find_pages_for_text(doc: PagedDocument, snippet: str, max_len: int = 300) -> tuple[int | None, list[int]]:
    flat_doc, mapping = _flatten_with_offset_map(doc.clean_text)
    truncated = snippet[:max_len] if len(snippet) > max_len else snippet

    if truncated.endswith("..."):
        truncated = truncated[:-3].rstrip()
    flat_target, _ = _flatten_with_offset_map(truncated)
    if not flat_target:
        return None, []

    pos = flat_doc.find(flat_target)
    if pos == -1:
        return None, []

    start_offset = mapping[pos]
    end_offset = mapping[pos + len(flat_target) - 1] + 1
    pages = _offset_range_to_pages(doc, start_offset, end_offset)
    return pages[0], pages

def page_for_offset(doc: PagedDocument, offset: int) -> int:
    offset = max(0, min(offset, len(doc.clean_text) - 1))
    for i, (start, end) in enumerate(doc.page_ranges):
        if start <= offset < end:
            return i + 1
    return doc.num_pages

def chunks_with_pages(doc: PagedDocument, max_chars: int = 500, overlap_chars: int = 100):
    chunks = create_clinical_chunks(doc.clean_text, max_chars=max_chars, overlap_chars=overlap_chars)
    annotated = []
    for c in chunks:
        pages = pages_for_chunk(doc, c)
        annotated.append({
            "text": c.text,
            "start": c.start,
            "end": c.end,
            "primary_page": pages[0],
            "pages": pages,
        })
    return annotated
