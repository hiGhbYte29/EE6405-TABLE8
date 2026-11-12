"""Load the EE6405 raw dataset into modality-specific logical documents."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from ee6405.config import REPO_ROOT
from ee6405.types import Document

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv"}
_WEEK_RE = re.compile(r"(?:^|[^A-Za-z0-9])W(\d{1,2})(?:[^0-9]|$)|(?:^|[^A-Za-z0-9])(?<!pytest-)(\d{1,2})(?:[^0-9]|$)", re.I)


def extract_week(value: str | Path) -> str | None:
    text = str(value)
    explicit = re.search(r"(?<![A-Za-z0-9])W(\d{1,2})(?!\d)", text, re.I)
    if explicit:
        return f"W{int(explicit.group(1)):02d}"
    # Bare week values occur in structured Week: 03 headers only.
    bare = re.search(r"(?im)^\s*Week:\s*(\d{1,2})\b", text)
    return f"W{int(bare.group(1)):02d}" if bare else None


def _doc_id(path: Path, unit: str = "") -> str:
    return hashlib.sha1(f"{path.resolve()}::{unit}".encode()).hexdigest()[:16]


def _repo_path(path: str | Path) -> str:
    """Represent repository files with portable, workspace-relative paths."""
    original = Path(path)
    try:
        return original.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        portable = str(path).replace("\\", "/")
        marker = "/data/"
        if marker in portable:
            return portable[portable.index(marker) + 1:]
        return portable


def _doc(path: Path, text: str, modality: str, metadata: dict, unit: str = "") -> Document:
    return Document(_doc_id(path, unit), text.strip(), _repo_path(path), modality, metadata)


def _pdf_pages(path: Path) -> Iterator[tuple[int, str]]:
    from pypdf import PdfReader
    for page_no, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            yield page_no, text


def load_slides(path: str | Path) -> list[Document]:
    path = Path(path); week = extract_week(path); title = path.stem
    return [_doc(path, text, "slides", {"week": week, "source": _repo_path(path), "page": page,
        "doc_title": title, "locator": f"page:{page}"}, f"page:{page}") for page, text in _pdf_pages(path)]


def _clarification_records(text: str) -> list[dict[str, str | None]]:
    """Parse one record per Original Clarification with optional nearest header."""
    anchors = list(re.finditer(r"(?im)^\s*Original Clarification:\s*", text))
    headers = list(re.finditer(r"(?im)^\s*(?:Slide Reference|Topic):\s*.*$", text))
    records: list[dict[str, str | None]] = []
    for i, anchor in enumerate(anchors):
        prior = [h for h in headers if h.start() < anchor.start()]
        header = prior[-1] if prior else None
        start = header.start() if header else (0 if i == 0 else anchors[i - 1].end())
        following = [h for h in headers if h.start() > anchor.start()]
        end = following[0].start() if following else len(text)
        block = text[start:end]
        fields: dict[str, str | None] = {"week": None, "document_type": None,
            "related_material": None, "slide_page": None, "topic": None,
            "source_label": None, "text": ""}
        for key, value in re.findall(r"(?im)^\s*([A-Za-z ]+):\s*(.*?)\s*$", block):
            k = key.strip().lower().replace(" ", "_")
            if k == "slide_reference": fields["slide_page"] = value.strip()
            elif k in fields and k != "text": fields[k] = value.strip()
        body = re.search(r"(?is)Original Clarification:\s*(.*?)(?=\n\s*(?:Slide Reference|Topic):|$)", block)
        fields["text"] = body.group(1).strip() if body else ""
        week_match = re.search(r"(?im)^\s*Week:\s*(W?\d{1,2})\b", text)
        fields["week"] = extract_week(week_match.group(1)) if week_match else extract_week(text)
        records.append(fields)
    return records


def parse_clarifications(path: str | Path) -> list[Document]:
    path = Path(path); text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for idx, rec in enumerate(_clarification_records(text)):
        week = rec["week"] or extract_week(path)
        slide_page = rec["slide_page"]
        meta = {"week": week, "related_material": rec["related_material"],
                "slide_page": slide_page, "topic": rec.get("topic"),
                "document_type": rec["document_type"], "source": _repo_path(path),
                "locator": f"slide:{slide_page or rec.get('topic') or idx + 1}"}
        out.append(_doc(path, str(rec["text"]), "clarification", meta, f"record:{idx}"))
    return out


def load_notebook(path: str | Path) -> list[Document]:
    path = Path(path); data = json.loads(path.read_text(encoding="utf-8"))
    week = extract_week(path); variant = "Old" if re.search(r"_Old(?:\.|_)", path.name, re.I) else ("New" if re.search(r"_New(?:\.|_)", path.name, re.I) else None)
    cells = data.get("cells", []); out = []; i = 0
    while i < len(cells):
        cell = cells[i]; kind = cell.get("cell_type"); src = "".join(cell.get("source", [])).strip()
        if not src: i += 1; continue
        text = src; end = i
        if kind == "markdown" and i + 1 < len(cells) and cells[i+1].get("cell_type") == "code":
            code = "".join(cells[i+1].get("source", [])).strip()
            if code: text = f"{src}\n\n```python\n{code}\n```"; end = i + 1
        elif kind == "code": text = f"```python\n{src}\n```"
        meta = {"week": week, "source": _repo_path(path), "cell_idx": i, "variant": variant,
                "locator": f"cell:{i}"}
        out.append(_doc(path, text, "notebook", meta, f"cell:{i}")); i = end + 1
    return out


def _parse_mcqs(text: str) -> list[tuple[str, str, str]]:
    matches = list(re.finditer(r"(?m)^#\s*Q(\d+)\s*:\s*(.*?)(?=^#\s*Q\d+\s*:|\Z)", text, re.S))
    return [(m.group(1), m.group(2).strip(), m.group(0).strip()) for m in matches]


def parse_mcq(path: str | Path, answer_path: str | Path | None = None) -> list[Document]:
    path = Path(path); week = extract_week(path); part_m = re.search(r"Part\s*(\d+)", path.stem, re.I); part = part_m.group(1) if part_m else None
    answers: dict[str, str] = {}
    if answer_path and Path(answer_path).exists():
        for q, ans in re.findall(r"(?im)^\s*Q(\d+)\s*:\s*(\S+)", Path(answer_path).read_text(encoding="utf-8", errors="replace")):
            answers[q] = ans
    out = []
    for qid, _, block in _parse_mcqs(path.read_text(encoding="utf-8", errors="replace")):
        answer = answers.get(qid)
        text = f"{block}\n\nAnswer: {answer}" if answer is not None else block
        out.append(_doc(path, text, "mcq", {"week": week, "part": part, "qid": qid,
            "answer": answer, "source": _repo_path(path), "locator": f"question:{qid}"}, f"q:{qid}"))
    return out


def load_assessment_pdf(path: str | Path) -> list[Document]:
    path = Path(path); week = extract_week(path)
    return [_doc(path, text, "assessment_pdf", {"week": week, "source": _repo_path(path), "page": page,
        "locator": f"page:{page}"}, f"page:{page}") for page, text in _pdf_pages(path)]


def load_misc(path: str | Path) -> list[Document]:
    path = Path(path)
    return [_doc(path, text, "misc", {"source": _repo_path(path), "page": page, "locator": f"page:{page}"}, f"page:{page}") for page, text in _pdf_pages(path)]


def load_coding_quiz_ocr(path: str | Path) -> list[Document]:
    """Load OCR markdown for a coding quiz as one assessment document per page."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = list(re.finditer(r"(?m)^##\s+Page\s+(\d+)\s*$", text, re.I))
    out: list[Document] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        page_text = text[marker.end():end].strip()
        if not page_text:
            continue
        page = int(marker.group(1))
        metadata = {
            "week": None,
            "page": page,
            "source": _repo_path(path),
            "extraction": "ocr",
            "locator": f"page:{page}",
        }
        out.append(_doc(path, page_text, "assessment_pdf", metadata, f"page:{page}"))
    return out


def load_coding_quiz_ocr_tree(root: str | Path) -> Iterator[Document]:
    """Yield page documents from all OCR coding quiz markdown files."""
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.glob("*.md")) if root.is_dir() else [root]:
        yield from load_coding_quiz_ocr(path)


def _time_locator(seconds: float) -> str:
    """Return a stable video locator in ``t:mm:ss`` form."""
    whole = max(0, int(seconds))
    return f"t:{whole // 60:02d}:{whole % 60:02d}"


def _ends_sentence(text: str) -> bool:
    """Whether a segment ends at a likely sentence boundary."""
    return bool(re.search(r"[.!?][\"')\]]*$", text.strip()))


def _split_transcript_segment(segment: dict) -> list[dict]:
    """Split multi-sentence ASR segments without losing approximate timing."""
    text = str(segment.get("text_corrected", segment.get("text", ""))).strip()
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(parts) <= 1:
        return [segment]
    start = float(segment.get("start", 0.0)); end = float(segment.get("end", start))
    total = max(1, len(text))
    pieces: list[dict] = []
    offset = 0
    for part in parts:
        begin = start + (end - start) * offset / total
        offset += len(part) + 1
        finish = start + (end - start) * min(offset, total) / total
        pieces.append({**segment, "start": begin, "end": finish,
                       "text": part, "text_corrected": part})
    return pieces


def load_transcript(path: str | Path, chunk_size: int = 800) -> list[Document]:
    """Load a corrected Whisper JSON as retrieval-sized time windows.

    Segments are accumulated until the window is at least a minute and close
    to the configured character budget, then flushed at a segment sentence
    boundary.  A quiet/short section is capped at 90 seconds (also flushed at
    the next sentence boundary), so each document remains useful for video
    navigation while avoiding one-document-per-ASR-segment retrieval.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    week = extract_week(data.get("week") or path.name)
    lecture = data.get("lecture") or "L01"
    source_video = _repo_path(data.get("source")) if data.get("source") else _repo_path(path)
    segments = data.get("segments") or []
    out: list[Document] = []
    window: list[dict] = []
    chars = 0

    def flush() -> None:
        nonlocal window, chars
        if not window:
            return
        start = float(window[0].get("start", 0.0))
        end = float(window[-1].get("end", start))
        text = " ".join(str(s.get("text_corrected", s.get("text", ""))).strip() for s in window).strip()
        if text:
            unit = f"window:{len(out)}"
            metadata = {
                "week": week,
                "lecture": lecture,
                "start": start,
                "end": end,
                "source": _repo_path(path),
                "source_video": source_video,
                "locator": _time_locator(start),
            }
            out.append(_doc(path, text, "transcript", metadata, unit))
        window = []
        chars = 0

    for segment in segments:
        for piece in _split_transcript_segment(segment):
            corrected = str(piece.get("text_corrected", piece.get("text", ""))).strip()
            if not corrected:
                continue
            window.append(piece)
            chars += len(corrected) + (1 if len(window) > 1 else 0)
            elapsed = float(piece.get("end", 0.0)) - float(window[0].get("start", 0.0))
            # Keep a 60--90 second target where possible.  The sentence check
            # prevents a sentence from being split merely at a time threshold.
            ready = elapsed >= 60 and chars >= chunk_size
            hard_limit = elapsed >= 90
            if (ready or hard_limit) and _ends_sentence(corrected):
                flush()
    flush()
    return out


def load_transcript_tree(root: str | Path, chunk_size: int = 800) -> Iterator[Document]:
    """Yield corrected transcript windows from a directory of JSON files."""
    root = Path(root)
    if not root.exists():
        return
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    for path in files:
        yield from load_transcript(path, chunk_size=chunk_size)


def load_tree(root: str | Path, modalities: set[str] | None = None) -> Iterator[Document]:
    root = Path(root)
    if not root.exists(): raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not any(x in SKIP_DIRS for x in p.parts))
    answer_by_stem = {p.stem.replace("_Answers", "_MCQ"): p for p in files if p.name.endswith("_Practice_Answers.txt")}
    for path in files:
        s = path.suffix.lower(); parts = str(path).replace("\\", "/")
        if "/02_Lecture_Videos/" in parts or s in {".mp4", ".vtt", ".srt"}: continue
        if "/01_Lecture_Slides" in parts and s == ".txt": docs = parse_clarifications(path); mod = "clarification"
        elif "/01_Lecture_Slides" in parts and s == ".pdf": docs = load_slides(path); mod = "slides"
        elif "/03_Code/" in parts and s == ".ipynb": docs = load_notebook(path); mod = "notebook"
        elif "/04_Assessments/01_Practice/" in parts and s == ".md":
            docs = parse_mcq(path, answer_by_stem.get(path.stem)); mod = "mcq"
        elif "/04_Assessments/" in parts and s == ".pdf": docs = load_assessment_pdf(path); mod = "assessment_pdf"
        elif "/05_Miscellaneous/" in parts and s == ".pdf": docs = load_misc(path); mod = "misc"
        else: continue
        if modalities is None or mod in modalities:
            yield from docs

def load_file(path: str | Path) -> Document | None:
    """Compatibility helper: return the first logical unit from a supported file."""
    path = Path(path); parts = str(path).replace("\\", "/"); docs: list[Document] = []
    if path.suffix.lower() == ".pdf":
        docs = load_misc(path) if "/05_Miscellaneous/" in parts else (load_assessment_pdf(path) if "/04_Assessments/" in parts else load_slides(path))
    elif path.suffix.lower() == ".ipynb": docs = load_notebook(path)
    elif path.suffix.lower() == ".md": docs = parse_mcq(path)
    elif path.suffix.lower() == ".txt": docs = parse_clarifications(path)
    return docs[0] if docs else None
