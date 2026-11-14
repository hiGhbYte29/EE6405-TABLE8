"""Open the original file behind a citation, at the cited place.

Each helper returns ``None`` when the raw file (or the library to render it) is
missing, so the app can fall back to the chunk's own text. The prebuilt bundle
has the index and the transcripts but not the raw slides, videos or notebooks.
"""

from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ee6405.config import REPO_ROOT
from ee6405.types import Chunk

RAW_ROOT = REPO_ROOT / "data" / "raw" / "EE6405_Raw_Dataset"
VIDEO_DIR = RAW_ROOT / "02_Lecture_Videos"


@dataclass(frozen=True)
class Kind:
    label: str
    color: str  # st.badge colour


KINDS = {
    "slides": Kind("课件", "blue"),
    "transcript": Kind("视频转写", "orange"),
    "notebook": Kind("代码", "green"),
    "mcq": Kind("练习题", "violet"),
    "assessment_pdf": Kind("测验", "violet"),
    "misc": Kind("课程文件", "gray"),
    "clarification": Kind("讲师澄清", "gray"),
}


def kind_of(chunk: Chunk) -> Kind:
    return KINDS.get(chunk.modality, Kind(chunk.modality, "gray"))


def resolve(rel: str | None) -> Path | None:
    """Repo-relative path from the index -> existing local file, else None."""
    if not rel:
        return None
    p = Path(rel)
    p = p if p.is_absolute() else REPO_ROOT / p
    return p if p.is_file() else None


QUIZ_PDF_DIR = RAW_ROOT / "04_Assessments" / "02_Coding Quiz"


def original_file(chunk: Chunk) -> Path | None:
    """The file to open for a citation, else None.

    Coding quizzes are scanned PDFs, so the index holds their OCR text
    (``coding_quiz_ocr/Coding Quiz 4.md``); the thing to show is the PDF itself.
    """
    if chunk.metadata.get("extraction") == "ocr":
        pdf = QUIZ_PDF_DIR / f"{Path(chunk.source).stem}.pdf"
        if pdf.is_file():
            return pdf
    return resolve(chunk.source)


def display_name(chunk: Chunk) -> str:
    if chunk.modality == "transcript":
        lec = str(chunk.metadata.get("lecture") or "")
        lec = f" Lecture {lec.lstrip('L').lstrip('0')}" if lec.startswith("L") else ""
        return f"{chunk.week or ''}{lec} 视频".strip()
    return Path(chunk.source).name


def location_label(chunk: Chunk) -> str:
    loc = chunk.locator or ""
    kind, _, val = loc.partition(":")
    if kind == "page":
        return f"第 {val} 页"
    if kind == "t":
        return val
    if kind == "cell":
        return f"cell {val}"
    if kind == "q":
        return f"Q{val}"
    return loc


# --------------------------------------------------------------------- PDF


def pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def pdf_page_png(path: Path, page: int, scale: float = 1.6) -> bytes | None:
    """Render one 1-based page to PNG bytes, or None if pypdfium2 is missing."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    pdf = pdfium.PdfDocument(str(path))
    try:
        if not 1 <= page <= len(pdf):
            return None
        image = pdf[page - 1].render(scale=scale).to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


# ------------------------------------------------------------------- video


def video_path(chunk: Chunk) -> Path | None:
    rel = chunk.metadata.get("source_video")
    if not rel:
        return None
    p = VIDEO_DIR / rel
    return p if p.is_file() else None


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def transcript_window(chunk: Chunk, before: float = 15.0, after: float = 8.0):
    """Segments around the cited window as ``(mm:ss, seconds, text, cited)``.

    Reads the corrected transcript JSON the chunk came from (it ships in the
    bundle), so this works without the raw videos.
    """
    path = resolve(chunk.source)
    start = float(chunk.metadata.get("start", 0) or 0)
    end = float(chunk.metadata.get("end", start) or start)
    if path is None or path.suffix != ".json":
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for seg in data.get("segments", []):
        s = float(seg.get("start", 0))
        if start - before <= s <= end + after:
            text = (seg.get("text_corrected") or seg.get("text") or "").strip()
            rows.append((_mmss(s), s, text, start <= s < end))
    return rows


# ---------------------------------------------------------------- notebook

_NB_CSS = """
body{margin:0;font-family:Inter,'Noto Sans SC',system-ui,sans-serif;color:#1F2328;background:#FFFFFF}
.cell{margin:0 4px 12px;border:1px solid #E1E4E8;border-radius:8px;overflow:hidden}
.cell.cited{border:2px solid #2B59C3;box-shadow:0 0 0 3px rgba(43,89,195,.15)}
.bar{display:flex;justify-content:space-between;padding:4px 10px;background:#F6F8FA;
 border-bottom:1px solid #E1E4E8;font:12px ui-monospace,monospace;color:#59636E}
.cited .bar .tag{color:#1F4296;font-weight:600}
.md{padding:4px 14px;font-size:14px;line-height:1.6}
.md pre,.md code{font-family:ui-monospace,monospace;font-size:12px}
.src{margin:0;padding:10px 14px;font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.out{margin:0;padding:8px 14px;border-top:1px dashed #E1E4E8;background:#FBFBFC;
 font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;color:#3D444D;max-height:240px;overflow:auto}
.out img{max-width:100%}
"""


_B64 = re.compile(r"[A-Za-z0-9+/=\s]+")


def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _outputs_html(cell: dict, limit: int = 2000) -> str:
    parts = []
    for out in cell.get("outputs", []) or []:
        data = out.get("data", {}) or {}
        if "image/png" in data:
            png = data["image/png"]
            png = ("".join(png) if isinstance(png, list) else str(png)).strip()
            # The page runs in a same-origin iframe: only real base64 may reach the attribute.
            if _B64.fullmatch(png):
                parts.append(f'<img alt="cell output" src="data:image/png;base64,{png}">')
            continue
        # text/html outputs are skipped on purpose: render plain text only.
        text = out.get("text") or data.get("text/plain") or ""
        text = "".join(text) if isinstance(text, list) else str(text)
        if text.strip():
            parts.append(html.escape(text[:limit] + ("…" if len(text) > limit else "")))
    return f'<div class="out">{"<br>".join(parts)}</div>' if parts else ""


def cited_cells(cells: list[dict], cell_idx: int) -> set[int]:
    """The loader joins a markdown cell with the code cell right after it."""
    hit = {cell_idx}
    if (
        0 <= cell_idx < len(cells) - 1
        and cells[cell_idx].get("cell_type") == "markdown"
        and cells[cell_idx + 1].get("cell_type") == "code"
    ):
        hit.add(cell_idx + 1)
    return hit


def notebook_html(path: Path, cell_idx: int) -> str:
    """Whole notebook as one HTML page that scrolls to the cited cell."""
    from markdown_it import MarkdownIt
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    md = MarkdownIt("commonmark", {"html": False})
    fmt = HtmlFormatter(nowrap=True, style="default")
    hit = cited_cells(cells, cell_idx)
    first = min(hit)
    body = []
    for i, cell in enumerate(cells):
        text = _cell_text(cell)
        if not text.strip() and not cell.get("outputs"):
            continue
        cited = i in hit
        tag = "被引用" if cited else ""
        anchor = ' id="cited"' if i == first else ""
        if cell.get("cell_type") == "markdown":
            inner = f'<div class="md">{md.render(text)}</div>'
        else:
            code = highlight(text, PythonLexer(), fmt)
            inner = f'<pre class="src">{code}</pre>' + _outputs_html(cell)
        body.append(
            f'<div class="cell{" cited" if cited else ""}"{anchor}>'
            f'<div class="bar"><span>cell {i}</span><span class="tag">{tag}</span></div>'
            f"{inner}</div>"
        )
    css = _NB_CSS + fmt.get_style_defs(".src")
    script = (
        "<script>var c=document.getElementById('cited');"
        "if(c){c.scrollIntoView({block:'start'});}</script>"
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(body)}{script}</body></html>"


# --------------------------------------------------------------- citations

# Code spans are split off first, so `a[0][1]` in code is never touched here.
_CITE = re.compile(r"(?<!\w)\[(\d{1,2})\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def badge_citations(text: str, n_hits: int) -> str:
    """Turn ``[2]`` in an answer into a Streamlit badge, outside code only."""

    def sub(m: re.Match) -> str:
        n = int(m.group(1))
        # Leading space keeps adjacent badges ([2][3]) apart.
        return f" :blue-badge[{n}]" if 1 <= n <= n_hits else m.group(0)

    parts = _CODE.split(text)
    return "".join(p if i % 2 else _CITE.sub(sub, p) for i, p in enumerate(parts))

