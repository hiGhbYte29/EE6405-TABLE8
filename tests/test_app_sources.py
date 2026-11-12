import json

import pytest

from ee6405.app import sources as src
from ee6405.config import REPO_ROOT
from ee6405.types import Chunk

BRIEF_PDF = REPO_ROOT / "Group project details" / "Group project details.pdf"


def _chunk(modality, source, locator="", **meta):
    return Chunk("c", "d", "text", str(source), modality, 0, meta, locator, meta.get("week"))


def test_badges_only_valid_citations_outside_code():
    text = "NSP is removed [2][3]. See `x[1]` and [9].\n```python\na[2]\n```"
    out = src.badge_citations(text, n_hits=5)
    assert ":blue-badge[2]" in out and ":blue-badge[3]" in out
    assert "`x[1]`" in out and "a[2]" in out  # code untouched
    assert "[9]" in out and ":blue-badge[9]" not in out  # no such hit


def test_markdown_cell_is_cited_with_its_code_cell():
    cells = [{"cell_type": "markdown"}, {"cell_type": "code"}, {"cell_type": "code"}]
    assert src.cited_cells(cells, 0) == {0, 1}
    assert src.cited_cells(cells, 2) == {2}


def test_notebook_html_scrolls_to_cited_cell_and_skips_html_outputs(tmp_path):
    nb = {"cells": [
        {"cell_type": "code", "source": ["import torch"], "outputs": []},
        {"cell_type": "markdown", "source": ["### 1. Use BERT"]},
        {"cell_type": "code", "source": ["print('hi')"], "outputs": [
            {"output_type": "stream", "text": ["hi\n"]},
            {"output_type": "display_data", "data": {"text/html": "<script>alert(1)</script>", "text/plain": "table"}},
        ]},
    ]}
    p = tmp_path / "W06_Code.ipynb"
    p.write_text(json.dumps(nb))
    page = src.notebook_html(p, 1)
    assert page.count('id="cited"') == 1
    assert page.index('id="cited"') < page.index("Use BERT")
    assert page.count('class="cell cited"') == 2  # markdown + its code cell
    assert "alert(1)" not in page and "table" in page and "hi" in page


def test_notebook_html_escapes_markdown_html_and_bad_images(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["<img src=x onerror=alert(1)> **bold**"]},
        {"cell_type": "code", "source": ["</pre><script>alert(2)</script>"], "outputs": [
            {"output_type": "display_data", "data": {"image/png": 'AAA" onerror="alert(3)'}},
            {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo="}},
        ]},
    ]}
    p = tmp_path / "evil.ipynb"
    p.write_text(json.dumps(nb))
    page = src.notebook_html(p, 0)
    assert "<img src=x" not in page and "<strong>bold</strong>" in page
    assert "<script>alert(2)" not in page
    assert 'onerror="alert(3)' not in page
    assert page.count("<img ") == 1  # the valid base64 image is kept


def test_transcript_window_marks_the_cited_segments(tmp_path):
    p = tmp_path / "W06_L02_Video.json"
    segs = [{"start": t, "end": t + 3, "text": f"raw{t}", "text_corrected": f"seg{t}"} for t in range(770, 830, 5)]
    p.write_text(json.dumps({"segments": segs}))
    c = _chunk("transcript", p, "t:13:06", start=786, end=800)
    rows = src.transcript_window(c, before=10, after=5)
    cited = [r for r in rows if r[3]]
    assert [r[2] for r in cited] == ["seg790", "seg795"]
    assert rows[0][0] == "13:00" and all(not r[2].startswith("raw") for r in rows)


def test_missing_raw_files_return_none():
    c = _chunk("transcript", "data/interim/none.json", source_video="W99/nope.mp4")
    assert src.video_path(c) is None
    assert src.resolve("data/raw/EE6405_Raw_Dataset/nope.pdf") is None
    assert src.resolve(None) is None


@pytest.mark.skipif(not BRIEF_PDF.exists(), reason="course brief PDF not present")
def test_pdf_page_renders_to_png():
    pytest.importorskip("pypdfium2")
    png = src.pdf_page_png(BRIEF_PDF, 1, scale=0.5)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert src.pdf_page_png(BRIEF_PDF, 10_000) is None
    assert src.pdf_page_count(BRIEF_PDF) >= 1


def test_labels():
    c = _chunk("slides", "x/W06_Padlet.pdf", "page:9", week="W06")
    assert src.location_label(c) == "第 9 页" and src.display_name(c) == "W06_Padlet.pdf"
    v = _chunk("transcript", "x.json", "t:13:06", week="W06", lecture="L02")
    assert src.display_name(v) == "W06 Lecture 2 视频" and src.location_label(v) == "13:06"
