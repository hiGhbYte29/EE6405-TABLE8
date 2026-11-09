#!/usr/bin/env python3
"""Transcribe the scanned EE6405 coding quizzes with a vision model."""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROMPT = """Transcribe this page exactly and completely. Return only the transcription, with no explanation and do not answer any question.

- Include every visible word, heading, label, number, punctuation mark, and question choice.
- Preserve the page's question structure (Q1, Q2, subparts, etc.).
- Put every code block inside a Markdown fenced code block (```), preserving the original spelling, punctuation, line breaks, and Python indentation exactly. Do not reformat or infer missing code.
- Preserve mathematical notation and symbols as faithfully as possible.
- If the page is a chart or other figure with no readable text, briefly describe its visible content instead.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/EE6405_Raw_Dataset/04_Assessments/02_Coding Quiz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/coding_quiz_ocr"))
    parser.add_argument("--model", default=os.getenv("EE6405_LLM_MODEL", "claude-sonnet-5"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry-model", default="claude-opus-5")
    return parser.parse_args()


def render_pages(pdf: Path, temp_dir: Path, dpi: int) -> list[Path]:
    prefix = temp_dir / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return sorted(temp_dir.glob("page-*.png"))


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def transcribe(client: OpenAI, image: Path, model: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
                ],
            }
        ],
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError(f"The vision model returned an empty transcription for {image.name}")
    return text.strip()


def looks_bad(text: str) -> bool:
    lowered = text.lower()
    refusal_markers = ("i can't see", "i cannot see", "unable to read", "too blurry", "i'm unable")
    return len(text.strip()) < 10 or any(marker in lowered for marker in refusal_markers)


def main() -> int:
    args = parse_args()
    load_dotenv()
    base_url = os.environ.get("EE6405_LLM_BASE_URL")
    api_key = os.environ.get("EE6405_LLM_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("EE6405_LLM_BASE_URL and EE6405_LLM_API_KEY must be set in .env")

    client = OpenAI(base_url=base_url, api_key=api_key)
    pdfs = [args.input_dir / f"Coding Quiz {n}.pdf" for n in (1, 2, 3)]
    missing = [str(path) for path in pdfs if not path.is_file()]
    if missing:
        raise SystemExit("Missing source PDF(s): " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"model": args.model, "dpi": args.dpi, "files": []}

    with tempfile.TemporaryDirectory(prefix="ee6405-coding-quiz-") as temp_root:
        for pdf in pdfs:
            pdf_temp = Path(temp_root) / pdf.stem.replace(" ", "_")
            pdf_temp.mkdir()
            pages = render_pages(pdf, pdf_temp, args.dpi)
            page_rows: list[dict[str, object]] = []
            page_texts: list[str] = []
            for page_number, image in enumerate(pages, start=1):
                model_used = args.model
                text = transcribe(client, image, model_used, args.max_tokens)
                if looks_bad(text) and args.retry_model and args.retry_model != model_used:
                    model_used = args.retry_model
                    text = transcribe(client, image, model_used, args.max_tokens)
                page_texts.append(text)
                page_rows.append(
                    {"page": page_number, "characters": len(text), "model": model_used}
                )
            output_text = "\n\n".join(
                f"## Page {page_number}\n\n{text}" for page_number, text in enumerate(page_texts, start=1)
            )
            output_path = args.output_dir / f"{pdf.stem}.md"
            output_path.write_text(output_text + "\n", encoding="utf-8")
            manifest["files"].append(
                {
                    "source": str(pdf),
                    "output": str(output_path),
                    "pages": len(pages),
                    "characters": sum(len(text) for text in page_texts),
                    "page_details": page_rows,
                }
            )
            print(f"{pdf.name}: {len(pages)} pages, {sum(len(text) for text in page_texts)} characters")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
