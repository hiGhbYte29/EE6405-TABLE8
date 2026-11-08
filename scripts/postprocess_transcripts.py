#!/usr/bin/env python3
"""Audit and context-aware correct common ASR terminology errors.

Original transcript files are never modified. Corrected JSON/TXT files and audit
reports are written beneath ``--output-dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

TERMS = [
    "BERT", "RoBERTa", "ALBERT", "DistilBERT", "ELMo", "FastText", "GloVe", "ChatGPT",
    "ELIZA", "Weizenbaum", "DALL-E", "Midjourney", "RMSprop",
    "word2vec", "seq2seq", "LSTM", "GRU", "CRF", "GCN", "GAT", "TF-IDF",
    "BLEU", "ROUGE", "METEOR", "perplexity", "n-gram", "softmax", "logits",
    "embedding", "tokenization", "lemmatization", "POS", "NER", "RAG",
]

# Suspect forms are deliberately conservative. A form is only corrected when
# its context rule below says that it is an NLP term rather than ordinary text.
SUSPECTS = {
    "RoBERTa": re.compile(r"\brobota(?P<suffix>s|['’]s)?\b", re.I),
    "ChatGPT": re.compile(r"\bchatgdp(?P<suffix>s|['’]s)?\b", re.I),
    "ELIZA": re.compile(r"\belisa\b", re.I),
    "Weizenbaum": re.compile(r"\bwiesenbaum\b", re.I),
    "DALL-E": re.compile(r"\bdali\b", re.I),
    "Midjourney": re.compile(r"\bmeet[ -]+journey\b", re.I),
    "RMSprop": re.compile(r"\brms\s+prop\b", re.I),
    "BLEU": re.compile(r"\bblue(?P<suffix>s)?\b", re.I),
}
BLEU_CONTEXT = re.compile(
    r"\b(score|metric|metrics|evaluation|evaluate|translation|candidate|reference|rouge|meteor|bleu)\b",
    re.I,
)

# Counts are case-insensitive because retrieval can casefold terms. Plural and
# possessive forms are included so an ASR-produced ``LSTM's`` is not omitted.
TERM_PATTERNS = {
    "BERT": r"bert(?:s|['’]s)?",
    "RoBERTa": r"roberta(?:['’]s)?",
    "ChatGPT": r"chatgpt(?:s|['’]s)?",
    "ELIZA": r"eliza",
    "Weizenbaum": r"weizenbaum",
    "DALL-E": r"dall[-·]e",
    "Midjourney": r"midjourney",
    "RMSprop": r"RMSprop",
    "ALBERT": r"albert(?:s|['’]s)?",
    "DistilBERT": r"distilbert(?:s|['’]s)?",
    "ELMo": r"elmo(?:s|['’]s)?",
    "FastText": r"fasttext(?:s|['’]s)?",
    "GloVe": r"glove(?:s|['’]s)?",
    "word2vec": r"word2vec(?:s|['’]s)?",
    "seq2seq": r"seq2seq(?:s|['’]s)?",
    "LSTM": r"lstms?(?:['’]s)?",
    "GRU": r"grus?(?:['’]s)?",
    "CRF": r"crfs?(?:['’]s)?",
    "GCN": r"gcns?(?:['’]s)?",
    "GAT": r"gats?(?:['’]s)?",
    "TF-IDF": r"tf[- ]?idfs?(?:['’]s)?",
    "BLEU": r"bleus?(?:['’]s)?",
    "ROUGE": r"rouges?(?:['’]s)?",
    "METEOR": r"meteors?(?:['’]s)?",
    "perplexity": r"perplexit(?:y|ies)(?:['’]s)?",
    "n-gram": r"n-grams?(?:['’]s)?",
    "softmax": r"softmax(?:es|s)?(?:['’]s)?",
    "logits": r"logits?(?:['’]s)?",
    "embedding": r"embeddings?(?:['’]s)?",
    "tokenization": r"tokenizations?(?:['’]s)?",
    "lemmatization": r"lemmatizations?(?:['’]s)?",
    "POS": r"poss?(?:['’]s)?",
    "NER": r"ners?(?:['’]s)?",
    "RAG": r"rags?(?:['’]s)?",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("data/interim/transcripts"))
    p.add_argument("--output-dir", type=Path, default=Path("data/interim/transcripts_corrected"))
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--min-hallucination-run", type=int, default=3)
    p.add_argument("--reference-dir", type=Path, default=Path("data/raw/EE6405_Raw_Dataset"))
    return p.parse_args()


def canonical_count(text: str, term: str) -> int:
    body = TERM_PATTERNS[term]
    # Keep terms standalone while allowing plural and possessive variants.
    flags = 0 if term == "RMSprop" else re.I
    return len(re.findall(rf"(?<![\w-])(?:{body})(?![\w-])", text, flags))


def correction_for(segment_text: str, week: str, nearby_text: str) -> tuple[str, list[tuple[str, str, str]]]:
    corrected = segment_text
    changes: list[tuple[str, str, str]] = []
    safe_replacements = [
        ("ELIZA", "ELIZA"), ("Weizenbaum", "Weizenbaum"), ("DALL-E", "DALL-E"),
        ("Midjourney", "Midjourney"), ("RMSprop", "RMSprop"),
    ]
    for term, replacement in safe_replacements:
        before = corrected
        corrected, n = SUSPECTS[term].subn(replacement, corrected)
        if n:
            changes.append((term, before, corrected))

    def repl_roberta(match: re.Match[str]) -> str:
        return "RoBERTa" + (match.group("suffix") or "")
    corrected, n = SUSPECTS["RoBERTa"].subn(repl_roberta, corrected)
    if n:
        changes.append(("RoBERTa", segment_text, corrected))

    def repl_chatgpt(match: re.Match[str]) -> str:
        return "ChatGPT" + (match.group("suffix") or "")
    before_chatgpt = corrected
    corrected, n = SUSPECTS["ChatGPT"].subn(repl_chatgpt, corrected)
    if n:
        changes.append(("ChatGPT", before_chatgpt, corrected))

    # BLEU is corrected only when the sentence/window clearly describes an
    # evaluation metric or machine-translation context. Week W04 is an
    # additional prior because it is the recorded evaluation-metrics lecture,
    # but the context keyword guard still has to match.
    if BLEU_CONTEXT.search(f"{segment_text} {nearby_text}"):
        before = corrected
        corrected = SUSPECTS["BLEU"].sub(lambda match: "BLEU" + (match.group("suffix") or ""), corrected)
        if corrected != before:
            changes.append(("BLEU", before, corrected))
    return corrected, changes


COMMON_UPPERCASE = {
    "AI", "NLP", "AGI", "IDF", "GPT", "RNN", "NER", "POS", "LDA", "LSA",
    "NMF", "SVD", "PCA", "SOP", "Turing", "OpenAI", "ML", "API", "ROC",
    "TF", "GP", "AR", "XL", "AE", "EE6405", "NTU", "SVM", "BM25",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?:['’]s)?")


def discover_uppercase_tokens(files: list[Path]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for source_path in files:
        data = json.loads(source_path.read_text(encoding="utf-8"))
        for segment in data.get("segments", []):
            for raw in TOKEN_RE.findall(str(segment.get("text", ""))):
                token = raw.removesuffix("'s").removesuffix("’s")
                uppercase_count = sum(ch.isupper() for ch in token)
                if token in COMMON_UPPERCASE or uppercase_count < 2:
                    continue
                keys = [token]
                if "-" in token:
                    keys.append(token.split("-", 1)[0])
                for key in keys:
                    if key in COMMON_UPPERCASE or sum(ch.isupper() for ch in key) < 2:
                        continue
                    counts[key] = counts.get(key, 0) + 1
                    examples.setdefault(key, [])
                    if len(examples[key]) < 3 and source_path.name not in examples[key]:
                        examples[key].append(source_path.name)
    rows = [{"token": token, "count": count, "files": examples[token]}
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))]
    return rows[:40]


def hallucination_runs(texts: list[str], minimum: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    i = 0
    while i < len(texts):
        text = texts[i].strip()
        j = i + 1
        while text and j < len(texts) and texts[j].strip() == text:
            j += 1
        if text and j - i >= minimum:
            runs.append({"start_index": i, "end_index": j - 1, "count": j - i, "text": text})
        i = j if j > i + 1 else i + 1
    return runs

ABSENCE_TERMS = [
    "ELIZA", "Weizenbaum", "DALL-E", "Midjourney", "RMSprop", "Transformer",
    "Attention", "ChatGPT", "Stable Diffusion", "GCN", "GAT", "BERT", "RoBERTa",
    "TF-IDF", "LSTM", "GRU", "CRF", "RAG",
]
ABSENCE_PATTERNS = {
    "DALL-E": r"dall[-·]e",
    "Stable Diffusion": r"stable diffusion",
    "RMSprop": r"RMSprop",
}


def extract_reference_text(reference_dir: Path) -> str:
    roots = [reference_dir / "01_Lecture_Slides & Activities", reference_dir / "05_Miscellaneous"]
    pdfs = [path for root in roots if root.exists() for path in root.rglob("*.pdf")]
    chunks: list[str] = []
    for pdf in pdfs:
        try:
            result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                                    capture_output=True, text=True, timeout=30)
            chunks.append(result.stdout)
        except (OSError, subprocess.SubprocessError):
            continue
    return "\n".join(chunks)


def absence_scan(reference_text: str, transcript_text: str) -> list[dict[str, Any]]:
    rows = []
    for term in ABSENCE_TERMS:
        pattern = ABSENCE_PATTERNS.get(term, TERM_PATTERNS.get(term, re.escape(term)))
        reference_count = len(re.findall(rf"(?<![\\w-])(?:{pattern})(?![\\w-])", reference_text, re.I))
        transcript_count = len(re.findall(rf"(?<![\\w-])(?:{pattern})(?![\\w-])", transcript_text,
                                           0 if term == "RMSprop" else re.I))
        suspect_count = len(SUSPECTS[term].findall(transcript_text)) if term in SUSPECTS else 0
        rows.append({"term": term, "reference_count": reference_count,
                     "transcript_correct_count": transcript_count,
                     "suspect_count": suspect_count,
                     "status": "absent_or_low" if reference_count and transcript_count == 0 else "present"})
    return rows

def main() -> int:
    a = args()
    files = sorted(a.input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No transcript JSON files found in {a.input_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = a.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    uppercase_rows = discover_uppercase_tokens(files)
    (report_dir / "uppercase_tokens_top40.json").write_text(json.dumps(uppercase_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "uppercase_tokens_top40.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "count", "files"]); writer.writeheader()
        for row in uppercase_rows:
            writer.writerow({"token": row["token"], "count": row["count"], "files": ";".join(row["files"])})
    transcript_text = "\n".join(
        str(segment.get("text", ""))
        for source_path in files
        for segment in json.loads(source_path.read_text(encoding="utf-8")).get("segments", [])
    )
    absence_rows = absence_scan(extract_reference_text(a.reference_dir), transcript_text)
    (report_dir / "term_absence_scan.json").write_text(json.dumps(absence_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "term_absence_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "reference_count", "transcript_correct_count", "suspect_count", "status"]); writer.writeheader(); writer.writerows(absence_rows)
    audit_rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    hallucination_rows: list[dict[str, Any]] = []
    for source_path in files:
        data = json.loads(source_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        texts = [str(s.get("text", "")) for s in segments]
        all_text = "\n".join(texts)
        week = str(data.get("week", ""))
        runs = hallucination_runs(texts, a.min_hallucination_run)
        run_by_start = {run["start_index"]: run for run in runs}
        skipped_indices = {index for run in runs for index in range(run["start_index"] + 1, run["end_index"] + 1)}
        for run in runs:
            original_segments = [segments[index] for index in range(run["start_index"], run["end_index"] + 1)]
            hallucination_rows.append({
                "file": source_path.name, "source": data.get("source", source_path.name), "week": week,
                "start": original_segments[0].get("start"), "end": original_segments[-1].get("end"),
                "count": run["count"], "text": run["text"],
                "original_segments": original_segments,
            })
        for term in TERMS:
            row = {"file": source_path.name, "week": week, "term": term,
                   "correct_count": canonical_count(all_text, term),
                   "suspect_count": 0}
            suspect_re = SUSPECTS.get(term)
            if suspect_re:
                suspect_segments = [i for i, text in enumerate(texts) if suspect_re.search(text)]
                for i in suspect_segments:
                    if term == "BLEU" and not BLEU_CONTEXT.search(
                        " ".join(texts[max(0, i - 2): min(len(texts), i + 3)])):
                        continue
                    row["suspect_count"] += len(suspect_re.findall(texts[i]))
            audit_rows.append(row)

        corrected_data = dict(data)
        corrected_segments = []
        corrected_lines = []
        for i, segment in enumerate(segments):
            if i in skipped_indices:
                continue
            original = str(segment.get("text", ""))
            nearby = " ".join(texts[max(0, i - 2): min(len(texts), i + 3)])
            corrected, changes = correction_for(original, week, nearby)
            out_segment = dict(segment)
            out_segment["text_corrected"] = corrected
            if i in run_by_start:
                out_segment["hallucination_collapsed"] = True
                out_segment["hallucination_repeat_count"] = run_by_start[i]["count"]
            corrected_segments.append(out_segment)
            corrected_lines.append(f"[{int(float(segment.get('start', 0))) // 60:02d}:{int(float(segment.get('start', 0))) % 60:02d}] {corrected}")
            for term, before, after in changes:
                corrections.append({
                    "source": data.get("source", source_path.name), "file": source_path.name,
                    "week": week, "start": segment.get("start"), "end": segment.get("end"),
                    "term": term, "count": len(SUSPECTS[term].findall(original)), "original": before, "corrected": after,
                })
        corrected_data["segments"] = corrected_segments
        corrected_data["hallucination_collapsed_runs"] = sum(1 for run in runs)
        corrected_data["hallucination_collapsed_segments"] = sum(run["count"] for run in runs)
        out_json = a.output_dir / source_path.name
        out_json.write_text(json.dumps(corrected_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out_txt = a.output_dir / (source_path.stem + ".txt")
        out_txt.write_text("\n".join(corrected_lines) + ("\n" if corrected_lines else ""), encoding="utf-8")

    (report_dir / "hallucinations.json").write_text(json.dumps(hallucination_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "hallucinations.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["file", "source", "week", "start", "end", "count", "text"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in hallucination_rows:
            writer.writerow({key: row[key] for key in fields})
    with (report_dir / "hallucination_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["file", "hallucination_segments", "runs", "longest_run"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for source_path in files:
            file_runs = [row for row in hallucination_rows if row["file"] == source_path.name]
            writer.writerow({"file": source_path.name, "hallucination_segments": sum(row["count"] for row in file_runs),
                             "runs": len(file_runs), "longest_run": max((row["count"] for row in file_runs), default=0)})

    with (report_dir / "term_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "week", "term", "correct_count", "suspect_count"])
        writer.writeheader(); writer.writerows(audit_rows)
    (report_dir / "corrections.json").write_text(json.dumps(corrections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "corrections.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["source", "file", "week", "start", "end", "term", "count", "original", "corrected"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(corrections)
    summary = []
    for term in TERMS:
        rows = [r for r in audit_rows if r["term"] == term]
        summary.append({"term": term, "correct_count": sum(r["correct_count"] for r in rows),
                        "suspect_count": sum(r["suspect_count"] for r in rows)})
    (report_dir / "term_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "corrections": len(corrections), "hallucination_runs": len(hallucination_rows), "hallucination_segments": sum(row["count"] for row in hallucination_rows), "output_dir": str(a.output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
