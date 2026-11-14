"""Build the three quiz test sets used to measure what RAG adds.

    python -m ee6405.eval.quiz_sets            # writes data/interim/eval_sets/*.jsonl

The course gives us three kinds of graded question, and they stress the
assistant differently, so they are kept as three datasets:

``practice``  weekly practice MCQs, one correct option      (concepts + code)
``coding``    coding quizzes, one correct option             (reading code)
``theory``    IRA / TRA theory quizzes, several correct options (multi-label)
``course``    course-specific facts (syllabus, project brief, in-class
              activities, instructor clarifications), one correct option

The first three are the course's own graded questions. They mostly test general
NLP knowledge, which a modern LLM already has. ``course`` is the complement:
questions whose answer exists only in this course's material. No such quiz
exists, so an LLM writes one question per source chunk -- see ``_course``.

Practice MCQs are plain markdown with an answer key, so they parse exactly.
The coding and theory quizzes are LMS print-outs whose option text, score
markers and instructor feedback run together; an LLM restructures them, and
every question is then checked against the source text -- an option that is
not verbatim in the source, or a correct-answer mark the source does not
show, gets the question dropped. Nothing unverifiable reaches a test set.

The sets hold course assessment content, so they are written under
``data/interim`` (git-ignored) and travel in the data bundle, not in Git.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ee6405.agent.llm import LLM, build_llm
from ee6405.config import REPO_ROOT, settings
from ee6405.rag.store import ChunkStore

OUT_DIR = REPO_ROOT / "data" / "interim" / "eval_sets"
LETTERS = "ABCDE"

# Which quiz files feed which set. Extra quizzes lost their code screenshots in
# PDF text extraction, so questions that depend on them fail the completeness check.
SET_OF = {"Coding Quiz": "coding", "IRA": "theory"}


@dataclass
class QuizQuestion:
    qid: str
    dataset: str  # practice | coding | theory
    source: str  # index source path the question came from
    week: str | None
    stem: str  # question text including any code
    options: dict[str, str]  # letter -> text
    correct: list[str]  # sorted letters
    multi: bool = False
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------ practice


def _practice(store: ChunkStore) -> list[QuizQuestion]:
    out = []
    for c in store.all_chunks():
        if c.modality != "mcq" or c.metadata.get("answer") is None:
            continue
        body = re.split(r"\n\s*Answer\s*:\s*\S+\s*$", c.text, maxsplit=1, flags=re.I)[0]
        body = re.sub(r"^#\s*Q\d+\s*:\s*", "", body.strip())
        # Options are the trailing numbered list, usually inside its own code fence.
        m = re.search(r"\n(?:```[a-z]*\n)?((?:[1-5][.)]\s.*(?:\n(?![1-5][.)]\s|```).*)*\n?){2,5})(?:```)?\s*$", body)
        if not m:
            continue
        options = {}
        for num, text in re.findall(r"(?ms)^([1-5])[.)]\s(.*?)(?=^[1-5][.)]\s|\Z)", m.group(1)):
            options[LETTERS[int(num) - 1]] = text.strip()
        try:
            answer = LETTERS[int(str(c.metadata["answer"]).strip()) - 1]
        except (ValueError, IndexError):
            continue
        if answer not in options:
            continue
        stem = Path(c.source).stem.replace("_Practice_MCQ", "")
        out.append(QuizQuestion(
            qid=f"practice/{stem}_Q{c.metadata.get('qid')}", dataset="practice", source=c.source,
            week=c.week, stem=body[: m.start()].strip(), options=options, correct=[answer],
        ))
    return out


# ------------------------------------------------- coding / theory (LLM + check)

_PROMPT = """Below is the text of one course quiz exported from a learning platform.
Option text, score markers and instructor feedback are mixed together.

Return ONLY a JSON array with one object per question:
  {"number": 1,
   "stem": "<question text, plus any code it shows, verbatim>",
   "options": {"A": "<option text>", "B": "...", ...},
   "correct": ["E"],
   "complete": true}

Rules:
- Copy stem and option text VERBATIM from the quiz. Do not fix or rephrase anything.
- Remove only the markers: scores like "-20%", "0%", "33.33%", "100%", the labels
  "Correct answer", "Correct:", "Incorrect:", and markdown asterisks.
- "correct" lists every option the quiz labels "Correct answer". Ignore which
  option the student picked.
- Leave out instructor feedback and explanations entirely.
- An option or stem may continue on a later page; join the parts.
- Set "complete" to false when the question cannot be answered from this text
  alone: it refers to code, a snippet, a figure or a screenshot that is not in
  the text, or some of its options are missing.

QUIZ TEXT:
"""


def _norm(s: str) -> str:
    # NFKC folds PDF ligatures ("ﬁ") that the model writes back as plain letters.
    return re.sub(r"[\s*`_]+", "", unicodedata.normalize("NFKC", s)).lower()


def _marked_correct(source: str, option_text: str, letter: str) -> bool | None:
    """Does the source label this option "Correct answer"? None = option not found."""
    flat = unicodedata.normalize("NFKC", re.sub(r"[*]+", "", source))
    option_text = unicodedata.normalize("NFKC", option_text)
    # Locate the option by its first words, tolerant of line wraps.
    words = option_text.split()[:6]
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    best = None
    for m in re.finditer(pattern, flat):
        # The marker sits either between "X." and the text (PDF export) ...
        before = flat[max(0, m.start() - 60): m.start()]
        label = before.rfind(f"{letter}.")
        if label < 0:
            continue
        head = before[label:]
        # ... or at the end of the option's own line (OCR markdown).
        end = flat.find("\n\n", m.end())
        tail = flat[m.end(): end if end >= 0 else len(flat)][:400]
        nxt = re.search(r"\n[A-E]\.\s", tail)
        tail = tail[: nxt.start()] if nxt else tail
        best = "Correct answer" in head or "Correct answer" in tail
        if best:
            return True
    return best


def _structure(text: str, llm: LLM) -> list[dict]:
    raw = llm.chat([{"role": "user", "content": _PROMPT + text}], temperature=0.0)
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def _from_quiz(name: str, source: str, text: str, dataset: str, llm: LLM) -> tuple[list[QuizQuestion], list[str]]:
    kept, dropped = [], []
    flat = _norm(text)
    week = (re.search(r"\(Week\s*(\d+)\)", text) or [None, None])[1]
    for item in _structure(text, llm):
        tag = f"{name} Q{item.get('number')}"
        options = {k: str(v).strip() for k, v in (item.get("options") or {}).items() if k in LETTERS}
        correct = sorted(set(item.get("correct") or []) & set(options))
        if not item.get("complete", True):
            dropped.append(f"{tag}: needs code or options that are not in the text")
            continue
        if len(options) < 3 or not correct:
            dropped.append(f"{tag}: fewer than 3 options or no correct answer")
            continue
        if any(_norm(t)[:80] not in flat for t in options.values()):
            dropped.append(f"{tag}: an option is not verbatim in the source")
            continue
        marks = {k: _marked_correct(text, t, k) for k, t in options.items()}
        if sorted(k for k, v in marks.items() if v) != correct:
            dropped.append(f"{tag}: correct answers disagree with the source marks {marks}")
            continue
        kept.append(QuizQuestion(
            qid=f"{dataset}/{name.replace(' ', '_')}_Q{item.get('number')}", dataset=dataset,
            source=source, week=f"W{int(week):02d}" if week else None,
            stem=str(item.get("stem", "")).strip(), options=options, correct=correct,
            multi=dataset == "theory",
        ))
    return kept, dropped


# ------------------------------------------------------- course-specific facts

_COURSE_PROMPT = """Write ONE multiple-choice question from the course document excerpt below.

The question must test a fact that is specific to THIS course (NTU EE6405) and
stated in the excerpt: logistics, grading, deadlines, rules, what a particular
class activity asked students to do, or what the instructor clarified. A student
who knows NLP well but never saw this course's material should NOT be able to
answer it from general knowledge.

Return ONLY a JSON object:
  {"stem": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
   "correct": "B", "evidence": "<the sentence from the excerpt that proves it, verbatim>"}

Rules:
- Exactly one option is correct; the other three are plausible but wrong.
- Name what you ask about ("the Week 4 class activity", "the group project"); do
  not write "according to the excerpt" or "in this document".
- If the excerpt holds no course-specific fact, return {"skip": true}.

EXCERPT (from NAME):
"""

_ORACLE_PROMPT = """Answer the question using only the excerpt. Reply with one letter.

EXCERPT:
TEXT

QUESTION:
QUESTION_TEXT
"""


def _course_chunks(store: ChunkStore, per_file: int = 4, min_chars: int = 350) -> list:
    """Chunks that hold course-specific rather than textbook content."""
    by_file = defaultdict(list)
    for c in store.all_chunks():
        name = Path(c.source).name
        specific = c.modality in {"misc", "clarification"} or "Class" in name or "Tasks" in name
        if specific and len(c.text) >= min_chars:
            by_file[c.source].append(c)
    picked = []
    for source in sorted(by_file):
        chunks = by_file[source]
        if chunks[0].modality in {"misc", "clarification"}:
            picked.extend(chunks)
        else:  # spread evenly through the deck instead of taking the first pages
            step = max(1, len(chunks) // per_file)
            picked.extend(chunks[::step][:per_file])
    return picked


def _json_object(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _shuffled(options: dict[str, str], correct: str, seed: str) -> tuple[dict[str, str], str]:
    """Reorder the options. The writer model puts the key on the same letter
    almost every time, which would reward a model that simply favours that letter."""
    texts = list(options.values())
    answer = options[correct]
    random.Random(seed).shuffle(texts)
    return dict(zip(LETTERS, texts, strict=False)), LETTERS[texts.index(answer)]


def _course(store: ChunkStore, writer: LLM, checker: LLM) -> tuple[list[QuizQuestion], list[str]]:
    """One LLM-written question per course-specific chunk, kept only if it checks out.

    Two checks, neither of which looks at how a closed-book model does (that
    would be selecting the test set on the result): the quoted evidence must be
    verbatim in the chunk, and a second model given only that chunk must arrive
    at the same answer.
    """
    kept, dropped = [], []
    for c in _course_chunks(store):
        name = Path(c.source).name
        tag = f"{name} {c.locator}"
        item = _json_object(writer.chat(
            [{"role": "user", "content": _COURSE_PROMPT.replace("NAME", name) + c.text}],
            temperature=0.0,
        ))
        if not item or item.get("skip"):
            dropped.append(f"{tag}: no course-specific fact")
            continue
        options = {k: str(v).strip() for k, v in (item.get("options") or {}).items() if k in LETTERS}
        correct = str(item.get("correct", "")).strip().upper()
        if len(options) != 4 or correct not in options or not item.get("stem"):
            dropped.append(f"{tag}: malformed question")
            continue
        if _norm(str(item.get("evidence", "")))[:60] not in _norm(c.text):
            dropped.append(f"{tag}: evidence is not verbatim in the chunk")
            continue
        rendered = str(item["stem"]) + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        oracle = checker.chat(
            [{"role": "user", "content": _ORACLE_PROMPT.replace("QUESTION_TEXT", rendered).replace("TEXT", c.text)}],
            temperature=0.0,
        )
        m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", oracle or "")
        if not m or m.group(1) != correct:
            dropped.append(f"{tag}: a second model reading the chunk disagreed with the key")
            continue
        qid = f"course/{Path(c.source).stem.replace(' ', '')}_{c.locator.replace(':', '')}"
        options, correct = _shuffled(options, correct, qid)
        kept.append(QuizQuestion(
            qid=f"course/{Path(c.source).stem.replace(' ', '')}_{c.locator.replace(':', '')}",
            dataset="course", source=c.source, week=c.week, stem=str(item["stem"]).strip(),
            options=options, correct=[correct],
            meta={"synthetic": True, "evidence": str(item.get("evidence", ""))},
        ))
    return kept, dropped


def build_course_set(index_path: Path | None = None, writer: LLM | None = None,
                     checker: LLM | None = None) -> tuple[list[QuizQuestion], list[str]]:
    store = ChunkStore(index_path or settings.index_path, read_only=True)
    try:
        return _course(store, writer or build_llm(allow_offline=False),
                       checker or build_llm(allow_offline=False))
    finally:
        store.close()


def build_sets(index_path: Path | None = None, llm: LLM | None = None) -> tuple[dict[str, list[QuizQuestion]], list[str]]:
    store = ChunkStore(index_path or settings.index_path, read_only=True)
    sets: dict[str, list[QuizQuestion]] = defaultdict(list)
    sets["practice"] = _practice(store)

    pages: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for c in store.all_chunks():
        if c.modality == "assessment_pdf":
            pages[c.source].append((c.position, c.text))
    store.close()

    llm = llm or build_llm(allow_offline=False)
    dropped: list[str] = []
    for source in sorted(pages):
        name = Path(source).stem
        dataset = next((v for k, v in SET_OF.items() if name.startswith(k)), None)
        if dataset is None:
            continue
        text = "\n".join(t for _, t in sorted(pages[source]))
        kept, lost = _from_quiz(name, source, text, dataset, llm)
        sets[dataset].extend(kept)
        dropped.extend(lost)
    return sets, dropped


def load_sets(directory: Path = OUT_DIR) -> dict[str, list[QuizQuestion]]:
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        out[path.stem] = [
            QuizQuestion(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the quiz test sets from the index.")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--model", default=None, help="model used to restructure the quiz print-outs")
    p.add_argument("--course", action="store_true",
                   help="build only the LLM-written course-specific set")
    p.add_argument("--checker", default=None,
                   help="--course: model that re-answers each question from its chunk")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.course:
        questions, dropped = build_course_set(
            writer=build_llm(model=args.model, allow_offline=False),
            checker=build_llm(model=args.checker, allow_offline=False),
        )
        sets = {"course": questions}
        note = "dropped_course.txt"
    else:
        sets, dropped = build_sets(llm=build_llm(model=args.model, allow_offline=False))
        note = "dropped.txt"
    for name, questions in sets.items():
        path = args.out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in questions),
            encoding="utf-8",
        )
        print(f"{name:9s} {len(questions):3d} questions -> {path}")
    (args.out / note).write_text("\n".join(dropped) + "\n", encoding="utf-8")
    print(f"dropped   {len(dropped):3d} (see {note})")
    for line in dropped:
        print("  -", line)


if __name__ == "__main__":
    main()
