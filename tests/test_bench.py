"""Quiz benchmark: job rendering and scoring."""

import math

from ee6405.eval.bench import build_messages, parse_config
from ee6405.eval.bench_report import mcnemar_exact, parse_answer, wilson
from ee6405.eval.quiz_sets import QuizQuestion, _marked_correct, _shuffled


def test_parse_config():
    assert parse_config("closed") == (None, 0, None)
    assert parse_config("hybrid@5") == ("hybrid", 5, None)
    assert parse_config("bm25@10:slides") == ("bm25", 10, "slides")


def test_messages_carry_context_only_when_retrieved():
    q = QuizQuestion("q1", "theory", "src", "W04", "Pick the true statements.",
                     {"A": "one", "B": "two"}, ["A", "B"], multi=True)
    closed = build_messages(q, [])[1]["content"]
    with_rag = build_messages(q, ["CRF is discriminative."])[1]["content"]
    assert "COURSE MATERIAL" not in closed
    assert "[1] CRF is discriminative." in with_rag
    assert "One or more options are correct" in closed  # multi-answer instruction
    assert closed.rstrip().endswith("ANSWER: <letters separated by commas>")


def test_parse_answer_reads_the_last_answer_line():
    assert parse_answer("Option B looks right.\nANSWER: B") == ["B"]
    assert parse_answer("ANSWER: A\nOn reflection...\n**ANSWER:** C, a, E") == ["C", "E"]
    assert parse_answer("answer: A, C and D") == ["A", "C", "D"]
    assert parse_answer("A, C") == ["A", "C"]  # bare letters are unambiguous
    # prose that merely contains capital letters is not an answer
    assert parse_answer("A CRF is a discriminative model.") is None
    assert parse_answer("") is None


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson(40, 47)
    assert lo < 40 / 47 < hi and 0 < lo and hi < 1
    assert wilson(0, 10)[0] == 0 or wilson(0, 10)[0] < 1e-12
    assert math.isnan(wilson(0, 0)[0])


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(8, 0) - 2 / 2**8) < 1e-12  # all eight changes in one direction
    assert mcnemar_exact(8, 2) == mcnemar_exact(2, 8)


def test_source_marks_are_read_from_both_export_styles():
    pdf = "A. Correct answer33.33%Correct: CRF is discriminative\nB. -20%CRF is generative\n"
    assert _marked_correct(pdf, "CRF is discriminative", "A") is True
    assert _marked_correct(pdf, "CRF is generative", "B") is False
    ocr = "A. It replaces attention. -20%\n\nD. **Correct:** It normalizes per token. Correct answer 100%\n"
    assert _marked_correct(ocr, "It normalizes per token.", "D") is True
    assert _marked_correct(ocr, "It replaces attention.", "A") is False


def test_shuffle_keeps_the_right_answer_and_is_repeatable():
    options = {"A": "w", "B": "x", "C": "y", "D": "z"}
    shuffled, key = _shuffled(options, "B", "course/q7")
    assert sorted(shuffled.values()) == ["w", "x", "y", "z"] and shuffled[key] == "x"
    assert _shuffled(options, "B", "course/q7") == (shuffled, key)
