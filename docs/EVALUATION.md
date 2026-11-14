# Evaluation: what does RAG add to a course assistant?

Numbers in this document come from `reports/bench/RESULTS.md`, which is generated
by `python -m ee6405.eval.bench_report`. Figures are in `reports/figures/`.

## Question

The assistant answers course questions with an LLM plus retrieval over the course
material. A modern LLM already knows a lot of NLP. So the question is not "can it
answer" but **how much better does it answer with our retrieval than without, on
which kinds of question, and at what cost?**

## Design

| Design choice | What we did | Why |
| --- | --- | --- |
| Test sets | 4 sets, 152 questions (below) | One set cannot separate "general NLP knowledge" from "this course's content" |
| Baseline | The same model with no retrieval | Isolates the effect of RAG from the effect of the model |
| Strong baselines | 8 models from 5 vendors, small and large | "A bigger model without RAG" is the obvious alternative to building RAG |
| Algorithms compared | BM25, dense (BGE-m3), hybrid (reciprocal rank fusion) | Lexical vs semantic vs fused retrieval |
| Hyperparameters | top-k = 1, 3, 5, 10; retrieval restricted to one material type | Sensitivity, and which material carries the answers |
| Metrics | Exact-match accuracy; option-level precision / recall / F1; retrieval hit rate and MRR; tokens and latency | Accuracy alone hides partial credit and cost |
| Uncertainty | Wilson 95% intervals; exact McNemar test on paired questions | The sets are small, so differences need a test |

### Test sets

| Set | Questions | Source | Answer type |
| --- | --- | --- | --- |
| Practice MCQ | 47 | Weekly practice questions with an answer key | one of 4-5 |
| Coding quiz | 18 | Graded coding quizzes (scanned, OCR'd) | one of 5 |
| Theory quiz | 30 | IRA / TRA theory quizzes | several of 5 (multi-label) |
| Course-specific facts | 57 | Written by an LLM from the syllabus, project brief, in-class activity slides and instructor clarifications | one of 4 |

The first three are the course's real graded questions. The fourth exists because
none of the real quizzes asks about things only this course knows (deadlines,
grading rules, what an activity asked for); it is **synthetic** and is reported
separately everywhere.

How the sets are built (`src/ee6405/eval/quiz_sets.py`):

- Practice questions are parsed exactly from markdown and the answer key.
- Coding and theory quizzes are LMS print-outs. An LLM restructures them, then each
  question is checked against the source: every option must appear verbatim, and
  the correct options must be the ones the source labels "Correct answer". 13
  questions failed a check (mostly code that was a screenshot) and were dropped.
- Course-specific questions are kept only if the quoted evidence is verbatim in the
  source page and a second model, given only that page, reaches the same answer.
  Options are shuffled with a fixed seed, because the writer model put the key on
  "B" in 55 of 57 questions.

### Controls

- **No answer leakage.** Practice questions and graded quizzes are in the index
  with their answers. They are excluded from retrieval in every configuration, and
  the job builder fails if a single assessment chunk is retrieved (0 of 1,520 jobs).
- **Same context for every model.** Retrieval runs once; all models answer the
  identical prompt, so model differences are not retrieval differences.
- **Scoring cannot be gamed by verbosity.** Only the final `ANSWER:` line counts;
  2 of 5,775 answers had no parsable line and were scored wrong.
- Temperature 0 for every model.

## Results

### 1. RAG helps every model

Pooled over all 152 questions, accuracy with hybrid retrieval (top 5) is higher
than without retrieval for all 8 models, by 6.6 to 13.2 points, and the paired
difference is significant for each (McNemar p from < 0.001 to 0.035). Averaged
over models: **85.5% without retrieval, 94.7% with it.**

See `01_rag_effect_by_model.png` and `03_helped_vs_hurt.png`.

### 2. The gain depends on what kind of question it is

Mean over 8 models, no retrieval → hybrid:

| Set | No retrieval | Hybrid | Change |
| --- | --- | --- | --- |
| Practice MCQ | 86.2 | 91.0 | +4.8 |
| Coding quiz | 97.9 | 98.6 | +0.7 |
| Theory quiz | 87.5 | 90.8 | +3.3 |
| Course-specific facts | 80.0 | 98.5 | +18.4 |

Reading code needs no course material, so coding quizzes are at ceiling either way.
Course-specific facts are where retrieval is decisive. 102 of the 152 questions are
answered correctly by all 8 models with no retrieval at all: most graded questions
test general NLP knowledge that the models already have.

### 3. A small model with RAG beats a large model without

`qwen3.8-flash` with retrieval scores 94.7%; the strongest model without retrieval
(`gpt-6.1-sol`) scores 91.4%. Retrieval is worth more than moving up a model tier,
and the best result (98.0%) needs both.

### 4. The three retrievers are close; fusion is not a free win

Pooled, mean over models: BM25 94.7, dense 94.0, hybrid 94.7. Course vocabulary is
distinctive (model names, library calls), which suits lexical matching. Dense
retrieval earns its place on cross-lingual questions, which this English benchmark
does not measure: a Chinese question retrieves nothing with BM25 over English slides.

On the 57 questions with a known source page, hybrid retrieval puts that exact page
in the top 5 for 98% of questions (MRR 0.89). See `07_retrieval_quality.png`.

### 5. Three chunks are enough; one is not

For `deepseek-flash`, top-1 retrieval is worse than no retrieval on theory questions
(80.0 vs 86.7). A likely reason is that one chunk covers only part of a multi-answer
question; we did not test that directly. From k = 3 the result is stable. See `05_topk_sensitivity.png`.

### 6. No single material type is sufficient

Restricting retrieval to slides, transcripts or notebooks alone gives 91%, 79% and
75% on course-specific facts, against 100% with all sources: notebooks and
transcripts alone are worse than no retrieval (86%), because they cannot contain the
answer (the source pages are slides and course documents), yet the model still
leans on what it was given. See `06_source_ablation.png`.

### 7. Exact match understates multi-answer performance

On theory questions, exact match is 87.5% without retrieval but option-level F1 is
97.4%: models usually miss or add one option, not the whole question. Reporting only
exact match would make the models look much worse than they are, and reporting only
F1 would hide that one question in eight is still wrong. See `04_multilabel_metrics.png`.

### 8. Cost

Retrieval multiplies the prompt from about 280 to about 1,370 tokens per question
(roughly 5x) and does not measurably change response time.

## When RAG makes an answer worse

RAG broke 21 answers across the 8 models while fixing 132. Two patterns:

- **A similar-looking snippet misleads.** A question about setting Adam's learning
  rate was answered correctly by all 8 models alone; with retrieval, 3 switched to the
  option with a different learning-rate value. The retrieved chunks were course code
  and discussion posts on optimisers, not the answer.
- **Multi-answer questions lose an option.** On a stemming-versus-lemmatisation
  question, 3 models that had both correct options alone kept only one of them with
  retrieval.

And where it helps on real quiz questions:

- **The course's own definitions.** On a multi-answer stemming question, closed-book
  models disagreed with each other (5 different answers across 8 models); with the
  Week 1 slides and lecture transcript retrieved, all 8 gave the course's answer.
- **Library details the course teaches.** A question hinging on an NLTK smoothing
  method that does not exist was missed by 5 of 8 models alone, and by none once the
  course notebook was retrieved.

## What this means beyond the benchmark

- RAG is not what makes the assistant *know NLP*; the model does. RAG makes it answer
  **the way this course teaches it** and about **this course's specifics**, and it
  makes every claim checkable through the citation.
- The size of the benefit should be expected to track how much of the question
  depends on private or local knowledge. A course with unusual notation, a company
  wiki or a lab manual would gain more than a standard textbook course.
- The failure mode is trust in retrieved text. Mitigations to try next: require k ≥ 3,
  keep all material types in the pool, and filter chunks by a relevance threshold
  instead of always passing k of them.

## Limitations

- Small sets: 18 coding questions cannot resolve differences of a few points. Per-set
  differences between retrievers are within noise.
- The course-specific set is LLM-written. Models reach about 80% on it without
  retrieval (chance is 25%), so its distractors are guessable and the true gap on
  course-specific knowledge is probably larger than measured.
- Multiple choice measures whether the right fact is available, not whether a free-
  text explanation is good or faithful. Citation correctness is not scored.
- All benchmark questions are in English; the cross-lingual advantage of dense
  retrieval is shown in the demo but not quantified.
- One of 3,648 gateway requests kept timing out (`glm-5.3-flash`, one BM25 job), so
  that cell has 46 instead of 47 questions.
- One run per configuration at temperature 0; run-to-run variation is not measured.

## Reproducing

```bash
python -m ee6405.eval.quiz_sets                      # practice, coding, theory sets
python -m ee6405.eval.quiz_sets --course             # course-specific set
python -m ee6405.eval.bench                          # retrieval -> reports/bench/jobs.jsonl
python scripts/run_bench.py --jobs reports/bench/jobs.jsonl \
    --out reports/bench/responses_<name>.jsonl --models <model ...>
python -m ee6405.eval.bench_report                   # tables + figures
python scripts/draw_pipeline.py                      # system diagram
```

`run_bench.py` uses only the standard library and reads the endpoint and key from
`.env`, so it can be copied to any machine that reaches a model endpoint. The test
sets, prompts and raw model replies contain course assessment content and are not in
Git; scores, per-question outcomes and figures are.
