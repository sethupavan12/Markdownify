# olmOCR-Bench evals

Measures conversion quality on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
(1,403 single-page PDFs, about 7,000 pass/fail tests: text present, headers/footers absent, reading
order, table cells, math). Data is ODC-BY and is downloaded, never committed.

## Setup

The scorer lives in the `olmocr` package, which has heavy dependencies, so keep it in its own venv:

```bash
uv venv .venv-bench -p 3.12
VIRTUAL_ENV=.venv-bench uv pip install "olmocr[bench]" numpy
.venv-bench/bin/python -m playwright install chromium   # math tests render LaTeX in a browser
```

## Run

```bash
# 1. Fixed stratified subset (15 PDFs per category, seed 0 -> 105 PDFs, 608 tests)
uv run python evals/olmocr_bench/make_subset.py --per-category 15 --out evals/olmocr_bench/data/subset15

# 2. Convert every page with the library (needs a provider key, e.g. OPENAI_API_KEY)
uv run python evals/olmocr_bench/run.py evals/olmocr_bench/data/subset15 my_run --model gpt-6-luna

# 3. Score
.venv-bench/bin/python -m olmocr.bench.benchmark --dir evals/olmocr_bench/data/subset15 --candidate my_run
```

`--opt key=value` passes any `convert()` argument, e.g. `--opt max_image_px=1536 --opt profile=contracts`.

## Reading results

- The score is the average of per-category pass rates, plus a 95% bootstrap interval.
- On the 105-page subset, differences under about 3 points are noise. Confirm a win with a repeat run
  or a bigger subset (`--per-category 30`) before shipping it.
- Keep the provider's rate limit in mind: 12 workers on a 200k TPM key will hit limits, which the
  library retries.
- Current results are in `docs/ROADMAP.md`.
