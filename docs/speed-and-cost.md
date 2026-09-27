# Speed, cost and scale

There are three ways to run llm-markdownify, depending on what you are optimizing. All numbers below
were measured on 2026-09-27 on the same 105-page olmOCR-Bench subset (tables, math, multi-column,
old scans, tiny text, headers/footers) unless noted. Scores are the olmOCR-Bench pass rate.

| Mode | When | Speed | Cost | Quality |
|---|---|---|---|---|
| [Large jobs, cheap](#large-jobs-slow-and-cheap) | thousands of documents, results can wait | up to 24h | about half | same as normal requests |
| [Low latency](#low-latency) | a user or an agent is waiting | ~5 s per page, pages in parallel | normal | 80-82 |
| [Best quality](#best-quality) | accuracy matters most | ~14 s per page, pages in parallel | normal | 84-85 |
| [Local, no API](#local-no-api) | private data, no key | 2.5-7 s per page on a laptop (Docling) | no per-page cost | 48-55 (Docling) |

## Large jobs: slow and cheap

Back-filling a document archive into a RAG index is the typical case: tens of thousands of pages,
nobody waiting. Use the OpenAI Batch API through `markdownify-batch`. Batch requests cost about half
of normal requests and OpenAI finishes them within 24 hours (small jobs often finish in minutes: a
3-document test job completed in 2.5 minutes).

```bash
export OPENAI_API_KEY="sk-..."

# 1. Render every page and submit. Takes files and/or directories.
markdownify-batch submit ./archive -o ./archive-md --model gpt-5.4-mini

# 2. Check progress whenever you like (from any machine that has the output directory)
markdownify-batch status ./archive-md

# 3. Download finished pages and write one .md per input. Safe to run repeatedly.
markdownify-batch collect ./archive-md --retry-failed

# Or block until everything is done:
markdownify-batch wait ./archive-md
```

How it behaves:

- Job state lives in `./archive-md/.markdownify-batch/`. Submit, close the laptop, collect tomorrow.
- Requests are split into several batches to stay under OpenAI's per-file limits (200 MB, 50,000
  requests). Page images make each request a few hundred KB, so expect roughly one batch per 500
  pages.
- A document is written only when all of its pages have succeeded. `collect` lists missing pages, and
  `--retry-failed` resubmits just those pages.
- Each page is its own request, so tables that continue across a page break come back as two tables.
  Use the regular command if that matters for your documents.
- Budget: about 3,100 input and 900 output tokens per page with `gpt-5.4-mini`, measured on the
  subset. Batch pricing is half of the normal per-token price.
- OpenAI models only for now (`gpt-5.4-mini`, `gpt-6-luna`, ...). `--api-base` works with other servers
  that implement the OpenAI Batch API.

From Python:

```py
from llm_markdownify.batch import submit_batch, batch_status, collect_batch

submit_batch(["./archive"], "./archive-md", model="gpt-5.4-mini")
print(batch_status("./archive-md").completed)
result = collect_batch("./archive-md", retry_failed=True)
print(result.written, result.incomplete)
```

## Low latency

For interactive use, where a person or an agent is waiting on the result:

```bash
markdownify doc.pdf -o doc.md --model gpt-5.4-mini --no-grouping --concurrency 16
```

| Setting | Score | Median time per page |
|---|---|---|
| `gpt-5.4-mini` | 80.4 | 4.8 s |
| `gpt-5.4-mini --reasoning-effort low` | 82.0 | 5.8 s |
| `gpt-5.4-mini --reasoning-effort low --max-image-px 1536` | 77.6 | 5.7 s |
| `gpt-6-luna` (default settings) | 84.9 | 14.6 s |

What moves latency:

- **Pages run in parallel.** A document's wall time is close to its slowest page, not the sum, up to
  `--concurrency` pages at once (default 4). Raise it as far as your rate limit allows; rate-limit
  errors are retried automatically.
- **`--no-grouping`** skips the check for tables that continue onto the next page. That check is a
  full model call per page pair and runs as its own round before conversion starts, so on a
  multi-page PDF it roughly doubles both the wall time and the number of requests. Keep grouping on if
  your documents have long tables.
- **Smaller images do not help.** Capping at 1536 px cut input tokens by a third but cost 4 points of
  accuracy and saved no time.
- **`--cache`** makes repeat conversions of the same file instant.

## Best quality

Use the strongest vision model you have access to, with default settings (DPI 200, 2048 px images,
grouping on). On the same subset `gpt-6-luna` scores 84.9; the same model called with a bare
"convert this page to Markdown" prompt scores 72.6.

## Local, no API

No API key, no data leaving the machine. We measured IBM's [Docling](https://github.com/docling-project/docling)
(2.130) on the same 105 pages, on an Apple M4 laptop with 16 GB of RAM, pages one at a time after a
warm-up run:

| Local option | Score | Median per page | Mean | Slowest 10% |
|---|---|---|---|---|
| Docling, standard pipeline (layout model + OCR + TableFormer) | 54.7 | 2.5 s | 4.4 s | over 10.8 s |
| Docling + granite-docling-258M on MLX (Apple GPU) | 48.2 | 7.1 s | 14.4 s | over 50 s |
| For reference: llm-markdownify + `gpt-5.4-mini` (API) | 80.4 | 4.8 s | | |

What that means in practice:

- **The Docling standard pipeline is a reasonable local choice for clean, born-digital text and
  tables.** It removed headers and footers perfectly (100) and scored 82.4 on tables, at about 2.5 s a
  page on a laptop. It scored 0 on math, because formula recognition is off by default, and 30 on old
  scans.
- **granite-docling-258M was slower and less accurate on this benchmark** than Docling's standard
  pipeline: 9.8 on tiny text, 48.6 on tables, and some pages took close to a minute while generation
  ran long. It is a very small model; it does not replace a large vision model.
- Neither is close to an API vision model on hard pages (math, scans, dense text).

To try Docling yourself:

```bash
pip install docling
docling input.pdf --to md --output out/
```

The better fully-local route with this library is a small vision model served on your machine, with
llm-markdownify's prompt, through any OpenAI-compatible server:

```bash
# LM Studio, Ollama, vLLM or llama.cpp serving a vision model at localhost
markdownify input.pdf -o out.md --model openai/<local-model> --api-base http://localhost:1234/v1
```

Independent runs of small general models on olmOCR-Bench put Qwen3.5-4B at 75.4 and Qwen3.5-9B at
77.2 ([IDP leaderboard](https://benchmarking.nanonets.com/benchmarks/olmocr), third-party, not yet
measured by us). Benchmarked presets for local models are on the [roadmap](ROADMAP.md).
