# Speed, cost and scale

There are three ways to run llm-markdownify, depending on what you are optimizing. All numbers below
were measured on 2026-09-27 on the same 105-page olmOCR-Bench subset (tables, math, multi-column,
old scans, tiny text, headers/footers) unless noted. Scores are the olmOCR-Bench pass rate.

| Mode | When | Speed | Cost | Quality |
|---|---|---|---|---|
| [Large jobs, cheap](#large-jobs-slow-and-cheap) | thousands of documents, results can wait | up to 24h | about half | same as normal requests |
| [Low latency](#low-latency) | a user or an agent is waiting | ~5 s per page, pages in parallel | normal | 80-82 |
| [Best quality](#best-quality) | accuracy matters most | ~14 s per page, pages in parallel | normal | 84-85 |
| [Local, no API](#local-no-api) | private data, no key | depends on your hardware | no per-page cost | depends on the model |

## Large jobs: slow and cheap

Back-filling a document archive into a RAG index is the typical case: tens of thousands of pages,
nobody waiting. `markdownify-batch` sends them through a provider batch API, which costs about half
of normal requests and finishes within 24 hours. Small jobs often finish in minutes: a 3-document
test job completed in 2.5 minutes.

Two providers are supported. The model name picks one:

| Provider | Batch API | Key | Example `--model` |
|---|---|---|---|
| OpenAI | [Batch API](https://platform.openai.com/docs/guides/batch) | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| Anthropic | [Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing) | `ANTHROPIC_API_KEY` | `anthropic/claude-opus-5` |

```bash
# 1. Render every page and submit. Takes files and/or directories.
markdownify-batch submit ./archive -o ./archive-md --model gpt-5.4-mini
#   or: --model anthropic/claude-opus-5

# 2. Check progress whenever you like (from any machine that has the output directory)
markdownify-batch status ./archive-md

# 3. Download finished pages and write one .md per input. Safe to run repeatedly.
markdownify-batch collect ./archive-md --retry-failed

# Or block until everything is done:
markdownify-batch wait ./archive-md
```

How it behaves:

- Job state lives in `./archive-md/.markdownify-batch/`. Submit, close the laptop, collect tomorrow.
- Pages are rendered one at a time and split into batches of at most 190 MB or 50,000 pages, under
  both providers' limits. Page images make each request a few hundred KB, so expect roughly one
  batch per 500 pages.
- A document is written only when all of its pages have succeeded. Blank pages count as done.
  `collect` lists missing pages with the reason, and `--retry-failed` resubmits just those pages.
  `collect` and `wait` exit with code 1 when the job has finished but some documents are incomplete.
- A file that cannot be read is reported and skipped; the rest of the job continues.
- If the process dies mid-submit, the next `status` or `collect` finds the batch it started. (For
  Anthropic, which has no batch metadata, it matches by creation time and page count; if it cannot
  be sure, those pages show as missing and `--retry-failed` resubmits them.)
- Each page is its own request, so tables that continue across a page break come back as two tables.
  Use the regular command if that matters for your documents.
- Budget: about 3,100 input and 900 output tokens per page with `gpt-5.4-mini`, measured on the
  subset. Batch pricing is half of the normal per-token price for both providers.
- `--reasoning-effort` maps to OpenAI's `reasoning_effort` and Anthropic's `output_config.effort`.

From Python:

```py
from llm_markdownify.batch import submit_batch, batch_status, collect_batch

submit_batch(["./archive"], "./archive-md", model="anthropic/claude-opus-5")
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

Private documents, no key, no per-page cost: run a vision model on your own machine or server behind
an OpenAI-compatible API, then point llm-markdownify at it. The library does not bundle or manage
local models; any server that speaks the OpenAI chat completions API works.

```bash
markdownify input.pdf -o out.md --model openai/<model-name> --api-base <server>/v1
```

No API key is needed for a local server. Pick one way to serve the model:

| Server | Good for | Default address | Setup |
|---|---|---|---|
| [LM Studio](https://lmstudio.ai/docs/developer/openai-compat) | a desktop app, easiest on a laptop (Apple Silicon via MLX) | `http://localhost:1234/v1` | download a vision model in the app, then start the server (or `lms server start`) |
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | a one-line install on Mac, Linux, Windows | `http://localhost:11434/v1` | `ollama pull <vision-model>` |
| [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) | a GPU server, high throughput | `http://localhost:8000/v1` | `vllm serve <hf-model>` |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | GGUF models on CPU or small GPUs | `http://localhost:8080/v1` | `llama-server -m model.gguf --mmproj mmproj.gguf` |

Example with LM Studio, as verified on a MacBook:

```bash
lms server start
lms load <model-id>
markdownify scan.png -o scan.md --model openai/<model-id> --api-base http://localhost:1234/v1
```

Choosing a model:

- Use a **general vision-language model** (for example the Qwen-VL family). It follows
  llm-markdownify's prompt: reading order, LaTeX math, tables, no page furniture.
- **Document-specific small models** (granite-docling, SmolDocling, dots.ocr, olmOCR and similar) are
  trained to emit their own output format instead of following a prompt. Served this way,
  granite-docling returned DocTags markup, not Markdown. Run those with their own tooling, for example
  [Docling](https://docling-project.github.io/docling/) or its HTTP server
  [docling-serve](https://github.com/docling-project/docling-serve).
- Independent runs on olmOCR-Bench put Qwen3.5-4B at 75.4 and Qwen3.5-9B at 77.2
  ([IDP leaderboard](https://benchmarking.nanonets.com/benchmarks/olmocr), third-party). A 4B model
  needs roughly 4-8 GB of memory, depending on quantization.
- Local speed depends entirely on your hardware; measure on your own documents with
  [`evals/olmocr_bench`](../evals/olmocr_bench).
