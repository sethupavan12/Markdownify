# Roadmap

Goal: the fastest route from any document to accurate, RAG-ready Markdown, for people and for agents.
One job, done well: convert to Markdown. Retrieval, embedding and QA stay out of scope.

Quality is measured, not asserted. Every phase reports olmOCR-Bench numbers (see "How we measure").

## Where we are (2026-09-26)

olmOCR-Bench, fixed stratified subset (105 PDFs, 15 per category, 608 tests + 105 baseline checks),
model `gpt-6-luna`, each run through the public `convert()` API with default settings.

| Version | Overall | Hard failures | arXiv math | Hdr/Ftr | Tiny text | Multi-col | Old scans | Old math | Tables | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| v0.4.0 (latest litellm) | 72.5 ± 2.7 | 18 / 105 | 86.9 | 85.4 | 92.7 | 87.7 | 13.0 | 40.9 | 90.5 | 315 s |
| phase 1 (this branch) | **84.9 ± 2.7** | **0 / 105** | 85.2 | 87.8 | 93.9 | 86.2 | 55.4 | 81.9 | 89.2 | 166 s |

Ablations on the same subset (phase 1 code, one change each):

| Variant | Overall | Notes |
|---|---|---|
| default (repeat run) | 84.1 | run-to-run noise is about 1 point overall, larger per category |
| old `contracts` prompt | 78.9 | the new prompt is worth about 6 points (old scans, old math, tables) |
| PNG instead of JPEG | 84.0 | no gain, bigger uploads |
| `max_image_px=1536` | 81.9 | 34% fewer prompt tokens, loses about 3 points, mostly old scans |
| `gpt-5.6-luna` | 82.8 | best on tables (93.2), weaker on old math |

For scale, published full-benchmark numbers on the same harness family: Marker 2 balanced 76.0,
olmOCR 2 82.4, Mistral OCR 4 85.2 (claimed), Chandra 2 85.9, GPT-5.4 used raw 81.0. Our number is on a
subset, so it is indicative until the full run in phase 2.

## Phase 1 - Reliability and accuracy (done, branch `modernize-phase1`)

Fixed the bugs that made pages fail or silently degrade: PDFium thread crash, oversize images, grouping
dead on reasoning models, retrying non-retryable errors, API key leak in CLI errors, "None" written as
content, rate limiter bursts, inconsistent defaults. New general-purpose default prompt (reading order,
no page furniture, LaTeX math, HTML tables for merged cells). Every fix has a regression test.

## Phase 2 - A v1 API that agents and pipelines can use directly

The library should work well called from a script, a service, or an agent's shell.

- **Result object, not just a file.** `markdownify(source) -> Result` with `.markdown`, per-page
  Markdown and page numbers, grouping decisions, token usage and cost, per-page errors, timings.
  `convert(input, output)` stays as a thin wrapper for compatibility.
- **Inputs:** path, bytes, file-like, URL. **Outputs:** file, stdout (`-o -` and default when piped), JSONL
  per page.
- **Page ranges** (`--pages 1-5,8`) and **partial results**: `--continue-on-error` keeps good pages and
  marks failed ones; results stream to disk as pages finish so a crash or Ctrl-C loses nothing and
  `--resume` picks up where it stopped.
- **No global state.** Retry, rate limit, cache and logging settings live on a per-run context, so
  concurrent conversions in one process cannot interfere. Library logging uses `NullHandler`; the CLI
  configures output.
- **Async API** (`amarkdownify`) for services.
- **Agent surface:** `markdownify mcp` (stdio MCP server exposing `convert_document` with page ranges and
  continuation, following MinerU 4's progressive-reading design); `uvx llm-markdownify file.pdf` works
  with zero setup beyond a key; an `llms.txt` and an agent skill file documenting the CLI.
- **Backend decision (needs maintainer sign-off):** see "Open decisions".

## Phase 3 - Speed and cost at production scale

- **Hybrid text layer.** Most born-digital PDFs have a correct text layer. Pass it to the model as an
  anchor (olmOCR's "anchored" mode) and skip the VLM for pages that are plain text. Marker and MinerU
  get 5-20x speed and cost wins this way. Measure accuracy with and without anchors.
- **Cheaper continuation detection.** Today it is one full vision call per page pair (about 10 s each on
  a reasoning model). Replace with: text-layer heuristics first; bottom/top strip crops at low detail
  when needed; or single-pass conversion with the previous page's tail as context and table stitching in
  code.
- **Adaptive resolution.** Start at a lower pixel cap and escalate for dense or tiny text pages.
- **Batch mode** using provider batch APIs (about 50% cheaper) for large corpora, plus a cost estimate
  (`--dry-run`) before a big run.
- **Prompt caching** on providers that need explicit markers (Anthropic `cache_control`).

## Phase 4 - Fully local, no API key

A separate niche: private data, air-gapped, zero cost per page.

- **Local model presets.** Most small document VLMs serve an OpenAI-compatible API (Ollama, LM Studio,
  vLLM, llama.cpp), so the transport already works through `--api-base`. What is missing is per-model
  prompts and output parsing. Ship presets like `--preset lightonocr-2`, `--preset qwen3.5-4b`,
  `--preset paddleocr-vl`, benchmarked on the same subset.
- **Docling backend** (`pip install llm-markdownify[docling]`, `--backend docling`): fully local pipeline
  (layout + TableFormer, or granite-docling-258M via MLX on Apple Silicon), plus Office, HTML and EPUB
  inputs that we do not support today. Docling's standard pipeline scores about 50 on olmOCR-Bench, so we
  position it as "local and broad", and keep VLM mode as "accurate".
- Publish a local-vs-API table: accuracy, pages per minute on a MacBook and on one GPU, cost.

## Phase 5 - RAG-ready output

- Heading-aware chunks with page provenance (`--chunks jsonl`), sized by tokens.
- Front matter with source metadata (title, pages, model, version).
- Figure extraction: crop figures to files and link them, alongside the text description.
- Stable page anchors (`<!-- page 3 -->`) so citations can point back to the source page.

## Engineering hygiene (ongoing)

- CI: Python 3.11-3.14 matrix (3.10 is end of life on 2026-10-31), actions pinned to commit SHAs,
  Dependabot for actions and Python.
- Release: PyPI trusted publishing with attestations instead of an API token; trigger on release
  published.
- Pre-commit: update ruff to match the lockfile.
- Nightly quality job: olmOCR-Bench subset on one cheap model, failing if the score drops beyond noise.

## Open decisions

1. **LLM transport.** LiteLLM gives 100+ providers but brings about 70 dependencies, a supply-chain
   history (compromised 1.82.7/1.82.8 on 2026-03-24), and a model table that lags new releases (v0.4.0's
   lock could not call gpt-6 models). Option A: keep LiteLLM as the core, bounded and pinned (current
   state). Option B: make the official `openai` SDK the core, with provider presets for OpenAI-compatible
   endpoints (OpenAI, Azure, Gemini, Anthropic, OpenRouter, Ollama, vLLM, LM Studio), and move LiteLLM to
   an optional `[litellm]` extra for everything else. Recommendation: B.
2. **Default model.** `gpt-4.1-mini` is dated. Pick the default from a cost vs score sweep in phase 2.
3. **v1 breaking changes.** Phase 2's Result API is additive, but removing globals and changing
   logging are observable changes. Ship as 1.0 with a migration note.

## How we measure

- Benchmark: [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) (ODC-BY). Unit tests
  per page: text present, headers/footers absent, reading order, table cell relationships, math rendered
  equivalence. Chosen because it scores any Markdown output, needs no TeX or Docker, and is what most
  tools publish.
- Iteration: fixed stratified subset, seed 0, 15 PDFs per category. Differences under about 3 points are
  noise; confirm wins with a repeat run or a larger subset.
- Headline: the full 1,403-page benchmark, reported next to the same model called with a naive prompt,
  so the number shows what the library adds.
- Secondary: OmniDocBench v1.6 (non-commercial dataset) only if comparisons with it are needed.
