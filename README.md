# llm-markdownify

[![PyPI](https://img.shields.io/pypi/v/llm-markdownify)](https://pypi.org/project/llm-markdownify/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/sethupavan12/Markdownify/actions/workflows/ci.yml/badge.svg)](https://github.com/sethupavan12/Markdownify/actions/workflows/ci.yml)

Turn PDFs, scans and images into clean Markdown with the vision model you already pay for.
Built for RAG pipelines and AI agents.

```bash
pip install llm-markdownify
markdownify report.pdf -o report.md --model gpt-5.4-mini
```

Each page goes to a vision LLM with a prompt we measure on a public benchmark, so the Markdown comes
back the way a retriever or an agent wants it:

- the text exactly as printed, in reading order, including multi-column layouts
- no running headers, footers or page numbers polluting your chunks
- math as LaTeX (`$...$`, `$$...$$`)
- tables as Markdown, or as HTML when they have merged cells, and stitched back together when they
  break across pages
- charts as a description plus the data values, diagrams as Mermaid
- old scans and handwriting transcribed, not summarized

It works with any provider: OpenAI, Anthropic, Gemini, DeepSeek, Azure, OpenRouter, or a model running
on your own machine through Ollama, LM Studio or vLLM.

![Handwritten notes converted to Markdown](examples/image.png)

## How good is it?

We measure every change on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench), the
benchmark most PDF-to-Markdown tools publish. It checks 1,403 real pages with about 7,000 pass/fail
tests: is this sentence present, is the page footer gone, is paragraph A before paragraph B, is this
table cell next to that one, does this equation render the same.

**llm-markdownify 0.5 with `gpt-6-luna` scores 81.6 ± 1.0 on the full benchmark.**

| System | olmOCR-Bench | Source |
|---|---|---|
| Chandra 2 | 85.9 | [Datalab](https://huggingface.co/datalab-to/chandra-ocr-2) (self-reported) |
| Mistral OCR 4 | 85.2 | [Mistral](https://mistral.ai/news/ocr-4/) (self-reported) |
| olmOCR 2 (7B model trained for this benchmark) | 82.4 | [Ai2](https://github.com/allenai/olmocr) |
| **llm-markdownify 0.5 + gpt-6-luna** | **81.6** | measured with [`evals/olmocr_bench`](evals/olmocr_bench) |
| Marker 2 (balanced) | 76.0 | [Datalab](https://github.com/datalab-to/marker) (self-reported) |
| Docling | 50.3 | [Marker's benchmark](https://github.com/datalab-to/marker) |

By category: tables 89.1, headers/footers 86.6, old scans with math 82.5, multi-column 81.4, arXiv
math 79.4, tiny text 88.7, old scans 45.1. Old, faded scans are the weak spot.

What the library adds on top of the model, measured on a fixed 105-page stratified subset:

| Same pages | Score | Median time per page |
|---|---|---|
| `gpt-6-luna` with a bare "convert this page to Markdown" prompt | 72.6 | 11.9 s |
| llm-markdownify 0.4 + `gpt-6-luna` | 72.5 (18 pages failed) | 14.8 s |
| llm-markdownify 0.5 + `gpt-5.4-mini` | 80.4 | 4.8 s |
| llm-markdownify 0.5 + `gpt-6-luna` | 84.9 | 14.6 s |

The biggest single difference is headers and footers. With the bare prompt, 24% of the checks that
running headers, footers and page numbers are gone pass (88% with llm-markdownify). Left in, that
furniture ends up in every RAG chunk. Other projects' numbers are what they
published; ours come from running the benchmark's own scorer on our output.

Reproduce it yourself with [`evals/olmocr_bench`](evals/olmocr_bench). The harness scores the Markdown
this library writes, through the same `convert()` call you would use.

## Quickstart

```bash
export OPENAI_API_KEY="sk-..."

markdownify input.pdf -o output.md --model gpt-5.4-mini   # PDF
markdownify scan.png -o scan.md --model gpt-5.4-mini      # PNG, JPG, WEBP, TIFF (multi-page), BMP, GIF
markdownify big.pdf -o part.md --pages 1-5,12,40-         # only the pages you need
```

From Python, get the Markdown back directly:

```py
from llm_markdownify import markdownify

result = markdownify("report.pdf", model="gpt-5.4-mini")   # or bytes, a file object, or a URL
result.markdown          # the whole document
result.page(7).markdown  # one page
result.failed_pages      # [] when every page converted
result.usage             # requests, tokens, estimated cost in USD
```

`convert("input.pdf", "output.md", ...)` writes a file instead, and `amarkdownify()` is the async
version for web servers.

If a page still fails after retries, the rest of the document is kept: the failed page is marked
with an HTML comment and listed in `result.failed_pages` (the command exits with code 3). Answers
are cached on disk, so running the same command again only pays for the pages that failed. Pass
`strict=True` / `--strict` to fail the whole document instead, or `--no-cache` to skip the cache.

## Use any provider

The model name decides where the request goes (via [LiteLLM](https://docs.litellm.ai/docs/providers),
100+ providers). Set that provider's key and pick a vision-capable model.

| Provider | Key | Example `--model` |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-5` |
| Google Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/anthropic/claude-sonnet-5` |
| Azure OpenAI | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | `azure/<deployment>` |

Anything that speaks the OpenAI API works too, including local servers with no key at all:

```bash
# Ollama, LM Studio, vLLM, llama.cpp, or a hosted OpenAI-compatible provider
markdownify input.pdf -o output.md --model openai/<model-name> --api-base http://localhost:11434/v1
```

## Large jobs, low latency

Thousands of documents and no one waiting? `markdownify-batch` sends them through the OpenAI Batch
API or Anthropic Message Batches at about half the price, finished within 24 hours:

```bash
markdownify-batch submit ./archive -o ./archive-md --model gpt-5.4-mini   # or anthropic/claude-opus-5
markdownify-batch status ./archive-md
markdownify-batch collect ./archive-md --retry-failed
```

Someone waiting on the answer? Use a fast model and skip the cross-page check:

```bash
markdownify doc.pdf -o doc.md --model gpt-5.4-mini --no-grouping --concurrency 16   # ~5 s per page
```

No API at all? Serve a vision model locally with LM Studio, Ollama, vLLM or llama.cpp and pass
`--api-base`. Measured speed and quality trade-offs, and the local setup guide, are in
[docs/speed-and-cost.md](docs/speed-and-cost.md).

## Built for production

- **Safe to call from threads.** Use `convert()` from a web server or a worker pool. Each conversion
  keeps its own retry, rate-limit, cache and provider settings, including its API key.
- **Quiet as a library.** `convert()` prints nothing; your app's logging config decides what is
  shown (`logging.basicConfig(level=logging.INFO)` for progress messages). Pass
  `log_level="normal"` for the same output as the command-line tool.
- **Rate limits are waited out, real errors fail fast.** Rate limits, timeouts and 5xx responses are
  retried with backoff. A bad key or a bad request fails on the first attempt with one clear line,
  and never prints your API key.
- **Oversized pages are handled.** Page images are capped at 2048 px, below provider limits, so big
  scans don't get rejected.
- **Re-runs are free.** Answers are cached on disk (`~/.cache/llm-markdownify`, or
  `$LLM_MARKDOWNIFY_CACHE_DIR`), keyed on the exact page image and prompt. The cache holds the
  converted text of your documents and is never pruned: for confidential documents, turn it off
  (`--no-cache`, `enable_cache=False`) or delete the folder when you are done.
- **Safe with untrusted URLs.** URL input refuses private, local and cloud-metadata addresses,
  including through redirects (`--allow-private-urls` to permit them). Other strings are read as
  local file paths, so only pass trusted strings as the source.
- **One bad page does not sink a 300-page document.** It is marked and reported; rerun to fill it in.
- **Throughput and cost controls:** `--concurrency`, `--rate-limit`, `--max-image-px`,
  `--reasoning-effort`.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--model` | `gpt-4.1-mini` or `$LLM_MARKDOWNIFY_MODEL` | any LiteLLM model name |
| `--profile` | `generic` | prompt profile: `generic`, `contracts`, or a JSON file with your own prompts |
| `--dpi` | 200 | PDF render resolution (ignored for images) |
| `--max-image-px` | 2048 | longest side of each page image sent to the model |
| `--max-group-pages` | 3 | max pages merged when a table or chart continues onto the next page |
| `--no-grouping` | | skip cross-page detection (one fewer model call per page) |
| `--temperature`, `--max-tokens`, `--reasoning-effort` | provider defaults, 16000 | generation settings |
| `--api-base` | | any OpenAI-compatible endpoint |
| `--concurrency`, `--grouping-concurrency`, `--rate-limit` | 4, same, none | throughput |
| `--pages` | all | pages to convert, e.g. `1-5,12,40-` |
| `--strict` | off | fail the whole document if any page fails |
| `--cache/--no-cache`, `--cache-dir` | on, `~/.cache/llm-markdownify` | response cache |
| `-q`, `-v`, `--version` | | quiet, verbose, version |

Custom prompts: copy a built-in profile from
[`prompt_profiles.py`](src/llm_markdownify/prompt_profiles.py) into a JSON file with the fields `name`,
`continuation_system`, `continuation_user`, `markdown_system` and `markdown_user`, then pass
`--profile my_profile.json`.

DOCX input goes through Microsoft Word (macOS/Windows): `pip install "llm-markdownify[docx]"` and pass
`--allow-docx`. Exporting to PDF yourself is more reliable.

## Examples

The [gallery](examples/gallery.md) has about 80 inputs with their Markdown output: receipts, charts,
handwriting, formulas, forms, screenshots, scene text.

## Roadmap

Next up: an in-memory API that returns per-page results, stdout output and page ranges for agents, an
MCP server, a hybrid mode that uses the PDF's own text layer to cut cost, and presets for small local
models. Details and current numbers are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Markdownify Cloud

To run Markdownify in production on your own infrastructure, with your own LLMs, or tuned for your
documents, see [markdownify.xyz](https://www.markdownify.xyz/). The cloud version adds features beyond
the open-source library and comes with hands-on integration support.

## Contributing

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check src tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Maintainers and coding agents: start with [AGENTS.md](AGENTS.md).

## License

Apache 2.0. If you distribute this project, keep the `LICENSE` and `NOTICE` files intact, crediting
the original author, Sethu Pavan Venkata Reddy Pastula.
