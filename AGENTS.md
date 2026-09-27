# AGENTS.md

Guide for coding agents (and humans) working on `llm-markdownify`. Read this before editing.
Keep it short and true: if you change something described here, update this file in the same change.

## What this is

A Python library + CLI whose one job is converting documents to Markdown for LLM/RAG use.
It renders each page to an image and asks a vision LLM (any provider, via LiteLLM) to write Markdown.
Tables become GFM tables, charts become Mermaid, images get text descriptions.

- PyPI package: `llm-markdownify`. Import name: `llm_markdownify`. CLI entry point: `markdownify`.
- GitHub: `sethupavan12/Markdownify` (personal account, see global identity rules).
- License: Apache-2.0 with a `NOTICE` file. Keep `LICENSE`/`NOTICE` intact.

## Pipeline

```
input (.pdf | .png/.jpg/.jpeg | .docx opt-in)
  -> pager.load_document_pages      render pages to PNG (pypdfium2), one PageImage per page
  -> grouping.group_pages           optional: one LLM call per adjacent page pair asks
                                    CONTINUE_NEXT / NONE (split table/chart); merges up to
                                    max_group_pages consecutive pages into one group
  -> llm.generate_markdown          one vision LLM call per group (thread pool, `concurrency`)
  -> markdownifier.Markdownifier    joins group outputs in page order, writes the .md file
```

## Module map (`src/llm_markdownify/`)

| File | Role |
| --- | --- |
| `api.py` | `convert()` - public one-call Python API; builds a `MarkdownifyConfig` |
| `cli.py` | Typer app with a single command, so usage is `markdownify INPUT -o OUT.md` (no `run` subcommand) |
| `config.py` | `MarkdownifyConfig` pydantic model: validation and defaults, `LLM_MARKDOWNIFY_MODEL` env default |
| `markdownifier.py` | `Markdownifier` orchestrator; configures the module-level log/LLM/cache globals, then runs the pipeline |
| `pager.py` | Rendering. PDFs via pypdfium2 (under a global lock), images incl. multi-page TIFF. `PageImage` holds encoded bytes plus lazy data URLs (full image, and a 1024px JPEG for grouping) |
| `grouping.py` | Cross-page continuation detection and grouping |
| `llm.py` | LiteLLM calls, retries on transient errors only (rate limit, timeout, 5xx, empty answer), token-bucket `RateLimiter`, fence stripping, cache lookups. Silences LiteLLM logging at import |
| `cache.py` | Optional file cache of LLM responses (`~/.cache/llm-markdownify`), keyed on model + prompt + image hashes |
| `prompts.py` / `prompt_profiles.py` | Prompt text. Built-in profiles: `generic` (default) and `contracts` (legal heading rules). Custom profiles are JSON files with `name`, `continuation_system`, `continuation_user`, `markdown_system`, `markdown_user` |
| `logging.py` | `get_logger` / `set_log_level` (`quiet`/`normal`/`verbose`/`debug`) |

## Defaults to know

`MarkdownifyConfig` (`config.py`) is the single source of truth for defaults. `api.convert` and the CLI
pass only the values the caller set, so both behave the same (a regression test enforces this).
Key defaults: profile `generic`, DPI 200 capped at `max_image_px=2048`, JPEG page images, temperature
unset (provider default), `max_retries=5` for transient errors only. Model defaults to `gpt-4.1-mini`
unless `--model` or `LLM_MARKDOWNIFY_MODEL` is set.

## Gotchas

- PDFium is not thread-safe. Every pypdfium2 call must hold `pager._PDFIUM_LOCK`; concurrent rendering
  without it segfaults the interpreter.
- Reasoning models spend output tokens on thinking. Never put small `max_tokens` caps on calls, and do
  not send `temperature` unless the user set it.
- Never let the CLI print local variables on errors (they contain API keys). `pretty_exceptions_show_locals=False`.
- Retry/rate-limit/cache/log settings are module globals set per `Markdownifier`; tests reset them.

## Dev workflow

```bash
uv sync --all-extras --dev              # install
uv run pytest -q                        # tests (all mocked, no network)
uv run ruff check src tests             # lint (CI runs this)
uv run ruff format src tests            # format
uv run python scripts/add_header.py     # add SPDX headers to .py/.toml/.yml
uv run markdownify examples/data/ocr/DocVQA__fxxj0037_3.png -o /tmp/out.md --model <model>   # real E2E run
```

- CI: `.github/workflows/ci.yml` (lint + tests). Release: `.github/workflows/release.yml` publishes to PyPI
  when a GitHub Release is created. Bump the version in both `pyproject.toml` and `src/llm_markdownify/__init__.py`.
- Every `.py/.toml/.yml` file carries the Apache SPDX header (pre-commit adds it).
- Tests must not call real LLMs. Mock `llm_markdownify.llm` / LiteLLM. Real-model checks are manual E2E runs.
- Real E2E runs need a provider key in the environment (e.g. `OPENAI_API_KEY`). Never commit keys.

## Examples and data

- `examples/gallery.md` + `examples/data/ocr/*` (inputs) + `examples/data/markdown/*` (outputs) is a
  showcase of ~80 single-image OCR cases (charts, handwriting, receipts, tables, math). It is not a scored eval.
- `examples/quickstart.py` is a minimal library example.

## Evaluating quality

Accuracy is measured on olmOCR-Bench (allenai/olmOCR-bench): single-page PDFs with pass/fail unit tests
for text presence, header/footer absence, reading order, tables and math. Run the library's public
`convert()` over a fixed stratified subset, write `<candidate>/<category>/<stem>_pg1_repeat1.md`, and
score with `python -m olmocr.bench.benchmark --dir <subset> --candidate <name>`. Treat differences under
about 3 points on a ~100-page subset as noise. See `docs/ROADMAP.md` for current numbers.

## Conventions

- Python >= 3.10, full type hints on public functions, ruff line length 100.
- Keep modules small and focused. No framework-style abstraction layers.
- Log via `logging.get_logger`. Do not print.
- Prompts: output must be document content only (no meta commentary). Grouping merges pages only for split
  visual structures (tables, boxed panels, charts), never for plain text continuity.
- Do not hand-edit generated files (e.g. `uv.lock`); regenerate them with the tool.
