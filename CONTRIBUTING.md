# Contributing to llm-markdownify

Thanks for your interest in contributing! This guide explains how to set up your environment, coding practices, how to run tests, and how to ship releases.

## Overview
- Package name: `llm-markdownify`
- Repo: https://github.com/sethupavan12/Markdownify
- License: Apache-2.0 (SPDX). Please keep `LICENSE` and `NOTICE` intact.

## Getting started
### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (fast dependency manager)
- Git

### Setup
```bash
# Clone and enter
git clone git@github.com:sethupavan12/Markdownify.git
cd Markdownify

# Create & sync the virtual environment with dev tools
uv sync --all-extras --dev

# Install pre-commit to auto-run formatting, lint, and license headers
uv run pre-commit install
# Run once across the repo
uv run pre-commit run --all-files
```

## Development workflow
- Format & lint (Ruff is configured via pre-commit):
  ```bash
  uv run pre-commit run --all-files
  # or directly
  uv run ruff check src tests
  uv run ruff format src tests
  ```
- Run tests:
  ```bash
  uv run pytest -q
  ```
- Build the package:
  ```bash
  uv build
  ```
- Local smoke test:
  ```bash
  pip install -U dist/*.whl
  python -c "import llm_markdownify; print(llm_markdownify.__version__)"
  uv run markdownify --help
  ```

## Running the tool
- CLI:
  ```bash
  uv run markdownify run /absolute/path/to/input.pdf -o /absolute/path/to/output.md --model gpt-4.1-mini
  ```
- Library API:
  ```python
  from llm_markdownify import convert

  convert(
      "/path/input.pdf",
      "/path/output.md",
      model="gpt-4.1-mini",
      profile="contracts",
  )
  ```

## Prompt profiles
The prompts for continuation detection and markdown conversion are swappable.
- Built-ins live in `src/llm_markdownify/prompt_profiles.py` (default: `contracts`, also `generic`).
- You can pass a custom JSON profile at runtime via `--profile /path/to/profile.json`.
- JSON must include: `name`, `continuation_system`, `continuation_user`, `markdown_system`, `markdown_user`.

Prompts should:
- Produce only document content (no disclaimers/notes/meta commentary)
- Follow the heading policy for contracts (top-level numeric headings, strict decimal rules, TOC as lists)
- Emphasize tables as GFM tables and charts as Mermaid when feasible

## Grouping logic
The grouping algorithm merges consecutive pages only when a split visual structure (tables, boxed panels, charts) continues across the boundary. Plain text continuity alone must not trigger merging. Keep changes simple and readable.

## Performance tips
- DPI: Default is 72. Higher DPI improves fidelity but increases latency. Suggest 120–200 for harder documents.
- Concurrency: Configurable; be mindful of provider rate limits.
- Prefer PNG for crisp text; JPEG can be added in future for heavy scans.

## Coding standards
- Python typing everywhere; prefer explicit types for function signatures and public APIs.
- Readability first: meaningful names, early returns, guard clauses, handle edge cases.
- Logging via `src/llm_markdownify/logging.py` utility (no emojis; concise and actionable messages).
- Keep modules focused; avoid excessive abstractions.
- Docstrings (Google/NumPy style acceptable); explain “why” for complex parts.
- Include SPDX header on source files. Run:
  ```bash
  uv run python scripts/add_header.py
  ```

## Commits & PRs
- Small, focused PRs with clear descriptions.
- Suggested prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.
- Include tests for fixes/features where practical. For LLM behavior, prefer prompt/unit tests using mocks.
- No API keys or secrets in commits. Use environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `AZURE_API_KEY`).

## CI & releasing
- CI runs lint and tests on pushes/PRs.
- Publishing to PyPI is automated via GitHub Actions on tagged releases.
  1) Bump version in `pyproject.toml`.
  2) Create a GitHub Release with tag `vX.Y.Z` (must match the version).

## Issues & discussions
- File issues and feature requests here: https://github.com/sethupavan12/Markdownify/issues
- Please include steps to reproduce and environment details for bugs.

Thanks for contributing! 🙌
