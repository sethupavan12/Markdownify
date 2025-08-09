## markdownify

Convert documents (PDF, DOCX) to high-quality Markdown using Vision LLMs via LiteLLM. Tables become Markdown tables, charts become Mermaid diagrams, and images get concise summaries. Use as a CLI or Python library.

### Features
- High-fidelity page rendering from PDF
- Optional DOCX→PDF conversion
- LLM-driven grouping of continued content across pages (tables/charts/images)
- Vision LLM prompts tuned for clean Markdown, Mermaid, and structured headings
- Works with LiteLLM providers, including OpenAI, Gemini, Azure OpenAI, and OpenAI-compatible APIs

### Install
```bash
uv pip install llm-markdownify
# or
pip install llm-markdownify
```

Optional DOCX support (macOS/Windows via Word):
```bash
pip install llm-markdownify[docx]
```

### Quickstart (CLI)
```bash
markdownify run input.pdf -o output.md --model gpt-4o-mini
```

### Python API (one-liner)
```python
from llm_markdownify import convert

convert(
    "input.pdf",
    "output.md",
    model="gpt-4.1-mini",   # optional; can rely on env/provider defaults
    dpi=200,
    profile="contracts",    # or path to JSON profile
)
```

### Configure your provider (via LiteLLM)
Pick one of the following. See the full providers list and details in the LiteLLM docs: [Supported Providers](https://docs.litellm.ai/docs/providers).

- **OpenAI**
  - Set your API key:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
  - Example usage:
    ```bash
    markdownify run input.pdf -o output.md --model gpt-4o-mini
    ```

- **Google Gemini**
  - Set your API key (Google AI Studio key):
    ```bash
    export GOOGLE_API_KEY="AIza..."
    ```
  - Example usage (pick a Gemini vision-capable model):
    ```bash
    markdownify run input.pdf -o output.md --model gemini/gemini-1.5-flash
    ```

- **Azure OpenAI**
  - Set these environment variables (values from your Azure OpenAI resource):
    ```bash
    export AZURE_API_KEY="..."
    export AZURE_API_BASE="https://<your-resource>.openai.azure.com"
    export AZURE_API_VERSION="2024-02-15-preview"
    ```
  - Use your deployment name via the `azure/<deployment_name>` model syntax:
    ```bash
    markdownify run input.pdf -o output.md --model azure/<deployment_name>
    ```
  - See: [LiteLLM Azure OpenAI](https://docs.litellm.ai/docs/providers/azure_openai)

- **OpenAI-compatible APIs**
  - Many providers expose an OpenAI-compatible REST API. Set your API key and base URL:
    ```bash
    export OPENAI_API_KEY="..."
    export OPENAI_API_BASE="https://your-openai-compatible-endpoint.com/v1"
    ```
  - Use the model name supported by that endpoint:
    ```bash
    markdownify run input.pdf -o output.md --model <model-name>
    ```
  - Reference: [LiteLLM Providers](https://docs.litellm.ai/docs/providers)

For additional providers and advanced configuration (fallbacks, cost tracking, streaming), see the LiteLLM docs: [Getting Started](https://docs.litellm.ai/).

### Configuration flags
- `--model`: LiteLLM model (e.g., `gpt-4o-mini`, `azure/<deployment>`, `gemini/gemini-1.5-flash`)
- `--dpi`: Render DPI (default 200)
- `--max-group-pages`: Max pages to merge for continued content (default 3)
- `--no-grouping`: Disable LLM-based grouping
- `--temperature`, `--max-tokens`: LLM generation params

### Dev tooling
- Install pre-commit and enable the license header hook:
  ```bash
  pip install pre-commit
  pre-commit install
  # run on all files once
  pre-commit run --all-files
  ```
  This inserts the header:
  ```
  # Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
  # Licensed under the Apache License, Version 2.0. See LICENSE file for details.
  # SPDX-License-Identifier: Apache-2.0
  ```

### Attribution & License
This project uses the Apache 2.0 License, which includes an attribution/NOTICE requirement. If you distribute or use this project, please keep the `LICENSE` and `NOTICE` files intact, crediting the original author, Sethu Pavan Venkata Reddy Pastula.

- Project repository: https://github.com/sethupavan12/Markdownify

### Development
- Requires Python 3.10+
- Use `uv` for fast installs: `uv sync`
- Run tests: `pytest`
- Lint: `ruff check src tests`

### Releasing
GitHub Actions are configured to:
- Run tests on PRs/pushes
- Build & publish to PyPI on tagged releases (requires `PYPI_API_TOKEN` secret)
