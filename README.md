## Markdownify

Mardownify is a super easy-to-use PDF/image to high-quality Markdown converter using Vision LLMs. It supports text, images, signatures, tables, charts, flowcharts and preserves document structure (Headings, numbered lists etc).

Tables become Markdown tables, charts become Mermaid diagrams, and images get concise summaries. Use as a CLI or Python library. Works with 100+ LLMs. Recommended to use with `gpt-5-mini` or `gpt-4.1-mini` or even better models for better performance. 

If you don't believe it there's a whole gallery of examples with really wide range of OCR tasks you can explore here -> [Gallery](https://github.com/sethupavan12/Markdownify/blob/main/examples/gallery.md) 

<img width="2000" height="600" alt="image" src="https://github.com/user-attachments/assets/9a8b5176-03d8-4063-a8f3-4b1e52bdbe72" />

### Install
```bash
uv pip install llm-markdownify
# or
pip install llm-markdownify
```

### Quickstart (CLI)
```bash
# PDF input
markdownify input.pdf -o output.md --model gpt-5-mini

# Or single image input (PNG/JPG/JPEG)
markdownify input.png -o output.md --model gpt-5-mini
```

### Features
- High-quality complex markdown generation powered by LLMs. 
- Supports Text, Images, Tables, Charts.
- Built-in prompts tuned for clean Markdown, Mermaid, and structured headings along with ability to customise.
- Supports multi-page tables, charts and images.
- High-fidelity page rendering from PDF.
- Optional DOCX→PDF conversion using MS word installation.
- Works seamlessly with 100+ LLMs with LiteLLM Intergration.

### Python API (one-liner)
```py
from llm_markdownify import convert

convert(
    "input.pdf",  # or an image path like "input.png"
    "output.md",
    model="gpt-5-mini",   # optional; can rely on env/provider defaults
    dpi=72,
    profile="contracts",    # or path to JSON profile
)
```

Optional DOCX support (macOS/Windows via Word):
```bash
pip install llm-markdownify[docx]
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
    markdownify input.pdf -o output.md --model gpt-5-mini
    markdownify input.pdf -o output.md --model gpt-5-mini
    ```

- **Google Gemini**
  - Set your API key (Google AI Studio key):
    ```bash
    export GOOGLE_API_KEY="..."
    ```
  - Example usage (pick a Gemini vision-capable model):
    ```bash
    markdownify input.pdf -o output.md --model gemini/gemini-2.5-flash
    markdownify input.pdf -o output.md --model gemini/gemini-2.5-flash
    ```

- **Azure OpenAI**
  - Set these environment variables (values from your Azure OpenAI resource):
    ```bash
    export AZURE_API_KEY="..."
    export AZURE_API_BASE="https://<your-resource>.openai.azure.com"
    export AZURE_API_VERSION=""
    ```
  - Use your deployment name via the `azure/<deployment_name>` model syntax:
    ```bash
    markdownify input.pdf -o output.md --model azure/<deployment_name>
    markdownify input.pdf -o output.md --model azure/<deployment_name>
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
    markdownify input.pdf -o output.md --model <model-name>
    markdownify input.pdf -o output.md --model <model-name>
    ```
  - Reference: [LiteLLM Providers](https://docs.litellm.ai/docs/providers)

For additional providers and advanced configuration (fallbacks, cost tracking, streaming), see the LiteLLM docs: [Getting Started](https://docs.litellm.ai/).

### Configuration flags
- `--model`: LiteLLM model (e.g., `gpt-5-mini`, `azure/<deployment>`, `gemini/gemini-2.5-flash`)
- `--dpi`: Render DPI (default 72). Ignored for direct image inputs.
- `--max-group-pages`: Max pages to merge for continued content (default 3)
- `--no-grouping`: Disable LLM-based grouping
- `--temperature`, `--max-tokens`: LLM generation params

### Attribution & License
This project uses the Apache 2.0 License, which includes an attribution/NOTICE requirement. If you distribute or use this project, please keep the `LICENSE` and `NOTICE` files intact, crediting the original author, Sethu Pavan Venkata Reddy Pastula.

- Project repository: https://github.com/sethupavan12/Markdownify

### Development
- Requires Python 3.10+
- Use `uv` for fast installs: `uv sync`
- Run tests: `pytest`
- Lint: `ruff check src tests`

Check [CONTRIBUTING.md](CONTRIBUTING.md) for more details

### Releasing
GitHub Actions are configured to:
- Run tests on PRs/pushes
- Build & publish to PyPI on tagged releases
