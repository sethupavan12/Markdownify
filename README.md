## Markdownify

Mardownify is a super easy-to-use PDF/image to high-quality Markdown converter using Vision LLMs. It supports text, images, signatures, tables, charts, flowcharts and preserves document structure (Headings, numbered lists etc).

Tables become Markdown tables, charts become Mermaid diagrams, and images get concise summaries. Use as a CLI or Python library. Works with 100+ LLMs. Recommended to use with `gpt-5-mini` or `gpt-4.1-mini` or even better models for better performance. The best part is you have complete control over what gets rendered how!

If you don't believe it there's a whole gallery of examples with really wide range of OCR tasks you can explore here -> [Gallery](https://github.com/sethupavan12/Markdownify/blob/main/examples/gallery.md) 

![Handwritten notes converted to Markdown](examples/image.png)

<img width="1867" height="450" alt="image" src="https://github.com/user-attachments/assets/9a8b5176-03d8-4063-a8f3-4b1e52bdbe72" />

### Install
```bash
uv pip install llm-markdownify
# or
pip install llm-markdownify
```

### Quickstart (CLI)
```bash
# Set OpenAI key as env var
export OPENAI_API_KEY="sk-.."
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
    profile="generic",    # default; or "contracts", or a path to a JSON profile
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
    ```

- **Google Gemini**
  - Set your API key (Google AI Studio key):
    ```bash
    export GEMINI_API_KEY="..."
    ```
  - Example usage (pick a Gemini vision-capable model):
    ```bash
    markdownify input.pdf -o output.md --model gemini/gemini-2.5-flash
    ```

- **OpenRouter**
  - Set your API key (OpenRouter API Key):
    ```bash
    export OPENROUTER_API_KEY="..."
    ```
  - Example usage (pick a Gemini vision-capable model):
    ```bash
    markdownify input.pdf -o output.md --model openrouter/z-ai/glm-4.5v
    ```
- **Anthropic (Claude)**
  ```bash
  export ANTHROPIC_API_KEY="..."
  markdownify input.pdf -o output.md --model anthropic/claude-sonnet-5
  ```

- **DeepSeek**
  ```bash
  export DEEPSEEK_API_KEY="..."
  markdownify input.pdf -o output.md --model deepseek/deepseek-flash
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
    ```
  - See: [LiteLLM Azure OpenAI](https://docs.litellm.ai/docs/providers/azure/#overview)

- **OpenAI-compatible APIs**
  - Many providers expose an OpenAI-compatible REST API. Set your API key and base URL:
    ```bash
    export OPENAI_API_KEY="..."
    export OPENAI_API_BASE="https://your-openai-compatible-endpoint.com/v1"
    ```
  - Use the model name supported by that endpoint, prefixed with `openai/`, or pass the URL per run:
    ```bash
    markdownify input.pdf -o output.md --model openai/<model-name> --api-base https://your-endpoint/v1
    ```
  - The same works for local servers (Ollama, LM Studio, vLLM, llama.cpp) with no API key.
  - Reference: [LiteLLM Providers](https://docs.litellm.ai/docs/providers)

For additional providers and advanced configuration (fallbacks, cost tracking, streaming), see the LiteLLM docs: [Getting Started](https://docs.litellm.ai/).

### Configuration flags
- `--model`: LiteLLM model (e.g., `gpt-5-mini`, `azure/<deployment>`, `gemini/gemini-2.5-flash`)
- `--profile`: prompt profile, `generic` (default), `contracts`, or a path to a JSON profile
- `--dpi`: PDF render DPI (default 200). Ignored for image inputs.
- `--max-image-px`: longest side of each page image sent to the model (default 2048)
- `--max-group-pages`: max pages to merge when a table or chart continues across pages (default 3)
- `--no-grouping`: disable LLM-based detection of content continuing across pages
- `--temperature`, `--max-tokens`, `--reasoning-effort`: generation parameters
- `--api-base`: point at any OpenAI-compatible server (vLLM, Ollama, LM Studio)
- `--concurrency`, `--grouping-concurrency`, `--rate-limit`: throughput controls
- `--cache`: cache LLM responses on disk so re-runs are free

## Markdownify Cloud
If you’d like to run Markdownify in production with advanced features, on your own infrastructure, using your own LLMs, or tailored to your specific use case, visit [markdownify.xyz ](https://www.markdownify.xyz/) to explore our cloud offering and get in touch.

Markdownify Cloud gives you access to better version of Markdownify that gives better results than the open-source version, with additional features and hands-on support to help you integrate it into your workflow.

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
