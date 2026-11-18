# Quick Start Guide

## 1. Install dependencies

From this directory, create and activate a virtual environment, then install
the project requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure an LLM provider

Copy the example configuration and set the API key for the provider you use:

```bash
cp .env.example .env
```

The default `LLM_BACKEND` is `perplexity`. You can also configure `gemini` or
`ollama` in `.env`. Do not commit real API keys.

The **Data Sources** tab supports text-based PDF, HTML, TXT, DOCX, and PPTX.
Text extraction runs locally. Scanned/image-only PDFs are not supported because
local OCR is not configured; no external OCR service is used. Optional PDF
page-image vision analysis is a separate feature, is off by default, and may
send page images to the configured provider. Deep-Read has its own vision
workflow and may still require a vision provider.

## 3. Start the application

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860). Upload one or more
supported documents in the interface and build an index before asking research
questions.

For authenticated path-based API ingestion, place files under the local
`ingest/` directory or configure `RAG_INGEST_ROOT` to a trusted directory.
Docker uses a persistent `/data/ingest` volume. Direct API uploads are unchanged.

## Example research questions

- Summarize the main contributions of these papers.
- What limitations do the documents mention?
- Compare the methods proposed in these papers.
- What datasets were used?
- What future research directions are suggested?

For MCP client setup, see [MCP_README.md](MCP_README.md).

## Docker deployment notes

The Compose service runs as a non-root user and persists its index, ingestion
directory, artifacts, logs, user memory, and Hugging Face model cache in named
volumes. Optimization state is persisted separately under `/data/state`. The
service binds to `127.0.0.1:7860` by default. The HTTP health check reports
service reachability; Docker Compose does not restart a container solely
because it is marked unhealthy.
