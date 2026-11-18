# Agentic RAG Research Assistant

An agentic research assistant that indexes local PDF, HTML, TXT, DOCX, and
PPTX files plus arXiv papers, retrieves evidence with hybrid dense and sparse
search, and answers questions with sources. It includes document chat,
deep-read reports, evaluation and observability workflows, a Gradio UI served
by FastAPI, and an optional MCP server.

## Quick start

Requirements: Python 3.10+ and credentials for the configured LLM provider.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure the selected LLM provider in `.env`. The default `LLM_BACKEND` is
`perplexity`; `gemini` and `ollama` are also supported. PDF page-vision analysis
needs an image-capable provider: OpenAI, Gemini, or a vision-capable Ollama
model. The PDF indexing UI extracts text locally and leaves page-image vision
analysis off by default. If enabled for scanned pages or visual content,
page images are sent to the configured cloud provider when OpenAI or Gemini
vision is selected.

Start the app:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860), upload PDF, HTML, TXT,
DOCX, or PPTX files under **Data Sources**, build the index, and then use
Research or Document Chat. Text extraction runs locally. Scanned/image-only
PDFs are not supported because local OCR is not configured; no external OCR
service is called. The first run downloads local embedding and reranker models.
Indexes and other runtime data are stored locally and are not included in Git.

Document Chat uses single-pass RAG: one top-k retrieval followed by one
generation call per non-empty question. It does not run LLM relevance judges,
groundedness judges, or answer-regeneration retries. The separate **Run Agentic
RAG** research workflow remains multi-step by design.

Document indexing offers three chunking strategies: parent-child with semantic
children (the default), recursive fixed-size chunks, and semantic-only chunks.
The menu explains how each strategy works. The selected strategy applies to local document uploads in the Gradio app;
arXiv, API, and MCP ingestion retain their existing parent-child semantic
behavior. Optional PDF page-image vision analysis is separate from local text
parsing and may send page images to a configured provider. A failed index build
leaves the last successful index available; the status identifies which
indexing stage failed.

## API, MCP, and Docker

The FastAPI service exposes authenticated RAG endpoints including `/v1/search`,
`/v1/query`, `/v1/ingest`, and `/v1/upload`. Set a unique `RAG_API_KEY` in
`.env`; all endpoints except `/v1/health` require the `X-API-Key` header.
Path-based `/v1/ingest` is restricted to `RAG_INGEST_ROOT` (defaults to the
local `ingest/` directory). In Docker, `/data/ingest` is mounted as a
persistent volume. `/v1/upload` continues to accept direct file uploads.

MCP is included in the main `requirements.txt`. See
[MCP_README.md](./MCP_README.md) for server and client configuration. The
example client configuration is in
[mcp_client_config.example.json](./mcp_client_config.example.json).

For container deployment, run `docker compose up --build` from this directory.
The container runs as a non-root user. Compose persists the index, ingestion
files, artifacts, logs, user memory, optimization state, and downloaded models
in named Docker volumes, and restarts the service if its process exits. The
service is bound to localhost by default. The health check verifies the HTTP
endpoint; Docker does not automatically restart a container solely because it
is marked unhealthy.

## Evaluation

Run the native RAGAS evaluation after building the matching paper index and
configuring an LLM provider:

```bash
python evaluation/ragas_native.py \
  --golden evaluation/golden_set_ragas.json \
  --index index_store
```

The runner evaluates the 12 reference-answer questions in the golden set and
writes per-question results and metric averages to
`evaluation/ragas_native_results.jsonl`. It makes provider API calls for answer
generation and judging; embeddings run locally. Results apply only to the
corpus represented by that golden set. This baseline uses a context-only
generation prompt rather than the app's full document-chat or Self-RAG
orchestration, so it is not an end-to-end score for the production pipeline.

To inspect retrieval without calling an LLM provider:

```bash
python evaluation/ragas_native.py --retrieval-only \
  --golden evaluation/golden_set_ragas.json \
  --index index_store
```

This writes ranked sources and retrieval/reranker scores to
`evaluation/ragas_retrieval_trace.jsonl`; it does not calculate answer-quality
RAGAS metrics. The separate `ragas_batch.py` report includes custom retrieval
metrics and proxy scores; do not present those as native RAGAS metrics.

## Further documentation

- [Quick start](./QUICK_START.md)
- [Project guide](./PROJECT_GUIDE.md)
- [Project revision notes](./PROJECT_REVISION_NOTES.md)
- [MCP setup](./MCP_README.md)
- [Level 5 features](./LEVEL5_README.md)
