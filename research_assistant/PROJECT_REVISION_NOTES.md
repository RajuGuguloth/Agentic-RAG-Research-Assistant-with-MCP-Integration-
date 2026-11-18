# Agentic RAG Research Assistant — Revision Notes

These notes follow the same learning path as the reference board: the problem,
RAG basics, this project's ingestion and retrieval flow, and evaluation. The
examples below describe this project, not every RAG system.

## 1. The problem

Research papers are long, and a general-purpose LLM may answer from its
training data instead of the specific papers a user wants to understand. It can
also make a claim without showing where that claim came from.

This project lets a user index local research documents, ask questions about
them, and see the retrieved sources. The goal is to make answers more useful
and traceable—not to claim that retrieval guarantees a correct answer.

**Short version:** “I built a tool that searches a user's research documents
first, then asks an LLM to answer from the passages it found.”

## 2. Understanding RAG

RAG means **Retrieval-Augmented Generation**. It has two main stages:

1. **Prepare the knowledge:** extract text, split it into chunks, create
   embeddings, and store searchable chunks.
2. **Answer a question:** retrieve relevant chunks, put them in the prompt, and
   ask an LLM to answer using that context.

The documents are not used to retrain the LLM. Retrieval selects context for a
question at run time.

**Mental model:** `Question → Retrieve evidence → Generate an answer from it`

### Useful terms

- **Chunk:** a smaller piece of a document that can be searched and passed to
  the LLM.
- **Embedding:** a numeric representation used to find text with similar
  meaning.
- **Dense search:** finds semantically similar text using embeddings.
- **Sparse search:** finds text using words and terms; this project uses BM25.
- **Reranking:** scores an initial set of results again to improve their order.
- **Ground truth:** a reference answer used to compare an evaluated result.

## 3. Data ingestion in this project

### Supported files and extraction

The local document loader extracts text from PDF, HTML, TXT, DOCX, and PPTX
files. It keeps useful source metadata, such as the filename and page where
available. Scanned/image-only PDFs are not supported by the local text parser:
OCR is not configured.

### Chunking choices

The local upload UI offers three strategies:

| Strategy | Simple explanation |
|---|---|
| Parent-child with semantic children | Search smaller meaning-based child chunks, then use their larger parent passages as context. This is the default. |
| Recursive fixed-size | Split text into bounded chunks using recursive separators. |
| Semantic-only | Split by changes in meaning without a separate parent passage. |

Parent-child chunking separates **finding a precise match** from **giving the
LLM enough surrounding text**. The selected UI strategy applies to local
document uploads. Other ingestion paths retain their existing chunking
behavior.

### Indexing flow

`Local file → Text extraction → Chunking → Embeddings + BM25 index → Persisted index`

The index is prepared before the user asks questions. Indexing and querying are
separate steps.

## 4. Data retrieval and answering

### Search flow

For a question, the retriever:

1. Searches semantically with FAISS and by keywords with BM25.
2. Combines rankings with Reciprocal Rank Fusion (RRF).
3. Maps matching child chunks to their parent passages where applicable.
4. Reranks candidates with a CrossEncoder and returns the best passages.

`Question → FAISS + BM25 → RRF → Parent passages → Reranker → Top results`

Dense search helps when the question uses different wording from the paper.
BM25 helps with exact terms, names, and technical phrases. Combining them
covers both kinds of matches.

### Document Chat

Document Chat uses **single-pass RAG**: it retrieves the top results once and
uses them in one LLM generation call for a non-empty question. It does not run
a repeated agent loop, relevance-judge calls, groundedness-judge calls, or
answer-regeneration retries. The response includes source information.

### Where “agentic” applies

The project has more than one workflow:

- **Document Chat:** the simple, single-pass retrieve-then-answer workflow.
- **Research workflow:** a multi-step workflow for broader goals, using
  planning, sub-queries, retrieval, and synthesis.
- **Deep-Read:** a separate workflow that can create a structured report and
  presentation from a paper.

So, when asked whether every question goes through an agent loop, say **no**.
The normal chat path is deliberately bounded; the broader research workflow
uses multiple steps when the task benefits from them.

### Interfaces

The project provides a Gradio web UI served with FastAPI, authenticated RAG API
endpoints, and an MCP adapter for compatible clients. These are different ways
to use the project; they do not change the core retrieval idea.

## 5. Evaluation in this project

Evaluation checks behavior against known examples; it is not proof that every
answer is correct.

- A retrieval-only mode records ranked results without making LLM calls.
- The native RAGAS runner uses a golden set of 12 questions and reference
  answers. It supports faithfulness, answer relevancy, context precision,
  context recall, and answer correctness. Metrics that need reference answers
  depend on the golden-set data.
- The separate batch evaluation includes custom or proxy measurements. Do not
  describe a proxy as a native RAGAS result.

When presenting a score, name the exact metric, dataset, corpus, and evaluation
path. Do not call retrieval precision or a faithfulness proxy “overall
accuracy.” The native evaluation's context-only prompt is also not a complete
end-to-end score for the app's Document Chat workflow.

**Good interview wording:** “I added a golden-set evaluation so I can compare
retrieval and answer behavior on known questions. The score only describes
that dataset and evaluation setup; I would need broader data and repeated
testing before making a general quality claim.”

## 6. A simple walkthrough to practice

> “A user uploads research documents and builds an index. I extract the text
> locally and split it into chunks. When the user asks a question, I search
> the index with both embeddings and BM25, combine and rerank the results, and
> give the best passages to the LLM. Document Chat makes one retrieval pass
> and one generation call, then shows the sources. For broader research tasks,
> there is a separate multi-step workflow. I also have evaluation scripts to
> test the system on a small golden set.”

### Quick self-check

1. Why use RAG instead of sending every document to the LLM?
2. What is the difference between dense search and BM25?
3. Why split text into chunks?
4. Why does parent-child chunking retrieve a child but provide a parent?
5. What does RRF do?
6. Which workflow is single-pass, and which one is multi-step?
7. What can an evaluation score tell you—and what can it not tell you?

## 7. Honest limitations to remember

- Scanned PDFs need OCR, which is not configured for local text extraction.
- Retrieval can return irrelevant or incomplete evidence; hybrid search does
  not guarantee the right result.
- Evaluation coverage is limited by the golden set and corpus used.
- The single-pass chat path does not independently verify every generated
  claim with a second LLM judge.
- Local document-upload chunking options do not automatically apply to every
  ingestion path.

## Project reference

- [README](./README.md) for setup, supported files, API, MCP, and evaluation
  commands.
- [MCP_README](./MCP_README.md) for MCP tools and resources.
- [evaluation/ragas_native.py](./evaluation/ragas_native.py) for native RAGAS
  metric definitions and runner details.
