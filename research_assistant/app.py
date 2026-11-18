import asyncio
import logging
from typing import List, Tuple

import gradio as gr
from fastapi import FastAPI
from config import Config
from etl.document_loader import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    load_documents_from_files,
)
try:
    from etl.arxiv_loader import search_and_load
except ImportError:
    def search_and_load(*args, **kwargs):
        raise NotImplementedError("arxiv_loader not available — use Deep-Read with arXiv URL instead.")
from etl.chunker import (
    CHUNKING_STRATEGIES,
    STRATEGY_PARENT_CHILD_SEMANTIC,
    chunk_documents_parent_child,
)
from retrieval.runtime import get_hybrid_retriever, rebuild_index
from orchestration.document_chat import chat_turn, _append_turn
from orchestration.deep_read.orchestrator import run_deep_read
from orchestration.simple_summarizer import summarize_from_index, format_llm_error
from observability.feedback import append_feedback
from api.feedback_router import router as feedback_router
from api.rag_router import router as rag_router
from memory.user_memory import resolve_user_id, start_session, update_from_feedback
import uuid

# ── Global State ──────────────────────────────────────────────────────────────
DEFAULT_SUMMARY_GOAL = (
    "Summarize the uploaded document in simple, clear language. Include:\n"
    "- What the document is about (main topic)\n"
    "- Key points and findings\n"
    "- Important conclusions or recommendations\n"
    "Use short sections and bullet points. Avoid jargon."
)

hybrid_retriever = get_hybrid_retriever()
log = logging.getLogger("rag_app")


# ── Indexing Functions ────────────────────────────────────────────────────────
class DocumentIndexingFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"Document indexing failed during {stage}")


def _run_document_indexing(files, chunking_strategy, with_vision, progress):
    if not files:
        raise DocumentIndexingFailure(
            "input validation",
            ValueError("Select at least one document."),
        )

    stage = (
        "local document extraction and PDF vision analysis"
        if with_vision
        else "local document text extraction"
    )
    try:
        file_paths = [f.name for f in files]
        progress(0.1, desc="Extracting document text locally...")
        raw_docs = load_documents_from_files(
            file_paths,
            progress=progress,
            with_vision=with_vision,
        )
        if not raw_docs:
            raise ValueError(
                "No readable document text was extracted. Scanned or image-only "
                "PDFs are unsupported because local OCR is not configured."
            )

        stage = "document chunking"
        progress(0.3, desc="Chunking extracted text...")
        parent_docs, child_docs = chunk_documents_parent_child(
            raw_docs,
            progress=progress,
            strategy=chunking_strategy,
        )
        if not parent_docs or not child_docs:
            raise ValueError("Chunking produced no indexable content.")

        stage = "index build and persistence"
        progress(0.7, desc="Building and saving the new index...")
        result = rebuild_index(parent_docs, child_docs, replace=True)
        progress(1.0, desc="Indexing complete.")
        return result
    except DocumentIndexingFailure:
        raise
    except Exception as exc:
        raise DocumentIndexingFailure(stage, exc) from exc


def _document_indexing_error_message(exc: DocumentIndexingFailure) -> str:
    cause = exc.cause
    status_code = None
    current = cause
    while current is not None:
        status_code = getattr(
            getattr(current, "response", None),
            "status_code",
            None,
        )
        if status_code is not None:
            break
        current = current.__cause__
    provider = _vision_provider_label()
    if status_code == 429:
        detail = f"The {provider} vision provider rate-limited the request (HTTP 429)."
    elif status_code in (401, 403):
        detail = f"The {provider} vision provider rejected the request (HTTP {status_code})."
    elif status_code is not None and status_code >= 500:
        detail = f"The {provider} vision provider returned HTTP {status_code}."
    elif isinstance(cause, ValueError):
        detail = str(cause)
    else:
        detail = f"{type(cause).__name__}; see the server log for details."
    return (
        f"❌ Document indexing failed during {exc.stage}: {detail} "
        "Any previous index was preserved."
    )


def _vision_provider_label() -> str:
    provider = Config.VISION_PROVIDER
    if provider == "auto":
        if Config.OPENAI_API_KEY:
            provider = "openai"
        elif Config.GEMINI_API_KEY:
            provider = "gemini"
        elif Config.LLM_BACKEND == "ollama":
            provider = "ollama"
    return {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "ollama": "Ollama",
    }.get(provider, "configured")


def build_index_from_documents(
    files,
    chunking_strategy: str = STRATEGY_PARENT_CHILD_SEMANTIC,
    with_vision: bool = False,
    progress=gr.Progress(),
) -> str:
    """Extract, chunk, and index supported uploaded documents."""
    try:
        result = _run_document_indexing(files, chunking_strategy, with_vision, progress)
    except DocumentIndexingFailure as exc:
        log.error(
            "Document indexing failed during %s",
            exc.stage,
            exc_info=(type(exc.cause), exc.cause, exc.cause.__traceback__),
        )
        return _document_indexing_error_message(exc)

    vision_note = (
        " Optional PDF vision analysis was enabled."
        if with_vision
        else " Text extraction ran locally; scanned/image-only PDFs need local OCR, which is not configured."
    )
    return (
        f"✅ Index ready ({result['total_children']} chunks) using "
        f"**{CHUNKING_STRATEGIES[chunking_strategy]['label']}**.{vision_note} "
        f"Go to **🔍 Research** to summarize or chat with these documents."
    )


def build_index_from_arxiv(query: str, max_results: int, progress=gr.Progress()) -> str:
    """Fetch arXiv papers, chunk, and build Hybrid index."""
    try:
        progress(0.1, desc="Fetching arXiv papers...")
        raw_docs = search_and_load(query, max_results=max_results, progress=progress)
        
        progress(0.3, desc="Chunking documents (Parent-Child + Semantic)...")
        parent_docs, child_docs = chunk_documents_parent_child(raw_docs, progress=progress)
        
        progress(0.7, desc="Building Dense and Sparse Indexes...")
        result = rebuild_index(parent_docs, child_docs, replace=True)
        
        progress(1.0, desc="Indexing Complete!")
        return f"Successfully indexed {result['total_children']} child chunks from {len(raw_docs)} arXiv papers."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error indexing arXiv: {e}"


# ── Research Orchestration ────────────────────────────────────────────────────
async def run_simple_summary_workflow(goal: str) -> Tuple[str, str, str, str]:
    """Fast path: retrieve + one LLM call for an indexed document summary."""
    if not hybrid_retriever.dense.is_ready:
        return (
            "Error: Index not built. Upload a document under **📚 Data Sources** and click **Build Index** first.",
            "",
            "",
            "",
        )

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return f"Configuration Error: {e}", "", "", ""

    try:
        result = await summarize_from_index(goal, llm, hybrid_retriever)
        chart_html = (
            "<div style='padding: 12px; background: #f0fdf4; border-radius: 8px;'>"
            "Simple summary mode — no chart generated."
            "</div>"
        )
        return (
            result["report"],
            chart_html,
            result["sub_queries"],
            result["findings"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return format_llm_error(e), "", "", ""


async def run_research_workflow(goal: str, max_queries: int) -> Tuple[str, str, str, str]:
    """
    Execute the full Agentic RAG workflow:
    Plan -> Parallel Research -> Synthesize -> Visualize
    """
    if not hybrid_retriever.dense.is_ready:
        return "Error: Index not built. Please build the index first from the Data sources tab.", "", "", ""

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return f"Configuration Error: {e}", "", "", ""

    try:
        from orchestration.planner import ResearchPlanner
        from orchestration.agent import run_parallel_research
        from orchestration.synthesizer import ReportSynthesizer
        from visualization.chart_generator import ChartGenerator

        # 1. Plan
        planner = ResearchPlanner(llm)
        sub_queries = await planner.plan(goal, max_queries=max_queries)
        sub_queries_md = "### Generated Sub-Queries\n" + "\n".join([f"- {q}" for q in sub_queries])

        # 2. Parallel Research
        results = await run_parallel_research(sub_queries, llm, hybrid_retriever)
        
        # Format intermediate findings
        findings_md = "### Intermediate Findings\n"
        for res in results:
            findings_md += f"**Q: {res['sub_query']}**\n\n{res['answer']}\n\n---\n\n"

        # 3. Synthesize
        synthesizer = ReportSynthesizer(llm)
        final_report = await synthesizer.synthesize(goal, results)

        # 4. Visualize
        chart_gen = ChartGenerator(llm)
        has_chart, img_b64, chart_msg = await chart_gen.extract_and_plot(final_report)
        
        chart_html = ""
        if has_chart and img_b64:
            chart_html = f"<div><h3>Auto-Generated Visualization</h3><p>{chart_msg}</p><img src='data:image/png;base64,{img_b64}' style='max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'/></div>"
        else:
            chart_html = f"<div style='padding: 20px; background: #f3f4f6; border-radius: 8px;'><i>{chart_msg}</i></div>"

        return final_report, chart_html, sub_queries_md, findings_md

    except Exception as e:
        import traceback
        traceback.print_exc()
        return format_llm_error(e), "", "", ""


def launch_research(goal: str, max_queries: int) -> Tuple[str, str, str, str]:
    """Gradio wrapper for full agentic workflow."""
    return asyncio.run(run_research_workflow(goal, max_queries))


def launch_simple_summary(goal: str = None) -> Tuple[str, str, str, str]:
    """Gradio wrapper for a simple indexed-document summary."""
    return asyncio.run(run_simple_summary_workflow(goal or DEFAULT_SUMMARY_GOAL))


def summarize_indexed_pdf(max_queries: int = 3) -> Tuple[str, str, str, str]:
    """Run the default document summary workflow against the current index."""
    return launch_simple_summary(DEFAULT_SUMMARY_GOAL)


def build_index_and_summarize_documents(
    files,
    max_queries: int = 3,
    chunking_strategy: str = STRATEGY_PARENT_CHILD_SEMANTIC,
    with_vision: bool = False,
    progress=gr.Progress(),
):
    """Index uploaded documents, then immediately generate a summary."""
    try:
        result = _run_document_indexing(
            files,
            chunking_strategy,
            with_vision,
            progress,
        )
    except DocumentIndexingFailure as exc:
        log.error(
            "Document indexing failed during %s",
            exc.stage,
            exc_info=(type(exc.cause), exc.cause, exc.cause.__traceback__),
        )
        return _document_indexing_error_message(exc), "", "", "", ""

    progress(1.0, desc="Generating summary...")
    report, chart, sub_queries, findings = launch_simple_summary()
    return (
        f"✅ Index ready ({result['total_children']} chunks) using "
        f"**{CHUNKING_STRATEGIES[chunking_strategy]['label']}**. "
        "Summary generation was started; see **🔍 Research → 📝 Final Report**.",
        report,
        chart,
        sub_queries,
        findings,
    )


# ── Document Chat (NotebookLM-style) ─────────────────────────────────────────
async def _run_chat_turn(message: str, history: list, session_id: str) -> tuple:
    if not hybrid_retriever.dense.is_ready:
        reply = "⚠️ No documents indexed yet. Upload files under **📚 Data Sources** and click **Build Index** first."
        return "", _append_turn(history, message, reply), "", "", session_id

    try:
        Config.validate()
        llm = Config.get_llm()
    except Exception as e:
        return "", _append_turn(history, message, f"Configuration error: {e}"), "", "", session_id

    sid = session_id or str(uuid.uuid4())
    user_id = resolve_user_id(session_id=sid)
    if Config.LEVEL5_ENABLED:
        start_session(user_id, sid)

    updated_history, sources, trace_id = await chat_turn(
        message, history or [], llm, hybrid_retriever, user_id=user_id
    )
    return "", updated_history, sources, trace_id, sid


def respond_to_chat(message: str, history: list, session_id: str) -> tuple:
    """Gradio wrapper for one chat turn."""
    if not message or not message.strip():
        return message, history or [], "", "", session_id or ""
    return asyncio.run(_run_chat_turn(message, history or [], session_id or ""))


def submit_chat_feedback(trace_id: str, rating: str, session_id: str) -> str:
    """Record thumbs up/down for the last chat answer (also available via POST /api/feedback)."""
    if not (trace_id or "").strip():
        return "Send a message first — no trace id yet."
    try:
        user_id = resolve_user_id(session_id=session_id or "")
        append_feedback(
            trace_id.strip(),
            rating,
            metadata={"path": "chat", "source": "gradio", "user_id": user_id},
        )
        if Config.LEVEL5_ENABLED:
            update_from_feedback(user_id, rating)
        label = "👍 helpful" if rating == "positive" else "👎 not helpful"
        return f"Thanks — feedback recorded ({label}). Trace: `{trace_id[:8]}...`"
    except Exception as exc:
        return f"Could not save feedback: {exc}"


def clear_chat(session_id: str) -> tuple:
    return [], "", "", "", str(uuid.uuid4()), ""


# ── Deep-Read (structured paper analysis) ─────────────────────────────────────
async def _run_deep_read_async(paper_url: str, pdf_file, progress=gr.Progress()):
    progress(0, desc="Starting Deep-Read…")
    try:
        Config.validate()
    except Exception as e:
        return f"Configuration error: {e}", None, None, None

    def prog_cb(fraction: float, msg: str):
        progress(fraction, desc=msg)

    try:
        if pdf_file is not None:
            path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
            result = await run_deep_read(path, progress=prog_cb, is_upload=True)
        elif paper_url and paper_url.strip():
            result = await run_deep_read(paper_url.strip(), progress=prog_cb, is_upload=False)
        else:
            return "Provide a paper URL (arXiv/PDF link) or upload a PDF.", None, None, None

        status = (
            f"✅ **Deep-Read complete** ({result.latency_sec:.0f}s)\n\n"
            f"- **Title:** {result.title}\n"
            f"- **Pages:** {result.page_count} · **Figures:** {result.figure_count} · **Tables:** {result.table_count}\n"
            f"- **Job ID:** `{result.job_id}`\n"
            f"- **Artifacts:** `{result.artifacts_dir}`"
        )
        return status, result.report_markdown, result.report_path, result.ppt_path
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return format_llm_error(exc), None, None, None


def run_deep_read_ui(paper_url: str, pdf_file, progress=gr.Progress()):
    return asyncio.run(_run_deep_read_async(paper_url, pdf_file, progress=progress))


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def create_ui():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    css = """
    :root {
        --app-radius: 16px;
        --app-border: #dbe4f0;
        --app-muted: #64748b;
        --app-surface: #f8fafc;
    }
    .app-shell { max-width: 1440px; margin: 0 auto; }
    .app-hero {
        border: 1px solid var(--app-border);
        border-radius: var(--app-radius);
        padding: 24px 28px;
        background: linear-gradient(135deg, #172554 0%, #1e3a8a 55%, #2563eb 100%);
        color: white;
        margin-bottom: 18px;
    }
    .app-hero h1 { margin: 0 0 6px 0; font-size: 30px; }
    .app-hero p { margin: 0; opacity: .84; font-size: 15px; }
    .app-badge {
        display: inline-block;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px;
        padding: 5px 11px;
        margin-top: 16px;
        font-size: 12px;
        background: rgba(255,255,255,.1);
    }
    .mode-card {
        border: 1px solid var(--app-border);
        border-radius: 14px;
        padding: 16px 18px;
        background: var(--app-surface);
        min-height: 100px;
    }
    .mode-card h3 { margin: 0 0 6px 0; }
    .mode-card p { color: var(--app-muted); margin: 0; font-size: 14px; }
    .section-note {
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        background: #eff6ff;
        border-radius: 8px;
        color: #1e3a8a;
    }
    .source-panel {
        border: 1px solid var(--app-border);
        border-radius: 12px;
        padding: 14px;
        background: var(--app-surface);
    }
    .architecture {
        border: 1px solid var(--app-border);
        border-radius: 16px;
        padding: 18px;
        background: var(--app-surface);
    }
    .architecture-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .architecture-subtitle {
        color: var(--app-muted);
        font-size: 13px;
        margin-bottom: 14px;
    }
    .arch-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 10px;
    }
    .arch-box {
        border: 1px solid #cbd5e1;
        border-radius: 11px;
        padding: 12px;
        background: white;
        text-align: center;
    }
    .arch-box strong { display: block; color: #172554; }
    .arch-box span { color: var(--app-muted); font-size: 12px; }
    .arch-shared {
        border-color: #93c5fd;
        background: #eff6ff;
    }
    .arch-engine {
        border-color: #86efac;
        background: #f0fdf4;
    }
    .arch-arrow {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        line-height: 20px;
        margin: 2px 0;
    }
    .arch-footer {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    @media (max-width: 700px) {
        .arch-row, .arch-footer { grid-template-columns: 1fr; }
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        theme=theme,
        css=css,
        title="Agentic RAG Research Assistant",
        elem_id="agentic-rag-app",
    ) as demo:
        gr.HTML(
            """
            <header class="app-hero">
                <h1>Agentic RAG Research Assistant</h1>
                <p>Search, understand, and research your documents with source-grounded answers.</p>
                <span class="app-badge">FAISS + BM25 · RRF · CrossEncoder</span>
            </header>
            """,
            elem_classes=["app-shell"],
        )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div class="mode-card">
                        <h3>💬 Document Chat</h3>
                        <p>Fast, single-pass answers with citations from your indexed documents.</p>
                    </div>
                    """
                )
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div class="mode-card">
                        <h3>🔍 Research Agent</h3>
                        <p>Break complex goals into sub-queries and synthesize a detailed report.</p>
                    </div>
                    """
                )
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div class="mode-card">
                        <h3>📚 Build Knowledge</h3>
                        <p>Upload files, choose chunking, and build the shared retrieval index.</p>
                    </div>
                    """
                )

        with gr.Accordion("How this application works", open=False, elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="architecture">
                    <div class="architecture-title">Single shared RAG architecture</div>
                    <div class="architecture-subtitle">
                        Both user workflows reuse the same indexed document knowledge and retrieval engine.
                    </div>

                    <div class="arch-row">
                        <div class="arch-box">
                            <strong>💬 Normal Document Chat</strong>
                            <span>One retrieval → one generation</span>
                        </div>
                        <div class="arch-box">
                            <strong>🔍 Research Agent</strong>
                            <span>Plan → parallel workers → synthesis</span>
                        </div>
                    </div>
                    <div class="arch-arrow">↓</div>

                    <div class="arch-box arch-shared">
                        <strong>📚 Shared Document Knowledge</strong>
                        <span>Document extraction → parent/child chunks → FAISS + BM25 indexes</span>
                    </div>
                    <div class="arch-arrow">↓</div>

                    <div class="arch-box arch-engine">
                        <strong>⚙️ Shared RAG Retrieval Engine</strong>
                        <span>Hybrid search → RRF → parent mapping → CrossEncoder → context assembly</span>
                    </div>
                    <div class="arch-arrow">↓</div>

                    <div class="arch-box">
                        <strong>🧠 LLM Services</strong>
                        <span>Answer generation · planning · relevance checks · groundedness checks · synthesis</span>
                    </div>

                    <div class="arch-footer">
                        <div class="arch-box">
                            <strong>🔌 MCP Adapter</strong>
                            <span>External MCP client → mcp_server.py → FastAPI RAG API</span>
                        </div>
                        <div class="arch-box">
                            <strong>📈 Evaluation</strong>
                            <span>Separate offline retrieval and answer-quality measurement</span>
                        </div>
                    </div>
                </div>
                """
            )

        with gr.Tabs():
            # Tab 1: Research Workflow
            with gr.TabItem("🔍 Research"):
                index_ready = "✅ Index loaded — ready to summarize" if hybrid_retriever.dense.is_ready else "⚠️ No index yet — upload a document under **📚 Data Sources** first"
                gr.Markdown(
                    f"""
                    ### Research workspace
                    Use this mode for comparison, multi-part questions, and long-form research.

                    <div class="section-note">
                    <strong>Current index:</strong> {index_ready}<br>
                    The Research Agent plans sub-queries, runs parallel retrieval, checks evidence, and synthesizes a report.
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        goal_input = gr.Textbox(
                            label="Research goal",
                            value=DEFAULT_SUMMARY_GOAL,
                            lines=5,
                            placeholder="Example: Compare the paper's method, experiments, limitations, and future work.",
                        )
                    with gr.Column(scale=1):
                        max_queries_slider = gr.Slider(
                            minimum=1, maximum=6, value=3, step=1,
                            label="Maximum sub-queries",
                            info="More sub-queries improve coverage but increase LLM calls and latency.",
                        )
                        summarize_btn = gr.Button("Quick summary", variant="secondary")
                        research_btn = gr.Button("Run research", variant="primary")

                with gr.Tabs():
                    with gr.TabItem("📝 Final Report"):
                        report_output = gr.Markdown()
                    with gr.TabItem("📊 Visualizations"):
                        chart_output = gr.HTML()
                    with gr.TabItem("🧠 Agent Thinking"):
                        sub_queries_output = gr.Markdown()
                        findings_output = gr.Markdown()

                research_btn.click(
                    fn=launch_research,
                    inputs=[goal_input, max_queries_slider],
                    outputs=[report_output, chart_output, sub_queries_output, findings_output]
                )
                summarize_btn.click(
                    fn=summarize_indexed_pdf,
                    inputs=[max_queries_slider],
                    outputs=[report_output, chart_output, sub_queries_output, findings_output]
                )

            # Tab 2: Document Chat
            with gr.TabItem("💬 Chat with Documents"):
                chat_index_ready = (
                    "✅ Documents indexed — ask questions about them"
                    if hybrid_retriever.dense.is_ready
                    else "⚠️ Index documents first under **📚 Data Sources**"
                )
                gr.Markdown(
                    f"""
                    ### Ask your documents
                    <div class="section-note">
                    <strong>{chat_index_ready}</strong><br>
                    Each turn uses the shared hybrid retriever once, then generates one source-grounded answer.
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=480,
                        )
                        chat_input = gr.Textbox(
                            label="Your question",
                            placeholder="Ask anything about your indexed documents...",
                            lines=2,
                        )
                        gr.Examples(
                            examples=[
                                ["What is this document mainly about?"],
                                ["Explain the key method in simple language."],
                                ["What limitations does the author mention?"],
                            ],
                            inputs=chat_input,
                            label="Try a question",
                        )
                        with gr.Row():
                            chat_send_btn = gr.Button("Ask document", variant="primary")
                            chat_clear_btn = gr.Button("New conversation", variant="secondary")
                    with gr.Column(scale=1):
                        chat_sources = gr.Markdown(
                            label="Sources for last answer",
                            elem_classes=["source-panel"],
                        )
                        chat_trace_id = gr.Textbox(
                            label="Trace ID",
                            interactive=False,
                            info="Useful when reporting an answer problem.",
                            placeholder="Generated after an answer",
                        )
                        feedback_status = gr.Markdown("")
                        with gr.Row():
                            feedback_up = gr.Button("👍 Helpful")
                            feedback_down = gr.Button("👎 Not helpful")

                chat_session_id = gr.State(str(uuid.uuid4()))
                chat_msg = [chat_input, chatbot, chat_session_id]
                chat_out = [chat_input, chatbot, chat_sources, chat_trace_id, chat_session_id]

                chat_send_btn.click(fn=respond_to_chat, inputs=chat_msg, outputs=chat_out)
                chat_input.submit(fn=respond_to_chat, inputs=chat_msg, outputs=chat_out)
                chat_clear_btn.click(
                    fn=clear_chat,
                    inputs=[chat_session_id],
                    outputs=[chatbot, chat_input, chat_sources, chat_trace_id, chat_session_id, feedback_status],
                )

                feedback_up.click(
                    fn=lambda tid, sid: submit_chat_feedback(tid, "positive", sid),
                    inputs=[chat_trace_id, chat_session_id],
                    outputs=[feedback_status],
                )
                feedback_down.click(
                    fn=lambda tid, sid: submit_chat_feedback(tid, "negative", sid),
                    inputs=[chat_trace_id, chat_session_id],
                    outputs=[feedback_status],
                )

            # Tab 3: Deep-Read
            with gr.TabItem("📖 Deep-Read"):
                gr.Markdown(
                    """
                    ### Deep-read a paper
                    Paste an **arXiv link**, **PDF URL**, or **upload a PDF**. Deep-Read will:
                    1. Extract text, **handwriting**, figures, charts, and tables from every page
                    2. Analyze section-by-section: metadata → problem → method → figures → results → limitations
                    3. Merge into a **detailed report** + **PowerPoint** with paper images

                    *This is a separate, more expensive workflow from normal Document Chat. Page images are analyzed by your configured vision provider; cloud providers receive the page images.*
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        deep_read_url = gr.Textbox(
                            label="Paper URL",
                            placeholder="https://arxiv.org/abs/1706.03762 or direct PDF link",
                            lines=1,
                        )
                        deep_read_file = gr.File(
                            label="Or upload PDF",
                            file_types=[".pdf"],
                            file_count="single",
                        )
                        deep_read_btn = gr.Button("Run deep-read", variant="primary")
                        deep_read_status = gr.Markdown("")
                    with gr.Column(scale=3):
                        deep_read_report = gr.Markdown(label="Structured Report")

                with gr.Row():
                    deep_read_md_dl = gr.File(label="Download Report (.md)", interactive=False)
                    deep_read_ppt_dl = gr.File(label="Download Slides (.pptx)", interactive=False)

                deep_read_btn.click(
                    fn=run_deep_read_ui,
                    inputs=[deep_read_url, deep_read_file],
                    outputs=[deep_read_status, deep_read_report, deep_read_md_dl, deep_read_ppt_dl],
                )

            # Tab 4: Data Sources (Indexing)
            with gr.TabItem("📚 Data Sources"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            ### Build your document knowledge base
                            Upload text-based PDF, HTML, TXT, DOCX, or PPTX files. Text extraction runs locally.

                            **Recommended flow:** upload → choose chunking → build the index → ask questions in **Document Chat** or run a **Research Agent** report.

                            Scanned/image-only PDFs are not supported by local OCR. Optional page-image vision analysis is separate from text indexing and may send images to the configured provider.
                            """
                        )
                        pdf_file_input = gr.File(
                            label="Upload documents",
                            file_count="multiple",
                            file_types=sorted(SUPPORTED_DOCUMENT_EXTENSIONS),
                        )
                        chunking_strategy = gr.Dropdown(
                            choices=[
                                (details["label"], strategy)
                                for strategy, details in CHUNKING_STRATEGIES.items()
                            ],
                            value=STRATEGY_PARENT_CHILD_SEMANTIC,
                            label="Chunking strategy",
                            info="Choose how extracted document text is divided before it is indexed.",
                        )
                        chunking_explanation = gr.Markdown(
                            CHUNKING_STRATEGIES[STRATEGY_PARENT_CHILD_SEMANTIC]["description"]
                        )
                        chunking_strategy.change(
                            fn=lambda strategy: CHUNKING_STRATEGIES[strategy]["description"],
                            inputs=[chunking_strategy],
                            outputs=[chunking_explanation],
                        )
                        vision_analysis = gr.Checkbox(
                            label="Optional PDF page-image vision analysis",
                            value=False,
                            info=(
                                "PDF only. Sends page images to the configured vision provider; "
                                "this is not OCR and does not support scanned PDFs."
                            ),
                        )
                        index_and_summarize_btn = gr.Button(
                            "Build index + summarize",
                            variant="primary",
                        )
                        index_pdf_btn = gr.Button("Build index only")
                        pdf_status = gr.Textbox(label="Status", interactive=False)

                        index_pdf_btn.click(
                            fn=build_index_from_documents,
                            inputs=[pdf_file_input, chunking_strategy, vision_analysis],
                            outputs=[pdf_status]
                        )

                        index_and_summarize_btn.click(
                            fn=build_index_and_summarize_documents,
                            inputs=[
                                pdf_file_input,
                                max_queries_slider,
                                chunking_strategy,
                                vision_analysis,
                            ],
                            outputs=[pdf_status, report_output, chart_output, sub_queries_output, findings_output],
                        )
                        
                    with gr.Column():
                        gr.Markdown("### Fetch from arXiv")
                        arxiv_query_input = gr.Textbox(
                            label="arXiv Search Query",
                            value="cat:cs.AI and LLM",
                            placeholder="e.g., all:transformer"
                        )
                        arxiv_max_slider = gr.Slider(
                            minimum=5, maximum=50, value=20, step=5,
                            label="Max Papers to Fetch"
                        )
                        index_arxiv_btn = gr.Button("Fetch & Build Hybrid Index")
                        arxiv_status = gr.Textbox(label="Status", interactive=False)
                        
                        index_arxiv_btn.click(
                            fn=build_index_from_arxiv,
                            inputs=[arxiv_query_input, arxiv_max_slider],
                            outputs=[arxiv_status]
                        )
                        
            # Tab 5: Configuration
            with gr.TabItem("⚙️ Configuration"):
                gr.Markdown(
                    f"""
                    ### Current Settings (from `.env`)
                    - **LLM Backend**: {Config.LLM_BACKEND}
                    - **LLM Provider (auto)**: {Config.LLM_PROVIDER}
                    - **Gemini Model**: {Config.GEMINI_MODEL}
                    - **OpenAI Model**: {Config.OPENAI_MODEL}
                    - **Vision Provider**: {Config.VISION_PROVIDER}
                    - **Vision Model**: {Config.VISION_MODEL or "provider default"}
                    - **Embedding Model**: {Config.EMBEDDING_MODEL}
                    - **Perplexity Model**: {Config.PERPLEXITY_MODEL}
                    - **Chunk Size**: {Config.CHUNK_SIZE} chars (Overlap: {Config.CHUNK_OVERLAP})
                    - **Hybrid RRF K**: {Config.RRF_K}

                    **BYOK:** Set `OPENAI_API_KEY` or `GEMINI_API_KEY` for page vision. Configure `VISION_PROVIDER` and `VISION_MODEL` to select a separate image-capable model.

                    *Page images are sent to the configured cloud vision provider. For local-only analysis, use `VISION_PROVIDER=ollama` with a vision-capable Ollama model. Change settings in `.env` and restart the app.*
                    """
                )

    return demo


def create_app():
    """FastAPI app with Gradio UI and /api/feedback endpoint."""
    demo = create_ui()
    api = FastAPI(
        title="Agentic RAG Research Assistant",
        description="Hybrid RAG with Gemini verification and feedback API",
    )
    api.include_router(feedback_router)
    api.include_router(rag_router)
    return gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    application = create_app()
    uvicorn.run(application, host="0.0.0.0", port=7860)
