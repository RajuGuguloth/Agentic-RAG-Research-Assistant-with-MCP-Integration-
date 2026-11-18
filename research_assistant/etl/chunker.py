"""
ETL — Text Chunker
Splits LangChain Documents using selectable recursive and semantic strategies.
"""

from typing import List, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config

STRATEGY_PARENT_CHILD_SEMANTIC = "parent_child_semantic"
STRATEGY_RECURSIVE_FIXED = "recursive_fixed"
STRATEGY_SEMANTIC_ONLY = "semantic_only"

CHUNKING_STRATEGIES = {
    STRATEGY_PARENT_CHILD_SEMANTIC: {
        "label": "Parent-child + semantic (recommended)",
        "description": (
            "Creates larger parent passages, then groups related sentences into "
            "smaller semantic child chunks for search. Search hits can be expanded "
            "back to their parent passage."
        ),
    },
    STRATEGY_RECURSIVE_FIXED: {
        "label": "Recursive fixed-size",
        "description": (
            "Splits text near paragraph and sentence boundaries, targeting the "
            "configured character size and overlap. Simple and predictable; it "
            "does not compare sentence meaning."
        ),
    },
    STRATEGY_SEMANTIC_ONLY: {
        "label": "Semantic chunks only",
        "description": (
            "Uses local embeddings to group nearby sentences by meaning. Chunk "
            "lengths can vary, and this option does not first create larger parent "
            "passages."
        ),
    },
}


def _get_embeddings():
    """Lazy load embeddings to avoid loading multiple times."""
    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)


def _chunk_image_document(doc: Document) -> Tuple[Document, Document]:
    """Handle a single image document without text splitting."""
    parent_id = str(uuid4())
    image_path = doc.metadata.get("file_path") or doc.metadata.get("image_path", "")

    base_metadata = {
        **doc.metadata,
        "image_path": image_path,
        "parent_id": parent_id,
    }

    parent = Document(
        page_content="",
        metadata={**base_metadata, "doc_type": "parent"},
    )
    child = Document(
        page_content="",
        metadata={**base_metadata, "doc_type": "child"},
    )
    return parent, child


def chunk_documents_parent_child(
    documents: List[Document],
    parent_chunk_size: int = 1500,
    parent_chunk_overlap: int = 200,
    progress=None,
    strategy: str = STRATEGY_PARENT_CHILD_SEMANTIC,
) -> Tuple[List[Document], List[Document]]:
    """
    Split documents with the selected strategy while preserving parent-child IDs.

    Args:
        documents: List of LangChain Documents.
        parent_chunk_size: Max characters for parent chunk.
        parent_chunk_overlap: Overlap for parent chunk.
        progress: Optional Gradio progress.
        strategy: One of the keys in CHUNKING_STRATEGIES.

    Returns:
        (parent_docs, child_docs)
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unsupported chunking strategy {strategy!r}. "
            f"Choose from: {', '.join(CHUNKING_STRATEGIES)}."
        )

    image_docs = [d for d in documents if d.metadata.get("content_type") == "image"]
    text_docs = [d for d in documents if d.metadata.get("content_type") != "image"]

    parent_docs: List[Document] = []
    child_docs: List[Document] = []

    for doc in image_docs:
        parent, child = _chunk_image_document(doc)
        parent.metadata["chunking_strategy"] = strategy
        child.metadata["chunking_strategy"] = strategy
        parent_docs.append(parent)
        child_docs.append(child)

    if not text_docs:
        if progress:
            progress(1.0, desc=f"Created {len(parent_docs)} parents and {len(child_docs)} children.")
        print(
            f"[chunker] {strategy}: {len(documents)} raw docs -> "
            f"{len(parent_docs)} parents -> {len(child_docs)} children."
        )
        return parent_docs, child_docs

    if strategy == STRATEGY_RECURSIVE_FIXED:
        if progress:
            progress(0.3, desc="Generating recursive fixed-size chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(text_docs)
        for chunk in chunks:
            parent_id = str(uuid4())
            metadata = {
                **chunk.metadata,
                "parent_id": parent_id,
                "chunking_strategy": strategy,
            }
            parent_docs.append(
                Document(page_content=chunk.page_content, metadata={**metadata, "doc_type": "parent"})
            )
            child_docs.append(
                Document(page_content=chunk.page_content, metadata={**metadata, "doc_type": "child"})
            )
    elif strategy == STRATEGY_SEMANTIC_ONLY:
        if progress:
            progress(0.3, desc="Initializing semantic chunker (loading embeddings)...")
        semantic_chunker = SemanticChunker(
            _get_embeddings(), breakpoint_threshold_type="percentile"
        )
        total_documents = len(text_docs)
        if progress:
            progress(0.5, desc="Generating semantic chunks...")
        for i, doc in enumerate(text_docs):
            for text in semantic_chunker.split_text(doc.page_content):
                if not text.strip():
                    continue
                parent_id = str(uuid4())
                metadata = {
                    **doc.metadata,
                    "parent_id": parent_id,
                    "chunking_strategy": strategy,
                }
                parent_docs.append(
                    Document(page_content=text, metadata={**metadata, "doc_type": "parent"})
                )
                child_docs.append(
                    Document(page_content=text, metadata={**metadata, "doc_type": "child"})
                )
            if progress:
                progress(
                    0.5 + (0.5 * ((i + 1) / total_documents)),
                    desc=f"Semantic chunking ({i + 1}/{total_documents} documents)...",
                )
    else:
        if progress:
            progress(0.3, desc="Generating parent chunks...")

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        text_parent_docs = parent_splitter.split_documents(text_docs)

        for parent in text_parent_docs:
            parent.metadata["parent_id"] = str(uuid4())
            parent.metadata["doc_type"] = "parent"
            parent.metadata["chunking_strategy"] = strategy

        parent_docs.extend(text_parent_docs)

        if progress:
            progress(0.5, desc="Initializing semantic chunker (loading embeddings)...")

        semantic_chunker = SemanticChunker(
            _get_embeddings(), breakpoint_threshold_type="percentile"
        )
        total_parents = len(text_parent_docs)
        if progress:
            progress(0.6, desc="Generating semantic child chunks...")

        for i, parent in enumerate(text_parent_docs):
            try:
                splits = semantic_chunker.split_text(parent.page_content)
                for split_text in splits:
                    if not split_text.strip():
                        continue
                    child = Document(
                        page_content=split_text,
                        metadata={**parent.metadata, "doc_type": "child"},
                    )
                    child_docs.append(child)
            except Exception as e:
                print(f"[chunker] Warning: Semantic split failed on parent {i}: {e}")
                child = Document(
                    page_content=parent.page_content,
                    metadata={**parent.metadata, "doc_type": "child"},
                )
                child_docs.append(child)

            if progress and i % 10 == 0:
                progress(
                    0.6 + (0.4 * (i / total_parents)),
                    desc=f"Semantic chunking ({i}/{total_parents} parents)...",
                )

    if progress:
        progress(1.0, desc=f"Created {len(parent_docs)} parents and {len(child_docs)} children.")
        
    print(
        f"[chunker] {strategy}: {len(documents)} raw docs -> "
        f"{len(parent_docs)} parents -> {len(child_docs)} children."
    )
    return parent_docs, child_docs
