"""Local text extraction for supported document formats."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import List

from langchain_core.documents import Document

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".docx", ".pptx"}


class _TextHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "p", "section", "table", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _document(text: str, path: Path, **metadata) -> Document:
    content = text.strip()
    if not content:
        raise ValueError(f"No extractable text found in {path.name}.")
    return Document(
        page_content=content,
        metadata={
            "source": path.name,
            "file_path": str(path),
            "document_type": path.suffix.lower().lstrip("."),
            **metadata,
        },
    )


def _load_pdf(path: Path) -> List[Document]:
    from etl.pdf_loader import NoExtractablePDFTextError, load_pdf

    try:
        pages = load_pdf(str(path))
    except NoExtractablePDFTextError as exc:
        raise ValueError(
            f"{path.name} contains no extractable text. Scanned or image-only PDFs "
            "are not supported because local OCR is not configured; no external OCR "
            "service was called."
        ) from exc

    return [
        Document(
            page_content=page.page_content,
            metadata={
                **page.metadata,
                "document_type": "pdf",
            },
        )
        for page in pages
    ]


def _load_html(path: Path) -> List[Document]:
    parser = _TextHTMLParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    parser.close()
    return [_document(" ".join(parser.parts), path)]


def _load_text(path: Path) -> List[Document]:
    return [_document(path.read_text(encoding="utf-8-sig"), path)]


def _load_docx(path: Path) -> List[Document]:
    from docx import Document as WordDocument

    word_doc = WordDocument(str(path))
    parts = [paragraph.text for paragraph in word_doc.paragraphs if paragraph.text.strip()]
    for table in word_doc.tables:
        parts.extend(
            " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            for row in table.rows
        )
    return [_document("\n".join(parts), path)]


def _load_pptx(path: Path) -> List[Document]:
    from pptx import Presentation

    presentation = Presentation(str(path))
    documents = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            elif getattr(shape, "has_text_frame", False) and shape.text.strip():
                parts.append(shape.text.strip())
        if parts:
            documents.append(
                _document(
                    f"Slide {slide_number}\n" + "\n".join(parts),
                    path,
                    page=slide_number,
                    slide=slide_number,
                )
            )
    if not documents:
        raise ValueError(f"No extractable slide text found in {path.name}.")
    return documents


def load_documents_from_files(
    file_paths: List[str],
    progress=None,
    *,
    with_vision: bool = False,
) -> List[Document]:
    """Extract text locally from PDF, HTML, TXT, DOCX, and PPTX files."""
    if not file_paths:
        raise ValueError("No document files provided.")

    paths = [Path(file_path) for file_path in file_paths]
    unsupported = [
        path.name
        for path in paths
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS
    ]
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(
            f"Unsupported document format for {unsupported[0]}. "
            f"Supported extensions: {supported}."
        )
    if with_vision and any(path.suffix.lower() != ".pdf" for path in paths):
        raise ValueError("Page-image vision analysis can only be used with PDF files.")

    documents: List[Document] = []
    total = len(paths)
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        if progress:
            progress(index / total, desc=f"Extracting {path.name} locally...")

        extension = path.suffix.lower()
        if extension == ".pdf":
            loaded = _load_pdf(path)
            if with_vision:
                from etl.vision_loader import enrich_pdf_documents_with_vision

                loaded = enrich_pdf_documents_with_vision(loaded, progress=progress)
        elif extension in {".html", ".htm"}:
            loaded = _load_html(path)
        elif extension == ".txt":
            loaded = _load_text(path)
        elif extension == ".docx":
            loaded = _load_docx(path)
        else:
            loaded = _load_pptx(path)
        documents.extend(loaded)

    if progress:
        progress(1.0, desc=f"Extracted {len(documents)} document sections locally.")
    return documents
