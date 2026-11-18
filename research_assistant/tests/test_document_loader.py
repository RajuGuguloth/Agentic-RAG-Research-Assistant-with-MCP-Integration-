import sys
import tempfile
import unittest
from pathlib import Path

import fitz
from docx import Document as WordDocument
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from etl.document_loader import load_documents_from_files


class TestLocalDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extracts_text_from_supported_formats_locally(self):
        text_path = self.root / "notes.txt"
        text_path.write_text("Local text content", encoding="utf-8")

        html_path = self.root / "notes.html"
        html_path.write_text(
            "<html><body><h1>Local heading</h1><p>Visible text</p>"
            "<script>secret = 'ignore';</script></body></html>",
            encoding="utf-8",
        )

        docx_path = self.root / "notes.docx"
        word = WordDocument()
        word.add_paragraph("Word paragraph")
        table = word.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "Word table"
        word.save(docx_path)

        pptx_path = self.root / "slides.pptx"
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
        textbox.text = "Slide content"
        presentation.save(pptx_path)

        pdf_path = self.root / "paper.pdf"
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PDF page content")
        pdf.save(pdf_path)
        pdf.close()

        documents = load_documents_from_files(
            [str(text_path), str(html_path), str(docx_path), str(pptx_path), str(pdf_path)]
        )
        content = "\n".join(doc.page_content for doc in documents)

        self.assertIn("Local text content", content)
        self.assertIn("Local heading", content)
        self.assertIn("Visible text", content)
        self.assertNotIn("ignore", content)
        self.assertIn("Word paragraph", content)
        self.assertIn("Word table", content)
        self.assertIn("Slide content", content)
        self.assertIn("PDF page content", content)
        self.assertEqual(
            {doc.metadata["document_type"] for doc in documents},
            {"txt", "html", "docx", "pptx", "pdf"},
        )
        slide_doc = next(doc for doc in documents if doc.metadata["document_type"] == "pptx")
        self.assertEqual(slide_doc.metadata["page"], 1)

    def test_scanned_pdf_fails_with_local_ocr_explanation(self):
        path = self.root / "scanned.pdf"
        pdf = fitz.open()
        pdf.new_page()
        pdf.save(path)
        pdf.close()

        with self.assertRaisesRegex(ValueError, "local OCR is not configured"):
            load_documents_from_files([str(path)])

    def test_unsupported_extension_is_reported(self):
        path = self.root / "file.rtf"
        path.write_text("text", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unsupported document format"):
            load_documents_from_files([str(path)])


if __name__ == "__main__":
    unittest.main()
