import io
from pathlib import Path

import pytest
from docx import Document

from rag_support import parsing
from rag_support.parsing import chunk_document, normalize, parse_document, parse_tesseract_tsv

FIXTURES = Path(__file__).parent.parent / "fixtures" / "documents"


def assert_offsets(doc, chunks):
    for chunk in chunks:
        assert normalize(doc.text[chunk.start:chunk.end]) == chunk.text


def test_markdown_sections_title_and_offsets():
    doc = parse_document("guide.md", b"# Guide\n\nIntro text.\n\n## Billing\n\nPlans change monthly.\n\n### Refunds\n\nAnnual plans refundable.\n\n```\n# not a heading\n```\n")
    chunks = chunk_document(doc)
    assert doc.title == "Guide"
    assert [c.section for c in chunks] == ["Guide", "Guide > Billing", "Guide > Billing > Refunds"]
    assert "# not a heading" in chunks[-1].text
    assert_offsets(doc, chunks)


def test_chunks_are_bounded_overlap_and_stay_in_section():
    sentence = "Agents can resolve tickets quickly with macros and saved replies. "
    body = "\n\n".join(sentence * 3 for _ in range(12))
    doc = parse_document("long.md", f"# A\n\n{body}\n\n# B\n\nShort tail.".encode())
    chunks = chunk_document(doc, max_chars=400, overlap=60)
    assert all(c.end - c.start <= 400 for c in chunks)
    assert chunks[-1].section == "B" and chunks[-1].text == "Short tail."
    a_chunks = [c for c in chunks if c.section == "A"]
    assert len(a_chunks) > 3
    assert any(nxt.start < prev.end for prev, nxt in zip(a_chunks, a_chunks[1:])), "expected overlap"
    assert_offsets(doc, chunks)


def test_single_long_paragraph_is_split():
    doc = parse_document("one.txt", ("word " * 2000).encode())
    chunks = chunk_document(doc, max_chars=500, overlap=50)
    assert len(chunks) > 5 and all(c.end - c.start <= 500 for c in chunks)
    assert_offsets(doc, chunks)


def test_html_drops_navigation_and_tracks_headings():
    html = b"<html><head><title>Help</title><meta name='version' content='v3'></head><body><nav>Menu</nav><h1>Help</h1><h2>Hours</h2><ul><li><p>Open daily.</p></li></ul><table><tr><th>Plan</th><th>Seats</th></tr><tr><td>Basic</td><td>3</td></tr></table><script>x()</script></body></html>"
    doc = parse_document("help.html", html)
    chunks = chunk_document(doc)
    assert doc.title == "Help" and doc.version == "v3"
    assert "Menu" not in doc.text and "x()" not in doc.text
    assert chunks[0].section == "Help > Hours"
    assert "Open daily." in chunks[0].text and "Basic | 3" in chunks[0].text
    assert doc.text.count("Open daily.") == 1
    assert_offsets(doc, chunks)


def test_docx_keeps_document_order_and_metadata():
    document = Document()
    document.core_properties.title = "FAQ"
    document.core_properties.version = "7"
    document.add_heading("Setup", 1)
    document.add_paragraph("Before table.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Slack", "Basic"
    document.add_paragraph("After table.")
    buffer = io.BytesIO()
    document.save(buffer)
    doc = parse_document("faq.docx", buffer.getvalue())
    chunks = chunk_document(doc)
    assert (doc.title, doc.version) == ("FAQ", "7")
    assert chunks[0].section == "Setup"
    text = chunks[0].text
    assert text.index("Before table.") < text.index("Slack | Basic") < text.index("After table.")


def test_searchable_pdf_pages():
    doc = parse_document("security.pdf", (FIXTURES / "security.pdf").read_bytes())
    chunks = chunk_document(doc)
    assert doc.title == "Acme Desk security overview"
    assert {c.page for c in chunks} == {1, 2}
    assert any("14 days" in c.text and c.page == 2 for c in chunks)
    assert chunks[0].location == "page 1"
    assert_offsets(doc, chunks)


def test_low_quality_ocr_is_flagged_but_indexed(monkeypatch):
    monkeypatch.setattr(parsing, "_ocr_page", lambda data, page: ("Tr1al 2l days fr ee", 31.0))
    doc = parse_document("onboarding-scan.pdf", (FIXTURES / "onboarding-scan.pdf").read_bytes())
    chunks = chunk_document(doc)
    assert doc.warnings == [{"page": 1, "reason": "little text recognized; mean OCR confidence 31", "ocr": True}]
    assert chunks and chunks[0].low_ocr_quality is True


def test_missing_ocr_tools_give_clear_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("pdftoppm")
    monkeypatch.setattr(parsing.subprocess, "run", missing)
    with pytest.raises(ValueError, match="OCR requires"):
        parse_document("onboarding-scan.pdf", (FIXTURES / "onboarding-scan.pdf").read_bytes())


def test_tesseract_tsv_rebuilds_lines_and_confidence():
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = ["5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tHello", "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t70\tworld", "4\t1\t1\t1\t1\t0\t0\t0\t1\t1\t-1\t", "5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t80\tNext"]
    text, confidence = parse_tesseract_tsv(header + "\n".join(rows))
    assert text == "Hello world\n\nNext"
    assert confidence == pytest.approx(80)


def test_unsupported_format():
    with pytest.raises(ValueError, match="Supported formats"):
        parse_document("sheet.xlsx", b"data")
