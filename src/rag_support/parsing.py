"""Document parsing and chunking.

Every parser produces one ``extracted_text`` string per document plus passages that
point into it with character offsets. Chunks keep those offsets, so a reference can
be traced back to ``extracted_text[start:end]``.
"""

import csv
import io
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePath

from bs4 import BeautifulSoup
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED = {"pdf", "docx", "txt", "md", "markdown", "html", "htm"}


@dataclass
class Passage:
    start: int
    end: int
    page: int | None = None
    section: str | None = None
    low_ocr_quality: bool = False


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    page: int | None = None
    section: str | None = None
    low_ocr_quality: bool = False

    @property
    def location(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.section:
            parts.append(f"section {self.section}")
        return ", ".join(parts) or "document"


@dataclass
class ParsedDocument:
    title: str
    text: str
    passages: list[Passage]
    version: str | None = None
    warnings: list[dict] = field(default_factory=list)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _Builder:
    """Accumulates extracted text and heading-delimited passages."""

    def __init__(self):
        self.parts: list[str] = []
        self.pos = 0
        self.passages: list[Passage] = []
        self.headings: list[tuple[int, str]] = []
        self._open: Passage | None = None

    def _append(self, text: str) -> tuple[int, int]:
        start = self.pos
        self.parts.append(text + "\n\n")
        self.pos += len(text) + 2
        return start, start + len(text)

    def _close(self):
        if self._open:
            self.passages.append(self._open)
            self._open = None

    @property
    def section(self) -> str | None:
        return " > ".join(text for _, text in self.headings) or None

    def heading(self, level: int, text: str):
        text = normalize(text)
        if not text:
            return
        self._close()
        self.headings = [h for h in self.headings if h[0] < level] + [(level, text)]
        self._append(text)

    def block(self, text: str, page: int | None = None, low_ocr_quality: bool = False, own_passage: bool = False):
        text = text.strip()
        if not text:
            return
        if own_passage:
            self._close()
        start, end = self._append(text)
        if self._open and not own_passage:
            self._open.end = end
        else:
            self._open = Passage(start, end, page, self.section, low_ocr_quality)
        if own_passage:
            self._close()

    def result(self) -> tuple[str, list[Passage]]:
        self._close()
        return "".join(self.parts), self.passages


def _title_from(name: str) -> str:
    return PurePath(name).stem.replace("_", " ").replace("-", " ").strip() or name


def _ocr_page(data: bytes, page_number: int) -> tuple[str, float]:
    # Temporary files rather than pipes: older Poppler builds cannot write images to stdout.
    with tempfile.TemporaryDirectory(prefix="rag-ocr-") as tmp:
        source, image = Path(tmp) / "in.pdf", Path(tmp) / "page"
        source.write_bytes(data)
        try:
            subprocess.run(["pdftoppm", "-f", str(page_number), "-l", str(page_number), "-singlefile", "-r", "300", "-gray", str(source), str(image)], capture_output=True, check=True, timeout=60)
            result = subprocess.run(["tesseract", str(image) + ".pgm", "stdout", "tsv"], capture_output=True, check=True, timeout=90)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("Scanned PDF OCR requires Poppler pdftoppm and Tesseract") from exc
    return parse_tesseract_tsv(result.stdout.decode("utf-8", errors="replace"))


def parse_tesseract_tsv(tsv: str) -> tuple[str, float]:
    """Rebuild paragraph text from Tesseract TSV and return (text, mean word confidence)."""
    paragraphs: dict[tuple, dict[tuple, list[str]]] = {}
    confidences = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        word = (row.get("text") or "").strip()
        if row.get("level") != "5" or not word:
            continue
        try:
            confidence = float(row["conf"])
        except (KeyError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence)
        paragraph = (row["block_num"], row["par_num"])
        paragraphs.setdefault(paragraph, {}).setdefault(row["line_num"], []).append(word)
    text = "\n\n".join("\n".join(" ".join(words) for words in lines.values()) for lines in paragraphs.values())
    return text, (sum(confidences) / len(confidences) if confidences else 0.0)


def _alpha_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _parse_pdf(name: str, data: bytes, ocr_min_chars: int, ocr_min_confidence: float) -> ParsedDocument:
    reader = PdfReader(io.BytesIO(data))
    builder = _Builder()
    warnings = []
    for index, page in enumerate(reader.pages, 1):
        content = (page.extract_text() or "").strip()
        low_quality = False
        if _alpha_count(content) < ocr_min_chars:
            content, confidence = _ocr_page(data, index)
            content = content.strip()
            reasons = []
            if _alpha_count(content) < ocr_min_chars:
                reasons.append("little text recognized")
            if confidence < ocr_min_confidence:
                reasons.append(f"mean OCR confidence {confidence:.0f}")
            if reasons:
                low_quality = True
                warnings.append({"page": index, "reason": "; ".join(reasons), "ocr": True})
        if content:
            builder.block(content, page=index, low_ocr_quality=low_quality, own_passage=True)
        elif not low_quality:
            warnings.append({"page": index, "reason": "no text extracted", "ocr": True})
    text, passages = builder.result()
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title") or "").strip() or _title_from(name)
    return ParsedDocument(title, text, passages, None, warnings)


def _parse_markdown(name: str, content: str) -> ParsedDocument:
    builder = _Builder()
    title = None
    in_fence = False
    paragraph: list[str] = []

    def flush():
        if paragraph:
            builder.block("\n".join(paragraph))
            paragraph.clear()

    for line in content.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if heading:
            flush()
            level, text = len(heading.group(1)), heading.group(2)
            if level == 1 and title is None:
                title = normalize(text)
            builder.heading(level, text)
        elif not line.strip() and not in_fence:
            flush()
        else:
            paragraph.append(line)
    flush()
    text, passages = builder.result()
    return ParsedDocument(title or _title_from(name), text, passages)


def _parse_text(name: str, content: str) -> ParsedDocument:
    builder = _Builder()
    blocks = [b for b in re.split(r"\n\s*\n", content) if b.strip()]
    title = None
    if blocks and len(blocks[0].strip()) <= 100 and "\n" not in blocks[0].strip():
        title = blocks[0].strip()
        builder.heading(1, title)
        blocks = blocks[1:]
    for block in blocks:
        builder.block(block)
    text, passages = builder.result()
    return ParsedDocument(title or _title_from(name), text, passages)


BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "dt", "dd", "td", "th", "caption", "figcaption", "tr"]


def _parse_html(name: str, content: str) -> ParsedDocument:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "template"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "version"})
    version = meta.get("content") if meta else None
    root = soup.body or soup
    builder = _Builder()
    for element in root.find_all(BLOCK_TAGS):
        if element.name in {"td", "th"}:
            continue  # rendered via their row
        if element.name != "tr" and element.find(BLOCK_TAGS):
            continue  # container; its leaf blocks are emitted separately
        if element.name == "tr":
            text = " | ".join(cell.get_text(" ", strip=True) for cell in element.find_all(["td", "th"]))
        else:
            text = element.get_text(" ", strip=True)
        if element.name[0] == "h" and element.name[1:].isdigit():
            if not title and element.name == "h1":
                title = normalize(text)
            builder.heading(int(element.name[1:]), text)
        else:
            builder.block(text)
    text, passages = builder.result()
    return ParsedDocument(title or _title_from(name), text, passages, version)


def _parse_docx(name: str, data: bytes) -> ParsedDocument:
    doc = Document(io.BytesIO(data))
    builder = _Builder()
    title = (doc.core_properties.title or "").strip()
    version = (doc.core_properties.version or "").strip() or None
    for element in doc.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, doc)
            style = paragraph.style.name if paragraph.style is not None else ""
            text = paragraph.text
            if style == "Title":
                title = title or normalize(text)
                builder.heading(1, text)
            elif style.startswith("Heading") and style[7:].strip().isdigit():
                builder.heading(int(style[7:].strip()), text)
            else:
                builder.block(text)
        elif tag == "tbl":
            rows = []
            for row in Table(element, doc).rows:
                cells = list(dict.fromkeys(cell.text.strip() for cell in row.cells))
                rows.append(" | ".join(cells))
            builder.block("\n".join(rows))
    text, passages = builder.result()
    return ParsedDocument(title or _title_from(name), text, passages, version)


def parse_document(name: str, data: bytes, ocr_min_chars: int = 40, ocr_min_confidence: float = 60.0) -> ParsedDocument:
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix not in SUPPORTED:
        raise ValueError("Supported formats: PDF, DOCX, TXT, MD, HTML")
    if suffix in {"pdf", "docx"}:
        try:
            return _parse_pdf(name, data, ocr_min_chars, ocr_min_confidence) if suffix == "pdf" else _parse_docx(name, data)
        except ValueError:
            raise
        except Exception as exc:  # malformed or encrypted files surface as library-specific errors
            raise ValueError(f"Could not read the {suffix.upper()} file: {type(exc).__name__}") from exc
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text documents must be UTF-8") from exc
    if suffix in {"html", "htm"}:
        return _parse_html(name, content)
    if suffix in {"md", "markdown"}:
        return _parse_markdown(name, content)
    return _parse_text(name, content)


_SEPARATOR = re.compile(r"\n[ \t]*\n\s*|(?<=[.!?:])[ \t]*\n")


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, cursor = [], start
    for match in _SEPARATOR.finditer(text, start, end):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, end))
    result = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            result.append((a, b))
    return result


def _split_long(text: str, a: int, b: int, limit: int) -> list[tuple[int, int]]:
    pieces = []
    while b - a > limit:
        cut = text.rfind(" ", a + limit // 2, a + limit)
        cut = cut if cut > a else a + limit
        pieces.append((a, cut))
        a = cut
        while a < b and text[a].isspace():
            a += 1
    if a < b:
        pieces.append((a, b))
    return pieces


def _overlap_start(text: str, start: int, floor: int) -> int:
    """Move a candidate overlap start forward to the next word boundary."""
    if start <= floor:
        return floor
    space = text.find(" ", start)
    return space + 1 if 0 <= space < len(text) else start


def chunk_document(doc: ParsedDocument, max_chars: int = 1300, overlap: int = 180) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ``max_chars`` raw characters.

    Chunks never cross a passage (page or section) boundary. Consecutive chunks in a
    passage share up to ``overlap`` characters.
    """
    if overlap >= max_chars // 2:
        raise ValueError("overlap must be less than half of max_chars")
    text = doc.text
    chunks = []
    for passage in doc.passages:
        units = []
        for a, b in _paragraph_spans(text, passage.start, passage.end):
            units.extend(_split_long(text, a, b, max_chars - overlap))
        cur_start = cur_end = None
        for a, b in units:
            if cur_start is None:
                cur_start, cur_end = a, b
            elif b - cur_start <= max_chars:
                cur_end = b
            else:
                chunks.append((cur_start, cur_end, passage))
                start = _overlap_start(text, cur_end - overlap, cur_start + 1)
                cur_start = start if b - start <= max_chars and start < a else a
                cur_end = b
        if cur_start is not None:
            chunks.append((cur_start, cur_end, passage))
    return [Chunk(normalize(text[a:b]), a, b, p.page, p.section, p.low_ocr_quality) for a, b, p in chunks if normalize(text[a:b])]
