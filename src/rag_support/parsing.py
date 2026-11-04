import io
import re
import subprocess
from dataclasses import dataclass

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


@dataclass
class Passage:
    text: str
    location: str


def parse_document(name: str, data: bytes) -> list[Passage]:
    suffix = name.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        pages = []
        reader = PdfReader(io.BytesIO(data))
        for index, page in enumerate(reader.pages):
            content = (page.extract_text(extraction_mode="layout") or "").strip()
            if len(content) < 40:
                try:
                    rendered = subprocess.run(["pdftoppm", "-f", str(index + 1), "-l", str(index + 1), "-singlefile", "-r", "200", "-png", "-", "-"], input=data, capture_output=True, check=True, timeout=45)
                    ocr = subprocess.run(["tesseract", "stdin", "stdout"], input=rendered.stdout, capture_output=True, check=True, timeout=45)
                except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    raise ValueError("Scanned PDF OCR requires Poppler pdftoppm and Tesseract") from exc
                content = ocr.stdout.decode("utf-8", errors="replace").strip()
            if content:
                pages.append(Passage(content, f"page {index + 1}"))
        return pages
    if suffix == "docx":
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        return [Passage("\n".join(parts), "document")]
    if suffix in {"txt", "md", "markdown", "html", "htm"}:
        content = data.decode("utf-8-sig")
        if suffix in {"html", "htm"}:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            content = soup.get_text("\n")
        return [Passage(content, "document")]
    raise ValueError("Supported formats: PDF, DOCX, TXT, MD, HTML")


def chunk_passages(passages: list[Passage], max_chars: int = 1300, overlap: int = 180) -> list[Passage]:
    chunks = []
    for passage in passages:
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|(?<=\.)\s*\n", passage.text) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            while len(paragraph) > max_chars:
                if current:
                    chunks.append(Passage(current, passage.location))
                    current = ""
                cut = paragraph.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                chunks.append(Passage(paragraph[:cut], passage.location))
                paragraph = paragraph[max(0, cut - overlap):].strip()
            if len(current) + len(paragraph) + 1 > max_chars and current:
                chunks.append(Passage(current, passage.location))
                current = current[-overlap:].strip()
            current = f"{current}\n{paragraph}".strip()
        if current:
            chunks.append(Passage(current, passage.location))
    return chunks
