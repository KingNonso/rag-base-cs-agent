"""Regenerate the binary fixture documents (DOCX, searchable PDF, scanned PDF).

Requires the test extras: pip install -e '.[test]'
Usage: python scripts/make_fixtures.py [output_dir]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont


def make_docx(path: Path):
    doc = Document()
    doc.core_properties.title = "Acme Desk FAQ"
    doc.core_properties.version = "2026.1"
    doc.add_heading("Acme Desk FAQ", 0)
    doc.add_heading("Account security", 1)
    doc.add_paragraph("Two-factor authentication can be enabled from Settings > Security. Account administrators can require two-factor authentication for every agent.")
    doc.add_heading("Integrations", 1)
    doc.add_paragraph("The table below lists which plans include each integration.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Integration", "Plans"
    for name, plans in [("Slack", "Basic and Pro"), ("Salesforce", "Pro only")]:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = name, plans
    doc.add_paragraph("Integrations are configured from Settings > Integrations.")
    doc.save(path)


def make_pdf(path: Path):
    pdf = FPDF()
    pdf.set_title("Acme Desk security overview")
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    pages = [
        ("Data protection", ["All customer data is encrypted at rest with AES-256 and in transit with TLS 1.2 or later.", "Access to production systems requires hardware security keys."]),
        ("Backups", ["Backups are taken every 24 hours and kept for 14 days.", "Restores are performed by the Acme Desk operations team on request."]),
    ]
    for heading, paragraphs in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=12)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 7, paragraph)
            pdf.ln(4)
    pdf.output(str(path))


def make_scanned_pdf(path: Path):
    lines = [
        ("Onboarding checklist", 56),
        ("", 30),
        ("New accounts include a 21-day free trial", 40),
        ("of the Pro plan.", 40),
        ("", 30),
        ("During the trial, up to five agents can", 40),
        ("be invited from Settings > Team.", 40),
    ]
    image = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(image)
    y = 180
    for text, size in lines:
        draw.text((150, y), text, fill=0, font=ImageFont.load_default(size=size))
        y += int(size * 1.6)
    # Embed as lossless PNG (FlateDecode); some older Poppler builds render JPEG-only scans blank.
    png = path.with_suffix(".png")
    image.save(png)
    pdf = FPDF(unit="pt", format=(612, 792))
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    pdf.add_page()
    pdf.image(str(png), x=0, y=0, w=612, h=792)
    pdf.output(str(path))
    png.unlink()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "fixtures" / "documents"
    out.mkdir(parents=True, exist_ok=True)
    make_docx(out / "faq.docx")
    make_pdf(out / "security.pdf")
    make_scanned_pdf(out / "onboarding-scan.pdf")
    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
