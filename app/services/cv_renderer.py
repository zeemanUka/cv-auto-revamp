from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import textwrap
import os


def generate_pdf_from_text(text: str, output_path: str) -> None:
    """
    Generate a simple PDF from text and save to output_path.
    This is MVP formatting; later you can improve fonts, spacing, etc.
    """
    output = Path(output_path)
    os.makedirs(output.parent, exist_ok=True)

    c = canvas.Canvas(str(output), pagesize=LETTER)
    width, height = LETTER

    margin = 50
    y = height - margin
    line_height = 14
    max_chars_per_line = 90  # adjust later if needed

    paragraphs = text.split("\n")
    for para in paragraphs:
        if not para.strip():
            y -= line_height  # blank line
            continue

        lines = textwrap.wrap(para, max_chars_per_line)
        for line in lines:
            if y <= margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height

        y -= line_height  # extra space after paragraph

    c.save()
