"""Shared helpers for building small synthetic PDFs with PyMuPDF."""
from __future__ import annotations

import io
from typing import Iterable, List, Optional, Sequence, Tuple

import pytest

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

PAGE_W, PAGE_H = 595.0, 842.0


def new_pdf():
    return fitz.open()


def add_page(pdf, width: float = PAGE_W, height: float = PAGE_H):
    return pdf.new_page(width=width, height=height)


def write_lines(page, lines: Iterable[str], x: float = 72, y: float = 72, size: float = 10, font: str = "helv",
                leading: Optional[float] = None) -> float:
    """Insert one text line per entry, return the baseline y after the last line."""
    for text in lines:
        page.insert_text((x, y), text, fontsize=size, fontname=font)
        y += leading if leading is not None else size * 1.2
    return y


def draw_grid_table(page, x0: float, y0: float, col_widths: Sequence[float], row_height: float,
                    cells: Sequence[Sequence[str]], size: float = 9) -> Tuple[float, float, float, float]:
    """Draw a ruled table and return its bbox (x0, y0, x1, y1)."""
    x1 = x0 + sum(col_widths)
    y1 = y0 + row_height * len(cells)
    shape = page.new_shape()
    y = y0
    for _ in range(len(cells) + 1):
        shape.draw_line((x0, y), (x1, y))
        y += row_height
    x = x0
    for w in list(col_widths) + [0]:
        shape.draw_line((x, y0), (x, y1))
        x += w
    shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()
    for r, row in enumerate(cells):
        x = x0
        for c, text in enumerate(row):
            page.insert_text((x + 3, y0 + r * row_height + row_height * 0.7), str(text), fontsize=size, fontname="helv")
            x += col_widths[c]
    return (x0, y0, x1, y1)


def make_png(width: int = 200, height: int = 150, color=(200, 30, 30), text: Optional[str] = None,
             text_size: int = 40) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color)
    if text:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=text_size)
        except TypeError:  # pragma: no cover - very old Pillow
            font = ImageFont.load_default()
        draw.text((20, 20), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def insert_png(page, rect: Tuple[float, float, float, float], png: bytes) -> None:
    page.insert_image(fitz.Rect(*rect), stream=png)


def save(pdf, path) -> str:
    pdf.save(str(path))
    pdf.close()
    return str(path)


def body_lines(n: int, prefix: str = "This is body text line") -> List[str]:
    return [f"{prefix} {i + 1} written to fill the full width of the synthetic column area." for i in range(n)]


@pytest.fixture
def simple_pdf(tmp_path) -> str:
    """One page with a title, a numbered heading and two paragraphs."""
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 90), "A Synthetic Test Document", fontsize=18, fontname="hebo")
    page.insert_text((72, 140), "1 Introduction", fontsize=14, fontname="hebo")
    y = write_lines(page, body_lines(3), y=170)
    write_lines(page, body_lines(3, "Second paragraph line"), y=y + 14)
    return save(pdf, tmp_path / "simple.pdf")


@pytest.fixture
def multipage_pdf(tmp_path) -> str:
    """Four pages with running header, page numbers and body text continuing across pages."""
    pdf = new_pdf()
    for i in range(4):
        page = add_page(pdf)
        page.insert_text((72, 30), "Journal of Synthetic Tests", fontsize=9, fontname="helv")
        page.insert_text((290, 820), str(i + 1), fontsize=9, fontname="helv")
        if i == 0:
            page.insert_text((72, 90), "Running Text Paper", fontsize=18, fontname="hebo")
            page.insert_text((72, 140), "1 Introduction", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(4), y=170)
            write_lines(page, body_lines(3, "Tail paragraph line"), y=760)
        elif i == 1:
            write_lines(page, body_lines(3, "continues on the next page line"), y=72)
            page.insert_text((72, 150), "2 Methods", fontsize=14, fontname="hebo")
            page.insert_text((72, 180), "2.1 Data", fontsize=12, fontname="hebo")
            write_lines(page, body_lines(3, "Data paragraph line"), y=200)
        else:
            write_lines(page, body_lines(5, f"Page {i + 1} paragraph line"), y=100)
    return save(pdf, tmp_path / "multipage.pdf")
