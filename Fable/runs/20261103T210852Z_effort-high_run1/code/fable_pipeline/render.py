"""Markdown rendering of a :class:`Document` for a generic chunker."""
from __future__ import annotations

from typing import List

from .models import Block, Document


def _heading_line(block: Block, offset: int) -> str:
    level = block.level if block.level is not None else 1
    hashes = "#" * max(1, min(6, level + offset))
    return f"{hashes} {' '.join(block.text.split())}"


def _render_table(block: Block) -> str:
    parts: List[str] = []
    td = block.table
    if td is not None and td.markdown:
        if td.caption:
            parts.append(f"**{td.caption}**")
        parts.append(td.markdown)
    else:
        parts.append(block.text)
    return "\n\n".join(p for p in parts if p)


def _render_figure(block: Block) -> str:
    fd = block.figure
    if fd is None:
        return block.text
    lines: List[str] = []
    lines.append(f"[Figure: {fd.caption}]" if fd.caption else "[Figure]")
    if fd.description:
        lines.append(f"Description: {fd.description.strip()}")
    text = (fd.ocr_text or fd.native_text or "").strip()
    if text:
        lines.append("Text in figure: " + " ".join(text.split()))
    return "\n".join(lines)


def to_markdown(document: Document) -> str:
    """Single clean Markdown rendering: headings as ``#``, tables as Markdown tables,
    figures as a description block, lists with one item per line."""
    has_title = any(b.type == "heading" and b.level == 0 for b in document.blocks)
    offset = 1 if has_title else 0
    out: List[str] = []
    for block in document.blocks:
        if block.type == "heading":
            out.append(_heading_line(block, 0 if block.level == 0 else offset))
        elif block.type == "paragraph":
            text = block.text.strip()
            if not text:
                continue
            out.append(text)
        elif block.type == "list":
            items = [ln for ln in block.text.split("\n") if ln.strip()]
            out.append("\n".join(items))
        elif block.type == "table":
            out.append(_render_table(block))
        elif block.type == "figure":
            out.append(_render_figure(block))
        elif block.type == "caption":
            out.append(f"*{block.text.strip()}*")
        else:
            out.append(block.text.strip())
    md = "\n\n".join(p for p in out if p)
    return md.strip() + "\n" if md.strip() else ""
