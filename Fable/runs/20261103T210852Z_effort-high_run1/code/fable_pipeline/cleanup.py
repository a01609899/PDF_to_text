"""Module 6: text cleanup.

Operates on the assembled blocks (and the metadata title).  Paragraph lines
are joined with ``"\\n"`` by the section module; this module is responsible
for turning them into clean running text: de-hyphenation across line breaks,
ligature and Unicode normalisation (via ``ftfy`` + explicit maps), removal of
control/zero-width characters, collapsing whitespace, fixing letter-spaced
words (``I N T R O D U C T I O N``) and dot leaders from tables of contents.

List blocks keep one item per line; table cells and figure texts are cleaned
individually and the derived renderings (Markdown/CSV/records) are rebuilt.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from .config import CleanupConfig, PipelineConfig
from .models import Block, Document, TableData

log = logging.getLogger(__name__)

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
    "\u0132": "IJ", "\u0133": "ij",
}
_LIG_TABLE = str.maketrans(LIGATURES)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff]")
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")
_HYPHEN_BREAK_RE = re.compile(r"(\w+)([\-\u2010\u2011])[ \t]*\n[ \t]*(\w+)")
_DOT_LEADER_RE = re.compile(r"(?:\s?[.·•]){4,}\s*")
_SPACED_RE = re.compile(r"(?<![A-Za-zÄÖÜäöüß])(?:[A-ZÄÖÜ] ){3,}[A-ZÄÖÜ](?![A-Za-zÄÖÜäöüß])")
_WS_RE = re.compile(r"[ \t\u00a0\u2009\u202f]+")
_PUNCT_SPACE_RE = re.compile(r" +([,.;:!?%)\]])(?=\s|$)")
_OPEN_SPACE_RE = re.compile(r"([(\[]) +")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_QUOTES = {"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u00ab": '"',
           "\u00bb": '"', "\u2039": "'", "\u203a": "'"}
_QUOTE_TABLE = str.maketrans(_QUOTES)

KEEP_HYPHEN_PREFIXES = {
    "self", "non", "anti", "co", "multi", "semi", "cross", "well", "long", "short", "high", "low", "inter",
    "intra", "sub", "quasi", "pseudo", "meta", "micro", "macro", "nano", "ultra", "mid", "half", "all",
    "ex", "pre", "post", "pro",
}


def _dehyphenate(text: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        a, b = m.group(1), m.group(3)
        if b[0].islower() and a.lower() not in KEEP_HYPHEN_PREFIXES and not a.isdigit() and not b[0].isdigit():
            return a + b
        return a + "-" + b

    return _HYPHEN_BREAK_RE.sub(repl, text)


def _fix_spaced_letters(text: str) -> str:
    return _SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def clean_text(text: str, cfg: Optional[CleanupConfig] = None, keep_newlines: bool = False) -> str:
    """Clean a text fragment.  ``keep_newlines=True`` keeps one item per line (lists)."""
    cfg = cfg or CleanupConfig()
    if not text:
        return ""
    if cfg.use_ftfy:
        try:
            import ftfy  # noqa: WPS433

            text = ftfy.fix_text(text, normalization=None)
        except ImportError:  # pragma: no cover
            pass
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.remove_control_chars:
        text = _CONTROL_RE.sub("", text)
        text = text.replace("\u00a0", " ")
    text = _SOFT_HYPHEN_RE.sub("", text)
    if cfg.fix_ligatures:
        text = text.translate(_LIG_TABLE)
    if cfg.normalize_unicode:
        text = unicodedata.normalize("NFC", text)
    if cfg.dehyphenate:
        text = _dehyphenate(text)
    if cfg.remove_dot_leaders:
        text = _DOT_LEADER_RE.sub(" ", text)
    if cfg.fix_spaced_letters:
        text = _fix_spaced_letters(text)
    if cfg.normalize_quotes:
        text = text.translate(_QUOTE_TABLE)
    if cfg.collapse_whitespace:
        if keep_newlines:
            lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
            text = "\n".join(ln for ln in lines if ln)
        else:
            text = _WS_RE.sub(" ", text.replace("\n", " "))
        text = _PUNCT_SPACE_RE.sub(r"\1", text)
        text = _OPEN_SPACE_RE.sub(r"\1", text)
        text = _MULTI_NL_RE.sub("\n", text)
    return text.strip()


_MARKER_RE = re.compile(r"^(?:[-*•]|\(?(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,5})[.)\]])\s+")


def _clean_list(text: str, cfg: CleanupConfig) -> str:
    """Clean a list block: join continuation lines into their item, keep items on separate lines."""
    text = clean_text(text, cfg, keep_newlines=True)
    items = []
    for ln in text.split("\n"):
        if items and not _MARKER_RE.match(ln):
            items[-1] = (items[-1] + " " + ln).strip()
        else:
            items.append(ln)
    return "\n".join(items)


def _clean_table(td: TableData, cfg: CleanupConfig) -> TableData:
    grid = [[clean_text(c, cfg) for c in td.columns]] + [[clean_text(c, cfg) for c in row] for row in td.rows]
    caption = clean_text(td.caption, cfg) if td.caption else None
    new = TableData.from_grid(grid, caption=caption, method=td.method, confidence=td.confidence, header_row=True)
    if not new.columns:  # table collapsed to nothing -> keep the original
        return td
    return new


def clean_block(block: Block, cfg: CleanupConfig) -> Block:
    from .figures import compose_figure_text  # local import avoids a circular import at module load
    from .tables import compose_table_text

    if block.type == "list":
        block.text = _clean_list(block.text, cfg)
        if "items" in block.extra:
            block.extra["items"] = [i for i in block.text.split("\n") if i]
    elif block.type == "table" and block.table is not None:
        block.table = _clean_table(block.table, cfg)
        block.text = compose_table_text(block.table)
        block.extra["caption"] = block.table.caption
    elif block.type == "figure" and block.figure is not None:
        fd = block.figure
        fd.caption = clean_text(fd.caption, cfg) if fd.caption else None
        fd.ocr_text = clean_text(fd.ocr_text, cfg, keep_newlines=True)
        fd.native_text = clean_text(fd.native_text, cfg)
        fd.description = clean_text(fd.description, cfg) if fd.description else None
        block.text = compose_figure_text(fd)
        block.extra["caption"] = fd.caption
    else:
        block.text = clean_text(block.text, cfg)
    block.section_path = [clean_text(p, cfg) for p in block.section_path]
    return block


def cleanup_document(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    config = config or PipelineConfig()
    cfg = config.cleanup
    if not cfg.enabled:
        return doc
    for block in doc.blocks:
        clean_block(block, cfg)
    doc.blocks = [b for b in doc.blocks if b.text.strip() or b.type in ("table", "figure")]
    if doc.metadata.title:
        doc.metadata.title = clean_text(doc.metadata.title, cfg)
    return doc
