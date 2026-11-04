"""Module 1: ingestion & layout parsing.

Backed by **PyMuPDF** (``fitz``): it is the fastest open-source PDF parser,
exposes glyph-level geometry, font name/size/flags and colour, and renders
pages (needed later for OCR and figure crops).  pdfplumber is only used in the
table module where its ruling-line heuristics are complementary.

Responsibilities
----------------
1. Extract spans/lines with bounding boxes and font information.
2. Re-merge line fragments that PyMuPDF splits on the same baseline.
3. Determine a reading order that is robust to multi-column layouts using a
   recursive XY-cut with band merging (see DESIGN.md).

The output is a :class:`Document` whose ``pages`` carry :class:`Line` objects
with ``order`` (reading order index), ``region`` (column/band id) and
``region_bbox`` set.  ``blocks`` is still empty at this stage.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

try:  # PyMuPDF >= 1.24 exposes both names
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from .config import IngestionConfig, PipelineConfig
from .layout_utils import median, union_bbox
from .models import PIPELINE_VERSION, BBox, Document, DocumentMetadata, ImageInfo, Line, Page, Span

log = logging.getLogger(__name__)

BOLD_RE = re.compile(r"bold|black|heavy|semibold|demibold|extrabold|ultrabold", re.I)
ITALIC_RE = re.compile(r"italic|oblique", re.I)
TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_layout(pdf_path: str, config: Optional[PipelineConfig] = None) -> Document:
    """Load a PDF and return a Document with the layout layer populated."""
    config = config or PipelineConfig()
    cfg = config.ingestion
    pdf = fitz.open(pdf_path)
    try:
        metadata = _build_metadata(pdf, pdf_path, config)
        pages: List[Page] = []
        for pno in range(pdf.page_count):
            fitz_page = pdf.load_page(pno)
            pages.append(parse_page(fitz_page, pno + 1, cfg))
    finally:
        pdf.close()
    log.info("ingested %s: %d pages", os.path.basename(pdf_path), len(pages))
    return Document(metadata=metadata, blocks=[], pages=pages)


def parse_page(fitz_page, number: int, cfg: IngestionConfig) -> Page:
    rect = fitz_page.rect
    page = Page(number=number, width=float(rect.width), height=float(rect.height), rotation=int(fitz_page.rotation))
    raw = fitz_page.get_text("dict", flags=TEXT_FLAGS)
    lines: List[Line] = []
    for bi, blk in enumerate(raw.get("blocks", [])):
        if blk.get("type") != 0:
            continue
        for raw_line in blk.get("lines", []):
            direction = raw_line.get("dir", (1.0, 0.0))
            vertical = abs(float(direction[0])) < 0.5
            spans = _make_spans(raw_line.get("spans", []))
            line = _line_from_spans(spans, number, bi, vertical)
            if line is not None:
                lines.append(line)
    lines = _merge_fragments(lines, cfg)
    page.lines = lines
    page.images = _image_infos(fitz_page, page)
    page.char_count = sum(len(ln.text) for ln in lines)
    assign_reading_order(page, cfg)
    return page


def dominant_font_size(pages: Sequence[Page], roles: Sequence[str] = ("body",)) -> float:
    """Character-weighted mode of font sizes (rounded to 0.5pt) over the given roles."""
    counter: Counter = Counter()
    for page in pages:
        for ln in page.lines:
            if ln.role in roles and ln.font_size > 0:
                counter[round(ln.font_size * 2) / 2] += len(ln.text)
    if not counter:
        return 10.0
    return float(counter.most_common(1)[0][0])


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _clean_meta(value) -> Optional[str]:
    value = (value or "").strip() if isinstance(value, str) else None
    return value or None


def _build_metadata(pdf, pdf_path: str, config: PipelineConfig) -> DocumentMetadata:
    md = pdf.metadata or {}
    outline = []
    try:
        for entry in pdf.get_toc(simple=True):
            level, title, pno = entry[0], str(entry[1]).strip(), int(entry[2])
            if title:
                outline.append({"level": int(level), "title": title, "page": pno})
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not read outline: %s", exc)
    return DocumentMetadata(
        source_path=str(pdf_path),
        file_name=os.path.basename(pdf_path),
        page_count=int(pdf.page_count),
        title=None,
        pdf_title=_clean_meta(md.get("title")),
        author=_clean_meta(md.get("author")),
        subject=_clean_meta(md.get("subject")),
        creation_date=_clean_meta(md.get("creationDate")),
        producer=_clean_meta(md.get("producer")),
        creator=_clean_meta(md.get("creator")),
        outline=outline,
        pipeline_version=PIPELINE_VERSION,
        config=config.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Spans and lines
# --------------------------------------------------------------------------- #
def _make_spans(raw_spans) -> List[Span]:
    spans: List[Span] = []
    for sp in raw_spans:
        text = sp.get("text", "")
        if not text:
            continue
        font = sp.get("font", "") or ""
        flags = int(sp.get("flags", 0))
        spans.append(
            Span(
                text=text,
                bbox=BBox.from_any(sp["bbox"]),
                font=font,
                size=float(sp.get("size", 0.0)),
                flags=flags,
                color=int(sp.get("color", 0)),
                bold=bool(flags & 16) or bool(BOLD_RE.search(font)),
                italic=bool(flags & 2) or bool(ITALIC_RE.search(font)),
                superscript=bool(flags & 1),
            )
        )
    return spans


def _apply_dominant_style(line: Line) -> None:
    size_w: Counter = Counter()
    font_w: Counter = Counter()
    bold_chars = italic_chars = total = 0
    for sp in line.spans:
        n = len(sp.text.strip())
        if n == 0:
            continue
        total += n
        size_w[round(sp.size, 1)] += n
        font_w[sp.font] += n
        if sp.bold:
            bold_chars += n
        if sp.italic:
            italic_chars += n
    if total == 0:
        return
    line.font_size = float(size_w.most_common(1)[0][0])
    line.font_name = font_w.most_common(1)[0][0]
    line.bold = bold_chars / total >= 0.8
    line.italic = italic_chars / total >= 0.8


def _line_from_spans(spans: List[Span], page_no: int, block_idx: int, vertical: bool) -> Optional[Line]:
    text = "".join(sp.text for sp in spans).strip()
    if not text:
        return None
    bbox = union_bbox(sp.bbox for sp in spans if sp.text.strip()) or union_bbox(sp.bbox for sp in spans)
    if bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return None
    line = Line(text=text, bbox=bbox, page=page_no, spans=spans, layout_block=block_idx, vertical=vertical)
    _apply_dominant_style(line)
    if vertical:
        line.role = "rotated"
    return line


def _merge_fragments(lines: List[Line], cfg: IngestionConfig) -> List[Line]:
    """Merge line fragments on the same baseline that PyMuPDF emitted separately."""
    lines = sorted(lines, key=lambda l: (round(l.bbox.y0, 1), l.bbox.x0))
    merged: List[Line] = []
    for ln in lines:
        target: Optional[Line] = None
        best_gap = 0.0
        for cand in merged[-12:]:
            if cand.vertical != ln.vertical:
                continue
            ov = cand.bbox.y_overlap(ln.bbox)
            if ov < 0.5 * min(cand.bbox.height, ln.bbox.height):
                continue
            gap = ln.bbox.x0 - cand.bbox.x1
            size = max(cand.font_size, ln.font_size, 1.0)
            factor = cfg.merge_line_gap_factor if cand.layout_block == ln.layout_block else cfg.merge_cross_block_gap_factor
            if -0.5 * size <= gap <= factor * size and (target is None or abs(gap) < abs(best_gap)):
                target, best_gap = cand, gap
        if target is None:
            merged.append(ln)
        else:
            size = max(target.font_size, ln.font_size, 1.0)
            sep = " " if best_gap > 0.1 * size else ""
            target.text = (target.text + sep + ln.text).strip()
            target.spans = target.spans + ln.spans
            target.bbox = target.bbox.union(ln.bbox)
            _apply_dominant_style(target)
    return merged


def _image_infos(fitz_page, page: Page) -> List[ImageInfo]:
    out: List[ImageInfo] = []
    try:
        infos = fitz_page.get_image_info(xrefs=True)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("get_image_info failed on page %d: %s", page.number, exc)
        return out
    seen = set()
    for inf in infos:
        try:
            bbox = BBox.from_any(inf["bbox"])
        except Exception:
            continue
        bbox = BBox(max(0.0, bbox.x0), max(0.0, bbox.y0), min(page.width, bbox.x1), min(page.height, bbox.y1))
        if bbox.width <= 1 or bbox.height <= 1:
            continue
        key = tuple(bbox.to_list())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ImageInfo(
                bbox=bbox,
                xref=int(inf.get("xref", 0) or 0),
                width=int(inf.get("width", 0) or 0),
                height=int(inf.get("height", 0) or 0),
            )
        )
    out.sort(key=lambda im: (im.bbox.y0, im.bbox.x0))
    return out


# --------------------------------------------------------------------------- #
# Reading order: recursive XY-cut with band merging
# --------------------------------------------------------------------------- #
def assign_reading_order(page: Page, cfg: IngestionConfig) -> None:
    """Assign ``order``/``region``/``region_bbox`` to all lines of ``page``.

    ``page.lines`` is sorted into reading order afterwards and ``index`` reset.
    """
    flow = [ln for ln in page.lines if not ln.vertical]
    rotated = [ln for ln in page.lines if ln.vertical]
    regions: List[List[Line]] = []
    if flow:
        heights = [ln.bbox.height for ln in flow]
        med_h = median(heights, 10.0) or 10.0
        med_gap = _median_line_gap(flow)
        band_gap = max(cfg.min_band_gap_factor * med_h, 1.6 * med_gap + 1.0, 2.0)
        if cfg.detect_columns:
            _xy_cut(flow, regions, cfg, band_gap, page.width, 0)
        else:
            regions.append(flow)
    order = 0
    for rid, region in enumerate(regions):
        rb = union_bbox(ln.bbox for ln in region)
        for ln in _sort_rows(region):
            ln.order = order
            ln.region = rid
            ln.region_bbox = rb
            order += 1
    for ln in sorted(rotated, key=lambda l: (l.bbox.y0, l.bbox.x0)):
        ln.order = order
        ln.region = len(regions)
        ln.region_bbox = ln.bbox
        order += 1
    page.lines = sorted(page.lines, key=lambda l: l.order)
    for i, ln in enumerate(page.lines):
        ln.index = i


def _median_line_gap(lines: List[Line]) -> float:
    srt = sorted(lines, key=lambda l: (l.bbox.y0, l.bbox.x0))
    gaps = []
    for a, b in zip(srt, srt[1:]):
        if a.bbox.x_overlap(b.bbox) > 0:
            gaps.append(max(0.0, b.bbox.y0 - a.bbox.y1))
    return median(gaps, 0.0)


def _split_bands(lines: List[Line], gap: float) -> List[List[Line]]:
    srt = sorted(lines, key=lambda l: (l.bbox.y0, l.bbox.x0))
    bands: List[List[Line]] = [[srt[0]]]
    max_y1 = srt[0].bbox.y1
    for ln in srt[1:]:
        if ln.bbox.y0 - max_y1 > gap:
            bands.append([ln])
            max_y1 = ln.bbox.y1
        else:
            bands[-1].append(ln)
            max_y1 = max(max_y1, ln.bbox.y1)
    return bands


def _find_gutters(lines: List[Line], cfg: IngestionConfig, page_w: float, min_lines: int) -> List[Tuple[float, float]]:
    """Vertical white-space corridors that no line of ``lines`` crosses."""
    if len(lines) < 2 * min_lines:
        return []
    intervals = sorted((ln.bbox.x0, ln.bbox.x1) for ln in lines)
    merged: List[List[float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if len(merged) < 2:
        return []
    rx0, rx1 = merged[0][0], merged[-1][1]
    min_col_w = cfg.min_column_width_ratio * page_w
    gutters: List[Tuple[float, float]] = []
    for i in range(len(merged) - 1):
        g0, g1 = merged[i][1], merged[i + 1][0]
        if g1 - g0 < cfg.min_gutter_width:
            continue
        if g0 - rx0 < min_col_w or rx1 - g1 < min_col_w:
            continue
        n_left = sum(1 for ln in lines if ln.bbox.x1 <= g0 + 0.5)
        n_right = sum(1 for ln in lines if ln.bbox.x0 >= g1 - 0.5)
        if n_left < min_lines or n_right < min_lines:
            continue
        gutters.append((g0, g1))
    return gutters


def _gutters_compatible(g1: List[Tuple[float, float]], g2: List[Tuple[float, float]]) -> bool:
    for a0, a1 in g1:
        for b0, b1 in g2:
            if min(a1, b1) - max(a0, b0) > 0:
                return True
    return False


def _split_columns(lines: List[Line], gutters: List[Tuple[float, float]]) -> List[List[Line]]:
    mids = sorted((g0 + g1) / 2.0 for g0, g1 in gutters)
    cols: List[List[Line]] = [[] for _ in range(len(mids) + 1)]
    for ln in lines:
        idx = sum(1 for m in mids if ln.bbox.cx > m)
        cols[idx].append(ln)
    return [c for c in cols if c]


def _xy_cut(lines: List[Line], regions: List[List[Line]], cfg: IngestionConfig, band_gap: float, page_w: float, depth: int) -> None:
    if len(lines) <= 1 or depth > 10:
        regions.append(lines)
        return
    bands = _split_bands(lines, band_gap)
    # group consecutive bands whose (loose) gutters overlap: they share a column structure
    groups: List[Tuple[List[Line], List[Tuple[float, float]]]] = []
    for band in bands:
        loose = _find_gutters(band, cfg, page_w, min_lines=1)
        if groups and loose and groups[-1][1] and _gutters_compatible(groups[-1][1], loose):
            merged_lines = groups[-1][0] + band
            merged_gutters = _find_gutters(merged_lines, cfg, page_w, min_lines=1)
            if merged_gutters:
                groups[-1] = (merged_lines, merged_gutters)
                continue
        groups.append((band, loose))
    if len(groups) == 1:
        strict = _find_gutters(lines, cfg, page_w, cfg.min_lines_per_column)
        if not strict:
            regions.append(lines)
            return
        for col in _split_columns(lines, strict):
            _xy_cut(col, regions, cfg, band_gap, page_w, depth + 1)
        return
    for glines, _ in groups:
        strict = _find_gutters(glines, cfg, page_w, cfg.min_lines_per_column)
        if strict:
            for col in _split_columns(glines, strict):
                _xy_cut(col, regions, cfg, band_gap, page_w, depth + 1)
        else:
            regions.append(glines)


def _sort_rows(lines: List[Line]) -> List[Line]:
    """Top-to-bottom, left-to-right sort that treats vertically overlapping lines as one row."""
    srt = sorted(lines, key=lambda l: (l.bbox.y0, l.bbox.x0))
    rows: List[List[Line]] = []
    for ln in srt:
        if rows:
            row = rows[-1]
            top = min(l.bbox.y0 for l in row)
            bottom = max(l.bbox.y1 for l in row)
            overlap = min(bottom, ln.bbox.y1) - max(top, ln.bbox.y0)
            if overlap >= 0.5 * min(ln.bbox.height, bottom - top) and overlap > 0:
                row.append(ln)
                continue
        rows.append([ln])
    out: List[Line] = []
    for row in rows:
        out.extend(sorted(row, key=lambda l: l.bbox.x0))
    return out
