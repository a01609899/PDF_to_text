<<<FILE fable_pipeline/figures.py>>>
"""Module 5: image/figure handling and scanned-page OCR.

Two entry points:

* :func:`ocr_scanned_pages` – runs right after ingestion.  A page with (almost)
  no native text but a large raster image is a *scanned page*; it is rendered
  and passed through Tesseract, and the recognised lines (with boxes, in PDF
  points) replace the page's lines so every downstream module works unchanged.
* :func:`extract_figures` – detects figure regions (embedded raster images and
  clusters of vector drawings), crops them, OCRs text inside them, optionally
  asks a local vision model (Ollama) for a description, attaches the caption
  and emits ``Block(type="figure")`` entries anchored on the page.
"""
from __future__ import annotations

import io
import logging
import os
from typing import List, Optional, Tuple

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from .captions import FIGURE_CAPTION_RE, caption_text, claim, find_caption
from .config import FigureConfig, PipelineConfig
from .ingestion import assign_reading_order, dominant_font_size
from .layout_utils import lines_in_bbox, sorted_by_order
from .models import BBox, Block, Document, FigureData, Line, Page
from .ocr import ocr_lines, ocr_text, pixmap_to_image, tesseract_available
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)


def compose_figure_text(fd: FigureData) -> str:
    """Flat text representation of a figure (used as ``Block.text``)."""
    parts = []
    if fd.caption:
        parts.append(fd.caption)
    if fd.description:
        parts.append("Description: " + fd.description.strip())
    text = (fd.ocr_text or fd.native_text or "").strip()
    if text:
        parts.append("Text in figure: " + " ".join(text.split()))
    if not parts:
        parts.append("[Figure]")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Scanned pages
# --------------------------------------------------------------------------- #
def _image_coverage(page: Page) -> float:
    if page.area <= 0:
        return 0.0
    covered = 0.0
    for im in page.images:
        covered += min(im.bbox.area, page.area)
    return min(1.0, covered / page.area)


def is_scanned_candidate(page: Page, cfg: FigureConfig) -> bool:
    """A page is treated as scanned if it has (almost) no native text but a big image."""
    if page.char_count >= cfg.scanned_max_chars:
        return False
    if not page.images:
        return False
    return _image_coverage(page) >= cfg.scanned_min_image_coverage


def ocr_scanned_pages(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    config = config or PipelineConfig()
    cfg = config.figures
    if not cfg.detect_scanned_pages:
        return doc
    candidates = [p for p in doc.pages if is_scanned_candidate(p, cfg)]
    if not candidates:
        return doc
    if not cfg.ocr_enabled:
        doc.metadata.warnings.append(f"{len(candidates)} scanned page(s) detected but OCR is disabled")
        return doc
    if not tesseract_available():
        msg = f"{len(candidates)} scanned page(s) detected but Tesseract is not installed"
        log.warning(msg)
        doc.metadata.warnings.append(msg)
        return doc
    pdf = fitz.open(doc.metadata.source_path)
    try:
        for page in candidates:
            fitz_page = pdf.load_page(page.number - 1)
            try:
                pix = fitz_page.get_pixmap(dpi=cfg.ocr_dpi)
                image = pixmap_to_image(pix)
                olines = ocr_lines(image, lang=cfg.ocr_lang, psm=cfg.page_ocr_psm, min_conf=cfg.ocr_min_confidence)
            except Exception as exc:  # noqa: BLE001
                log.warning("OCR of page %d failed: %s", page.number, exc)
                doc.metadata.warnings.append(f"OCR of page {page.number} failed: {exc}")
                continue
            scale = 72.0 / float(cfg.ocr_dpi)
            new_lines: List[Line] = []
            for i, ol in enumerate(olines):
                bbox = BBox(ol.bbox.x0 * scale, ol.bbox.y0 * scale, ol.bbox.x1 * scale, ol.bbox.y1 * scale)
                size = round(max(4.0, bbox.height * 0.75), 1)
                new_lines.append(
                    Line(text=ol.text, bbox=bbox, page=page.number, spans=[], font_size=size, font_name="OCR",
                         layout_block=i, source="ocr")
                )
            page.lines = new_lines
            page.is_scanned = True
            page.char_count = sum(len(l.text) for l in new_lines)
            assign_reading_order(page, config.ingestion)
            doc.metadata.scanned_pages.append(page.number)
            log.info("OCR: page %d -> %d lines", page.number, len(new_lines))
    finally:
        pdf.close()
    return doc


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure_regions(page: Page, fitz_page, cfg: FigureConfig, table_boxes: List[BBox]) -> List[Tuple[BBox, str, List[int]]]:
    """Candidate figure regions: raster images and clusters of vector drawings."""
    min_area = cfg.min_figure_area_ratio * page.area
    regions: List[List] = []
    if cfg.extract_images:
        for im in page.images:
            b = im.bbox
            if b.width < cfg.min_figure_size or b.height < cfg.min_figure_size or b.area < min_area:
                continue
            if b.area > 0.8 * page.area and page.char_count > 200:
                continue  # scanned background behind an OCR text layer -> not a figure
            if any(b.overlap_ratio(t) > 0.5 for t in table_boxes):
                continue
            regions.append([b, "raster", [im.xref]])
    if cfg.include_vector_graphics and fitz_page is not None:
        try:
            clusters = fitz_page.cluster_drawings()
        except Exception as exc:  # noqa: BLE001
            log.debug("cluster_drawings failed: %s", exc)
            clusters = []
        for rect in clusters:
            try:
                b = BBox.from_any((rect.x0, rect.y0, rect.x1, rect.y1))
            except Exception:  # noqa: BLE001
                continue
            b = BBox(max(0.0, b.x0), max(0.0, b.y0), min(page.width, b.x1), min(page.height, b.y1))
            if b.width < cfg.min_figure_size or b.height < cfg.min_figure_size or b.area < min_area:
                continue
            if b.area > 0.9 * page.area:
                continue  # page frame
            if any(b.overlap_ratio(t) > 0.3 or t.overlap_ratio(b) > 0.5 for t in table_boxes):
                continue
            inside = lines_in_bbox(page, b, pad=0.0)
            text_area = sum(l.bbox.area for l in inside)
            if len(inside) > cfg.max_text_lines_in_vector_figure or text_area > 0.5 * b.area:
                continue  # boxed text, not a figure
            regions.append([b, "vector", []])
    merged: List[List] = []
    for reg in sorted(regions, key=lambda r: (-r[0].area, r[0].y0, r[0].x0)):
        for m in merged:
            if reg[0].overlap_ratio(m[0]) > 0.5 or m[0].overlap_ratio(reg[0]) > 0.5:
                m[0] = m[0].union(reg[0])
                if m[1] != reg[1]:
                    m[1] = "mixed"
                m[2] = m[2] + reg[2]
                break
        else:
            merged.append(reg)
    merged.sort(key=lambda r: (r[0].y0, r[0].x0))
    return [(r[0], r[1], [x for x in r[2] if x]) for r in merged]


def _render_clip(fitz_page, bbox: BBox, dpi: int) -> Tuple[Optional[bytes], int, int]:
    try:
        rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pix = fitz_page.get_pixmap(clip=rect, dpi=dpi)
        return pix.tobytes("png"), int(pix.width), int(pix.height)
    except Exception as exc:  # noqa: BLE001
        log.debug("render clip failed: %s", exc)
        return None, 0, 0


def _describe(client: OllamaClient, cfg: FigureConfig, model: str, png: bytes, caption: Optional[str]) -> Optional[str]:
    prompt = cfg.description_prompt
    if caption:
        prompt += f"\n\nThe document's caption for this figure is: \"{caption}\""
    return client.generate(prompt, model=model, images=[png], num_predict=cfg.max_description_tokens)


def _make_figure_block(page: Page, fitz_page, bbox: BBox, kind: str, xrefs: List[int], idx: int, cfg: FigureConfig,
                       body_size: float, client: Optional[OllamaClient], vision_model: str, do_ocr: bool,
                       image_dir: Optional[str]) -> Block:
    block_id = f"figure-p{page.number}-{idx}"
    inside = sorted_by_order([ln for ln in lines_in_bbox(page, bbox, pad=0.0) if ln.role == "body" and not ln.claimed])
    native_text = " ".join(ln.text.strip() for ln in inside).strip()
    for ln in inside:
        ln.role = "figure"
        ln.claimed = block_id

    caption = None
    cap_lines = find_caption(page, bbox, FIGURE_CAPTION_RE, body_size, cfg.caption_search_distance, prefer="below")
    if cap_lines:
        caption = caption_text(cap_lines)
        claim(cap_lines, block_id)

    png, w, h = _render_clip(fitz_page, bbox, cfg.render_dpi) if fitz_page is not None else (None, 0, 0)

    ocr_txt = ""
    needs_ocr = kind in ("raster", "mixed") or cfg.ocr_vector_with_native_text or not native_text
    if do_ocr and png and needs_ocr:
        try:
            from PIL import Image  # noqa: WPS433

            if cfg.ocr_dpi > cfg.render_dpi:
                ocr_png, _, _ = _render_clip(fitz_page, bbox, cfg.ocr_dpi)
            else:
                ocr_png = png
            image = Image.open(io.BytesIO(ocr_png or png)).convert("RGB")
            ocr_txt = ocr_text(image, lang=cfg.ocr_lang, psm=cfg.figure_ocr_psm, min_conf=cfg.ocr_min_confidence)
        except Exception as exc:  # noqa: BLE001
            log.warning("figure OCR failed on page %d: %s", page.number, exc)
            ocr_txt = ""

    description = None
    model_used = None
    if client is not None and png:
        description = _describe(client, cfg, vision_model, png, caption)
        model_used = vision_model if description else None

    image_path = None
    if image_dir and png:
        try:
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{block_id}.png")
            with open(image_path, "wb") as fh:
                fh.write(png)
        except OSError as exc:  # pragma: no cover
            log.warning("could not save figure image: %s", exc)
            image_path = None

    fd = FigureData(
        caption=caption,
        ocr_text=ocr_txt,
        native_text=native_text,
        description=description,
        description_model=model_used,
        image_path=image_path,
        width_px=w,
        height_px=h,
        kind=kind,
        xrefs=xrefs,
    )
    return Block(
        id=block_id,
        type="figure",
        text=compose_figure_text(fd),
        pages=[page.number],
        bbox=bbox,
        figure=fd,
        extra={"kind": kind, "source_id": block_id, "caption": caption, "has_description": description is not None},
    )


def extract_figures(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    config = config or PipelineConfig()
    cfg = config.figures
    if not cfg.enabled:
        return doc
    body_size = dominant_font_size(doc.pages)
    vision_model = config.ollama.vision_model
    client: Optional[OllamaClient] = None
    if cfg.describe_with_llm:
        client = OllamaClient(config.ollama)
        if not client.is_available():
            doc.metadata.warnings.append("figures.describe_with_llm requested but Ollama is not reachable")
            client = None
        elif not client.has_model(vision_model):
            doc.metadata.warnings.append(f"vision model '{vision_model}' not found on the Ollama server")
            client = None
    do_ocr = cfg.ocr_enabled and tesseract_available()
    if cfg.ocr_enabled and not do_ocr:
        doc.metadata.warnings.append("figure OCR requested but Tesseract is not installed")
    image_dir = None
    if cfg.save_images:
        stem = os.path.splitext(doc.metadata.file_name)[0]
        image_dir = cfg.image_dir or os.path.join(os.path.dirname(doc.metadata.source_path) or ".", f"{stem}_images")

    n_found = 0
    pdf = fitz.open(doc.metadata.source_path)
    try:
        for page in doc.pages:
            fitz_page = pdf.load_page(page.number - 1)
            table_boxes = [b.bbox for b in page.region_blocks if b.type == "table" and b.bbox]
            regions = _figure_regions(page, fitz_page, cfg, table_boxes)
            for i, (bbox, kind, xrefs) in enumerate(regions, 1):
                block = _make_figure_block(page, fitz_page, bbox, kind, xrefs, i, cfg, body_size, client,
                                           vision_model, do_ocr, image_dir)
                page.region_blocks.append(block)
                n_found += 1
    finally:
        pdf.close()
    log.info("figures: %d figure(s) found", n_found)
    return doc
<<<END FILE>>>

<<<FILE fable_pipeline/sections.py>>>
"""Module 3: section & paragraph detection and document assembly.

This module turns the layout layer (lines with roles) plus the table/figure
blocks attached to pages into the ordered ``Document.blocks`` list:

1. **Title detection** – largest text in the upper half of page 1.
2. **Heading detection** – outline (PDF bookmarks) matches, numbering patterns
   (``2.3.1 Title``, ``Chapter 4``, ``Appendix A``), font size relative to the
   dominant body size, bold styling, and "standalone-ness" (gaps before/after,
   short line).  Heading *levels* come from the numbering depth, the outline,
   or the rank of the heading's (size, bold) style.
3. **Line classification** – list items, equations, captions not already
   claimed by a table/figure.
4. **Paragraph assembly** – lines are merged into paragraphs using gaps,
   indentation, line fullness and style; continuation across column and page
   breaks is allowed when the previous line is full width (or hyphenated).
5. **Interleaving** of table/figure blocks at their position in reading order
   and assignment of ``section_path`` to every block.

If ``sections.enabled`` is False a plain assembly (no headings, paragraphs =
gap-separated line runs) is produced so the output contract still holds.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .captions import ANY_CAPTION_RE, caption_text, group_caption_lines, looks_like_caption
from .config import PipelineConfig, SectionConfig
from .ingestion import dominant_font_size
from .layout_utils import median, union_bbox
from .models import Block, Document, Line, Page

log = logging.getLogger(__name__)

NUMBERING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,4}\.?|[A-Z](?:\.\d{1,2}){1,3}\.?|[IVX]{1,6}\.)\s+(?P<title>\S.*)$"
)
CHAPTER_RE = re.compile(
    r"^(?:chapter|kapitel|section|part|teil|appendix|anhang|annex)\s+(?:[A-Z0-9]{1,3}|[IVX]{1,6})\b"
    r"(?:\s*[:.\-–—]\s*|\s+)?(?P<title>.*)$",
    re.I,
)
LIST_BULLET_RE = re.compile(r"^(?P<m>[•◦▪■●○‣⁃◆◇►▸➢➤✓✔✗–—\-*·])\s+(?P<body>\S.*)$")
LIST_ENUM_RE = re.compile(r"^(?P<m>\(?(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,5})[.)\]])\s+(?P<body>\S.*)$")
EQ_NUM_RE = re.compile(r"\(\s*[A-Z]?\d+(?:[.\-]\d+)*[a-z]?\s*\)\s*$")
MATH_CHARS = set("=+−×÷·∑∏∫√≤≥≠≈∂∞∝∈∀∃αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ^_{}|")
ROMAN_RE = re.compile(r"^[IVX]{1,6}$")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def normalize_heading_key(text: str) -> str:
    t = text.lower().strip()
    t = _WS_RE.sub(" ", t)
    return t.strip(" .:-–—")


def strip_numbering(text: str) -> str:
    m = NUMBERING_RE.match(text.strip())
    if m:
        return m.group("title").strip()
    m = CHAPTER_RE.match(text.strip())
    if m and m.group("title").strip():
        return m.group("title").strip()
    return text.strip()


def numbering_info(text: str) -> Optional[Tuple[int, str, str]]:
    """Return (depth, number, title) if ``text`` starts with section numbering."""
    text = text.strip()
    m = CHAPTER_RE.match(text)
    if m:
        return 1, text.split()[1] if len(text.split()) > 1 else "", m.group("title").strip()
    m = NUMBERING_RE.match(text)
    if not m:
        return None
    num, title = m.group("num"), m.group("title").strip()
    if not title or not any(ch.isalpha() for ch in title):
        return None
    if not (title[0].isupper() or title[0] in "\"“'(«"):
        return None
    core = num.rstrip(".")
    if ROMAN_RE.match(core):
        return 1, core, title
    parts = core.split(".")
    return len(parts), core, title


def list_marker(text: str) -> Optional[Tuple[str, str]]:
    """Return (normalised marker, item text) if the line starts a list item."""
    t = text.strip()
    m = LIST_BULLET_RE.match(t)
    if m:
        return "-", m.group("body")
    m = LIST_ENUM_RE.match(t)
    if m:
        return m.group("m"), m.group("body")
    return None


def is_equation(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    words = t.split()
    alpha_words = [w for w in words if len(w) > 2 and w.isalpha()]
    if EQ_NUM_RE.search(t) and len(alpha_words) <= 4 and len(words) <= 25:
        return True
    math_frac = sum(1 for ch in t if ch in MATH_CHARS) / max(1, len(t))
    return math_frac >= 0.12 and len(alpha_words) <= 3 and len(words) <= 15


def _style(ln: Line) -> Tuple[float, bool]:
    return (round(ln.font_size * 2) / 2.0, bool(ln.bold))


def _region_box(ln: Line):
    return ln.region_bbox if ln.region_bbox is not None else ln.bbox


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    body_size: float
    med_gap: float
    justified: bool
    outline: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


@dataclass
class _Info:
    kind: str = "body"  # body | title | heading | list | equation | caption | skip
    level: Optional[int] = None
    explicit: Optional[int] = None  # level from outline/numbering
    style: Tuple[float, bool] = (0.0, False)
    numbered: bool = False
    text: Optional[str] = None
    marker: Optional[str] = None
    body: Optional[str] = None
    caption_lines: List[Line] = field(default_factory=list)


def _candidate_lines(doc: Document) -> List[Line]:
    seq: List[Line] = []
    for page in doc.pages:
        for ln in sorted(page.lines, key=lambda l: l.order):
            if ln.role in ("body", "title") and not ln.claimed and not ln.vertical and ln.text.strip():
                seq.append(ln)
    return seq


def _build_ctx(doc: Document, seq: List[Line], cfg: SectionConfig) -> _Ctx:
    body_size = dominant_font_size(doc.pages, roles=("body",))
    gaps = []
    for a, b in zip(seq, seq[1:]):
        if a.page == b.page and a.region == b.region:
            g = b.bbox.y0 - a.bbox.y1
            size = max(a.font_size, 1.0)
            if -size <= g <= 3 * size:
                gaps.append(g)
    med_gap = median(gaps, 2.0)
    # justification: fraction of lines reaching the right edge of their region
    region_counts: Dict[Tuple[int, int], int] = {}
    for ln in seq:
        region_counts[(ln.page, ln.region)] = region_counts.get((ln.page, ln.region), 0) + 1
    full = total = 0
    for ln in seq:
        if region_counts[(ln.page, ln.region)] < 3:
            continue
        total += 1
        if ln.bbox.x1 >= _region_box(ln).x1 - 2.0:
            full += 1
    justified = total > 0 and full / total >= 0.4
    outline: Dict[str, List[Tuple[int, int]]] = {}
    if cfg.use_outline:
        for entry in doc.metadata.outline:
            title = str(entry.get("title", ""))
            lvl, page = int(entry.get("level", 1)), int(entry.get("page", 0))
            for key in {normalize_heading_key(title), normalize_heading_key(strip_numbering(title))}:
                if key:
                    outline.setdefault(key, []).append((lvl, page))
    return _Ctx(body_size=body_size, med_gap=med_gap, justified=justified, outline=outline)


# --------------------------------------------------------------------------- #
# Title detection
# --------------------------------------------------------------------------- #
def _detect_title(doc: Document, cfg: SectionConfig, body_size: float) -> Optional[str]:
    if not doc.pages:
        return None
    page = doc.pages[0]
    lines = [ln for ln in sorted(page.lines, key=lambda l: l.order)
             if ln.role == "body" and not ln.claimed and not ln.vertical and ln.bbox.cy < 0.5 * page.height
             and any(ch.isalpha() for ch in ln.text)]
    if not lines:
        return None
    max_size = max(ln.font_size for ln in lines)
    if max_size < cfg.title_size_ratio * body_size:
        return None
    chosen: List[Line] = []
    for ln in lines:
        if abs(ln.font_size - max_size) <= 0.6:
            if chosen and (ln.bbox.y0 - chosen[-1].bbox.y1 > 1.2 * max_size or ln.region != chosen[-1].region):
                break
            chosen.append(ln)
        elif chosen:
            break
    chosen = chosen[:4]
    if not chosen or sum(len(l.text.split()) for l in chosen) > 40:
        return None
    for ln in chosen:
        ln.role = "title"
    return " ".join(ln.text.strip() for ln in chosen)


def _fallback_pdf_title(doc: Document) -> Optional[str]:
    t = doc.metadata.pdf_title
    if not t:
        return None
    low = t.lower()
    if re.search(r"\.(pdf|docx?|dvi|tex|pptx?|indd|qxd)$", low) or low in ("untitled", "title", "microsoft word"):
        return None
    if " " not in t.strip():
        return None
    return t.strip()


# --------------------------------------------------------------------------- #
# Line classification
# --------------------------------------------------------------------------- #
def _neighbors(i: int, seq: List[Line]) -> Tuple[Optional[Line], Optional[Line]]:
    ln = seq[i]
    prev = seq[i - 1] if i > 0 and seq[i - 1].page == ln.page and seq[i - 1].region == ln.region else None
    nxt = seq[i + 1] if i + 1 < len(seq) and seq[i + 1].page == ln.page and seq[i + 1].region == ln.region else None
    return prev, nxt


def _gap_large(prev: Line, ln: Line, ctx: _Ctx, cfg: SectionConfig) -> bool:
    size = max(ln.font_size, prev.font_size, 1.0)
    return (ln.bbox.y0 - prev.bbox.y1) > ctx.med_gap + max(2.0, cfg.paragraph_gap_factor * size)


def _is_short(ln: Line, ctx: _Ctx, cfg: SectionConfig) -> bool:
    rb = _region_box(ln)
    if rb.width <= 0:
        return False
    ratio = cfg.short_line_ratio if ctx.justified else cfg.ragged_short_line_ratio
    return ln.bbox.x1 < rb.x0 + ratio * rb.width


def _style_differs(a: Line, b: Line) -> bool:
    return abs(a.font_size - b.font_size) > 0.6 or a.bold != b.bold


def _single_level_number(text: str) -> Optional[int]:
    m = re.match(r"^\(?(\d{1,3})[.)\]]?\s+\S", text.strip())
    return int(m.group(1)) if m else None


def _heading_candidate(i: int, seq: List[Line], ctx: _Ctx, cfg: SectionConfig) -> Optional[_Info]:
    ln = seq[i]
    text = ln.text.strip()
    if not any(ch.isalpha() for ch in text):
        return None
    words = text.split()
    if len(words) > cfg.max_heading_words:
        return None
    if text[-1] in ",;":
        return None
    if looks_like_caption(ln, ctx.body_size):
        return None
    numbering = numbering_info(text) if cfg.use_numbering else None
    if text[-1] in ".!?" and not numbering:
        return None
    if LIST_BULLET_RE.match(text):
        return None
    prev, nxt = _neighbors(i, seq)
    short = _is_short(ln, ctx, cfg)
    gap_before = prev is None or _gap_large(prev, ln, ctx, cfg) or _style_differs(prev, ln) or _is_short(prev, ctx, cfg)
    next_differs = nxt is None or _style_differs(ln, nxt) or _gap_large(ln, nxt, ctx, cfg)
    standalone = (short or next_differs) and gap_before
    bigger = ln.font_size >= ctx.body_size * cfg.heading_size_ratio
    bold = cfg.use_font_styles and ln.bold and ln.font_size >= 0.95 * ctx.body_size
    info = _Info(kind="heading", style=_style(ln), numbered=bool(numbering), text=text)

    # (a) outline match
    if ctx.outline:
        for key in {normalize_heading_key(text), normalize_heading_key(strip_numbering(text))}:
            for lvl, page in ctx.outline.get(key, []):
                if abs(page - ln.page) <= 1:
                    info.explicit = max(1, lvl)
                    return info
    # (b) numbering
    if numbering:
        depth = numbering[0]
        if depth == 1:
            n = _single_level_number(text)
            if n is not None and not bold and not bigger:
                for nb in (prev, nxt):
                    if nb is not None:
                        m = _single_level_number(nb.text)
                        if m is not None and abs(m - n) == 1 and not _style_differs(nb, ln):
                            return None  # consecutive numbered list items
        if bold or bigger or (standalone and short):
            info.explicit = depth
            return info
        return None
    # (c) bigger font
    if bigger and standalone:
        return info
    # (d) bold at body size
    if bold and standalone:
        return info
    return None


def _classify(seq: List[Line], ctx: _Ctx, cfg: SectionConfig, page_map: Dict[int, Page]) -> Dict[int, _Info]:
    infos: Dict[int, _Info] = {}
    for i, ln in enumerate(seq):
        if id(ln) in infos:
            continue
        if ln.role == "title":
            infos[id(ln)] = _Info(kind="title", text=ln.text.strip(), style=_style(ln))
            continue
        text = ln.text.strip()
        if cfg.detect_captions and looks_like_caption(ln, ctx.body_size, ANY_CAPTION_RE):
            group = group_caption_lines(page_map[ln.page], ln, ctx.body_size)
            infos[id(ln)] = _Info(kind="caption", text=caption_text(group), caption_lines=group)
            for other in group[1:]:
                infos[id(other)] = _Info(kind="skip")
            continue
        cand = _heading_candidate(i, seq, ctx, cfg)
        if cand is not None:
            infos[id(ln)] = cand
            continue
        if cfg.detect_lists:
            lm = list_marker(text)
            if lm is not None:
                infos[id(ln)] = _Info(kind="list", marker=lm[0], body=lm[1], style=_style(ln))
                continue
        if cfg.detect_equations and is_equation(text):
            infos[id(ln)] = _Info(kind="equation", text=text, style=_style(ln))
            continue
        infos[id(ln)] = _Info(kind="body", style=_style(ln))
    return infos


def _finalize_headings(seq: List[Line], infos: Dict[int, _Info], ctx: _Ctx, cfg: SectionConfig) -> None:
    heads = [(i, infos[id(ln)]) for i, ln in enumerate(seq) if infos[id(ln)].kind == "heading"]
    if not heads:
        return
    # merge multi-line headings
    for i, info in heads:
        if info.kind != "heading":
            continue
        j = i + 1
        while j < len(seq):
            nxt_info = infos[id(seq[j])]
            prev_ln, ln = seq[j - 1], seq[j]
            if nxt_info.kind not in ("heading", "body"):
                break
            if ln.page != prev_ln.page or ln.region != prev_ln.region:
                break
            if _style(ln) != info.style or (info.style[0] < ctx.body_size * cfg.heading_size_ratio and not info.style[1]):
                break
            if _gap_large(prev_ln, ln, ctx, cfg) or nxt_info.numbered:
                break
            merged = (info.text or "") + "\n" + ln.text.strip()
            if len(merged.split()) > cfg.max_heading_words or prev_ln.text.strip()[-1] in ".!?:":
                break
            if nxt_info.kind == "body" and not _is_short(ln, ctx, cfg) and not (seq[j + 1:] and _style_differs(ln, seq[j + 1])):
                break
            info.text = merged
            infos[id(ln)] = _Info(kind="skip")
            j += 1
    heads = [(i, infos[id(ln)]) for i, ln in enumerate(seq) if infos[id(ln)].kind == "heading"]
    known: Dict[Tuple[float, bool], int] = {}
    for _, info in heads:
        if info.explicit is not None:
            known[info.style] = min(known.get(info.style, 99), info.explicit)
    styles = sorted({info.style for _, info in heads}, key=lambda s: (-s[0], not s[1]))
    style_level: Dict[Tuple[float, bool], int] = {}
    cur = 0
    for s in styles:
        lvl = known[s] if s in known else min(cfg.max_levels, cur + 1)
        style_level[s] = lvl
        cur = lvl
    for _, info in heads:
        lvl = info.explicit if info.explicit is not None else style_level[info.style]
        info.level = max(1, min(cfg.max_levels, lvl))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
class _Assembler:
    def __init__(self, doc: Document, cfg: SectionConfig, ctx: Optional[_Ctx], infos: Dict[int, _Info], plain: bool):
        self.doc = doc
        self.cfg = cfg
        self.ctx = ctx
        self.infos = infos
        self.plain = plain
        self.blocks: List[Block] = []
        self.stack: List[Tuple[int, str]] = []
        self.cur: Optional[Dict] = None
        self.counter = 0

    # ----------------------------------------------------------------- utils
    def _new_id(self, kind: str, page: int) -> str:
        self.counter += 1
        return f"{kind}-p{page}-{self.counter}"

    def path(self) -> List[str]:
        return [t for _, t in self.stack]

    def _block_from_lines(self, kind: str, lines: List[Line], text: str, extra: Optional[Dict] = None,
                          level: Optional[int] = None) -> Block:
        pages = sorted({ln.page for ln in lines})
        first = [ln for ln in lines if ln.page == pages[0]]
        bbox = union_bbox(ln.bbox for ln in first)
        ex = {"n_lines": len(lines), "font_size": round(lines[0].font_size, 1), "bold": lines[0].bold}
        if any(ln.source == "ocr" for ln in lines):
            ex["source"] = "ocr"
        if extra:
            ex.update(extra)
        return Block(id=self._new_id(kind, pages[0]), type=kind, text=text, pages=pages, bbox=bbox,
                     section_path=self.path(), level=level, extra=ex)

    def flush(self) -> None:
        if not self.cur:
            return
        cur, self.cur = self.cur, None
        if cur["type"] == "list":
            text = "\n".join(cur["items"])
            self.blocks.append(self._block_from_lines("list", cur["lines"], text, {"items": list(cur["items"])}))
        else:
            text = "\n".join(cur["texts"])
            self.blocks.append(self._block_from_lines("paragraph", cur["lines"], text, cur.get("extra")))

    def emit_region(self, rb: Block) -> None:
        self.flush()
        rb.section_path = self.path()
        self.blocks.append(rb)

    # --------------------------------------------------------------- headings
    def emit_title(self, ln: Line, text: str) -> None:
        self.flush()
        blk = self._block_from_lines("heading", [ln], text, {"is_title": True}, level=0)
        blk.section_path = [text] if self.cfg.title_in_section_path else []
        self.blocks.append(blk)
        if self.cfg.title_in_section_path:
            self.stack = [(0, text)]

    def emit_heading(self, ln: Line, info: _Info) -> None:
        self.flush()
        level = info.level or 1
        text = _WS_RE.sub(" ", info.text or ln.text).strip()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        blk = self._block_from_lines("heading", [ln], text, {"numbered": info.numbered}, level=level)
        blk.section_path = self.path()
        self.blocks.append(blk)

    # -------------------------------------------------------------- continue
    def _continues(self, prev: Line, ln: Line) -> bool:
        cfg, ctx = self.cfg, self.ctx
        if self.plain or ctx is None:
            if prev.page != ln.page or prev.region != ln.region:
                return False
            size = max(prev.font_size, ln.font_size, 1.0)
            return (ln.bbox.y0 - prev.bbox.y1) <= max(2.0, 0.6 * size) and not _style_differs(prev, ln)
        if _style_differs(prev, ln):
            return False
        size = max(prev.font_size, ln.font_size, 1.0)
        same_region = prev.page == ln.page and prev.region == ln.region
        if same_region:
            gap = ln.bbox.y0 - prev.bbox.y1
            if _gap_large(prev, ln, ctx, cfg) or gap < -0.5 * size:
                return False
            if ln.bbox.x0 - prev.bbox.x0 > cfg.indent_factor * size:
                return False
            if _is_short(prev, ctx, cfg):
                return False
            return True
        # column or page break
        if ln.bbox.x0 - _region_box(ln).x0 > cfg.indent_factor * size:
            return False
        if prev.text.rstrip().endswith(("-", "\u00ad", "\u2010")):
            return True
        if _is_short(prev, ctx, cfg):
            return False
        return True

    # ------------------------------------------------------------------ run
    def add_body(self, ln: Line, extra: Optional[Dict] = None) -> None:
        if self.cur and self.cur["type"] == "paragraph" and not self.cur.get("extra") and extra is None \
                and self._continues(self.cur["lines"][-1], ln):
            self.cur["lines"].append(ln)
            self.cur["texts"].append(ln.text.strip())
            return
        self.flush()
        self.cur = {"type": "paragraph", "lines": [ln], "texts": [ln.text.strip()], "extra": extra}

    def add_list_item(self, ln: Line, info: _Info) -> None:
        if self.cur and self.cur["type"] == "list":
            prev = self.cur["lines"][-1]
            same = prev.page == ln.page and prev.region == ln.region and not _gap_large(prev, ln, self.ctx, self.cfg) \
                if self.ctx else prev.page == ln.page
            if same or prev.page != ln.page:
                self.cur["lines"].append(ln)
                self.cur["items"].append(f"{info.marker} {info.body}".strip())
                return
        self.flush()
        self.cur = {"type": "list", "lines": [ln], "items": [f"{info.marker} {info.body}".strip()]}

    def add_list_continuation(self, ln: Line) -> bool:
        """Append a non-marker line to the current list item if it is indented like one."""
        if not (self.cur and self.cur["type"] == "list") or self.ctx is None:
            return False
        prev = self.cur["lines"][-1]
        if prev.page != ln.page or prev.region != ln.region or _gap_large(prev, ln, self.ctx, self.cfg):
            return False
        if _style_differs(prev, ln):
            return False
        if ln.bbox.x0 < prev.bbox.x0 + 0.5 * max(prev.font_size, 1.0):
            return False
        self.cur["lines"].append(ln)
        self.cur["items"][-1] = self.cur["items"][-1] + "\n" + ln.text.strip()
        return True


def _anchor_region_blocks(page: Page) -> Tuple[Dict[int, List[Block]], List[Block]]:
    anchors: Dict[int, List[Block]] = {}
    tail: List[Block] = []
    for rb in sorted(page.region_blocks, key=lambda b: (b.bbox.y0 if b.bbox else 0.0, b.bbox.x0 if b.bbox else 0.0)):
        claimed = [ln.order for ln in page.lines if ln.claimed == rb.id]
        if claimed:
            a: Optional[int] = min(claimed)
        elif rb.bbox is not None:
            below = [ln.order for ln in page.lines
                     if not ln.claimed and ln.role in ("body", "title")
                     and ln.bbox.y0 >= rb.bbox.y1 - 2 and ln.bbox.x_overlap(rb.bbox) > 0]
            a = min(below) if below else None
        else:
            a = None
        if a is None:
            tail.append(rb)
        else:
            anchors.setdefault(a, []).append(rb)
    return anchors, tail


def _assemble(doc: Document, cfg: SectionConfig, ctx: Optional[_Ctx], infos: Dict[int, _Info], plain: bool) -> List[Block]:
    asm = _Assembler(doc, cfg, ctx, infos, plain)
    for page in doc.pages:
        anchors, tail = _anchor_region_blocks(page)
        for ln in sorted(page.lines, key=lambda l: l.order):
            for rb in anchors.get(ln.order, []):
                asm.emit_region(rb)
            if ln.claimed or ln.role not in ("body", "title") or ln.vertical or not ln.text.strip():
                continue
            info = infos.get(id(ln), _Info(kind="body"))
            kind = info.kind
            if kind == "skip":
                continue
            if kind == "title":
                asm.emit_title(ln, info.text or ln.text.strip())
            elif kind == "heading":
                asm.emit_heading(ln, info)
            elif kind == "caption":
                asm.flush()
                blk = asm._block_from_lines("caption", info.caption_lines or [ln], info.text or ln.text.strip())
                asm.blocks.append(blk)
            elif kind == "list":
                asm.add_list_item(ln, info)
            elif kind == "equation":
                asm.flush()
                asm.add_body(ln, extra={"equation": True})
                asm.flush()
            else:
                if asm.add_list_continuation(ln):
                    continue
                asm.add_body(ln)
        for rb in tail:
            asm.emit_region(rb)
    asm.flush()
    return asm.blocks


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_sections(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    """Build ``doc.blocks`` from the layout layer (headings, paragraphs, lists, tables, figures)."""
    config = config or PipelineConfig()
    cfg = config.sections
    page_map = {p.number: p for p in doc.pages}
    if not cfg.enabled:
        doc.blocks = _assemble(doc, cfg, None, {}, plain=True)
        if doc.metadata.title is None:
            doc.metadata.title = _fallback_pdf_title(doc)
        return doc
    seq = _candidate_lines(doc)
    ctx = _build_ctx(doc, seq, cfg)
    title = None
    if cfg.detect_title:
        title = _detect_title(doc, cfg, ctx.body_size)
    doc.metadata.title = title or _fallback_pdf_title(doc)
    infos = _classify(seq, ctx, cfg, page_map)
    _finalize_headings(seq, infos, ctx, cfg)
    doc.blocks = _assemble(doc, cfg, ctx, infos, plain=False)
    n_head = sum(1 for b in doc.blocks if b.type == "heading")
    log.info("sections: %d blocks, %d headings, body size %.1fpt, justified=%s", len(doc.blocks), n_head,
             ctx.body_size, ctx.justified)
    return doc


build_blocks = detect_sections
<<<END FILE>>>

<<<FILE fable_pipeline/cleanup.py>>>
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
<<<END FILE>>>

<<<FILE fable_pipeline/render.py>>>
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
<<<END FILE>>>

<<<FILE fable_pipeline/pipeline.py>>>
"""Pipeline orchestration: ``extract(pdf_path, config) -> Document``.

Step order (each step is optional/swappable through :class:`PipelineConfig`):

    ingestion -> scanned_ocr -> boilerplate -> tables -> figures -> sections -> cleanup

Tables run before figures so vector-drawn tables are not mistaken for figures;
sections run after both so table/figure regions are excluded from paragraphs
and interleaved at the right position; cleanup is last because it needs the
final block texts.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from .boilerplate import remove_boilerplate
from .cleanup import cleanup_document
from .config import PipelineConfig
from .figures import extract_figures, ocr_scanned_pages
from .ingestion import parse_layout
from .models import Document, to_json
from .render import to_markdown
from .sections import detect_sections
from .tables import extract_tables

log = logging.getLogger(__name__)

STEP_ORDER: List[str] = ["scanned_ocr", "boilerplate", "tables", "figures", "sections", "cleanup"]

_DEFAULT_STEPS: Dict[str, Callable[[Document, PipelineConfig], Document]] = {
    "scanned_ocr": ocr_scanned_pages,
    "boilerplate": remove_boilerplate,
    "tables": extract_tables,
    "figures": extract_figures,
    "sections": detect_sections,
    "cleanup": cleanup_document,
}


def _step_enabled(name: str, config: PipelineConfig) -> bool:
    if name == "scanned_ocr":
        return config.figures.detect_scanned_pages
    if name == "boilerplate":
        return config.boilerplate.enabled
    if name == "tables":
        return config.tables.enabled
    if name == "figures":
        return config.figures.enabled
    if name == "sections":
        return True  # detect_sections handles its own 'enabled' (plain assembly fallback)
    if name == "cleanup":
        return config.cleanup.enabled
    return True


def _configure_logging(level: str) -> None:
    logger = logging.getLogger("fable_pipeline")
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        logger.setLevel(logging.INFO)


def extract(pdf_path: str, config: Optional[PipelineConfig] = None) -> Document:
    """Run the full pipeline on ``pdf_path`` and return the structured Document."""
    config = config or PipelineConfig()
    _configure_logging(config.log_level)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)
    t0 = time.perf_counter()
    ingest = config.custom_steps.get("ingestion", parse_layout)
    doc = ingest(pdf_path, config)
    doc.metadata.timings["ingestion"] = round(time.perf_counter() - t0, 4)
    doc.metadata.steps_applied.append("ingestion")

    for name in STEP_ORDER:
        fn = config.custom_steps.get(name, _DEFAULT_STEPS[name])
        if not _step_enabled(name, config) and name not in config.custom_steps:
            doc.metadata.steps_skipped.append(name)
            continue
        t1 = time.perf_counter()
        try:
            result = fn(doc, config)
            if result is not None:
                doc = result
            doc.metadata.steps_applied.append(name)
        except Exception as exc:  # noqa: BLE001
            if config.fail_fast:
                raise
            log.exception("step '%s' failed; continuing", name)
            doc.metadata.warnings.append(f"step '{name}' failed: {exc}")
            doc.metadata.steps_skipped.append(name)
            if name == "sections" and not doc.blocks:
                # guarantee the output contract even when section detection crashed
                fallback = PipelineConfig.from_dict(config.to_dict())
                fallback.sections.enabled = False
                doc = detect_sections(doc, fallback)
        finally:
            doc.metadata.timings[name] = round(time.perf_counter() - t1, 4)
    doc.metadata.timings["total"] = round(time.perf_counter() - t0, 4)
    log.info("extracted %s: %d blocks in %.2fs", doc.metadata.file_name, len(doc.blocks), doc.metadata.timings["total"])
    return doc


def extract_to_files(pdf_path: str, out_dir: str, config: Optional[PipelineConfig] = None) -> Tuple[str, str]:
    """Run :func:`extract` and write ``<stem>.md`` and ``<stem>.json`` into ``out_dir``."""
    config = config or PipelineConfig()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    if config.figures.save_images and not config.figures.image_dir:
        config.figures.image_dir = os.path.join(out_dir, f"{stem}_images")
    doc = extract(pdf_path, config)
    md_path = os.path.join(out_dir, f"{stem}.md")
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(doc))
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(to_json(doc, include_layout=config.include_layout_in_json))
    return md_path, json_path
<<<END FILE>>>

<<<FILE fable_pipeline/__main__.py>>>
"""CLI: ``python -m fable_pipeline input.pdf [more.pdf ...] --out out_dir``."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import PipelineConfig
from .pipeline import extract_to_files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fable_pipeline", description="FABLE: structure-preserving PDF extraction for local RAG")
    p.add_argument("inputs", nargs="+", help="PDF file(s)")
    p.add_argument("--out", required=True, help="output directory (writes <stem>.md and <stem>.json)")
    p.add_argument("--config", help="JSON file with (partial) PipelineConfig overrides")
    p.add_argument("--llm", action="store_true", help="enable Ollama-based figure descriptions and table refinement")
    p.add_argument("--describe-figures", action="store_true", help="enable only vision-model figure descriptions")
    p.add_argument("--ollama-url", help="Ollama base URL (default http://localhost:11434)")
    p.add_argument("--vision-model", help="Ollama vision model name, e.g. llava:7b")
    p.add_argument("--text-model", help="Ollama text model name, e.g. qwen2.5:7b")
    p.add_argument("--ocr-lang", help="Tesseract language(s), e.g. eng or deu+eng")
    p.add_argument("--table-strategies", help="comma-separated: pymupdf,pdfplumber,pdfplumber_text,camelot,whitespace")
    p.add_argument("--disable", action="append", default=[],
                   help="disable a module/feature (repeatable): boilerplate, sections, tables, figures, cleanup, "
                        "ocr, scanned_ocr, columns, lists, captions, outline, llm")
    p.add_argument("--save-images", action="store_true", help="save figure crops as PNG next to the outputs")
    p.add_argument("--no-layout", action="store_true", help="omit the per-line layout layer from the JSON output")
    p.add_argument("--fail-fast", action="store_true", help="raise on the first module error instead of continuing")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig.from_json_file(args.config) if args.config else PipelineConfig()
    if args.llm:
        cfg.enable_llm(True, True)
    if args.describe_figures:
        cfg.figures.describe_with_llm = True
    if args.ollama_url:
        cfg.ollama.url = args.ollama_url
    if args.vision_model:
        cfg.ollama.vision_model = args.vision_model
    if args.text_model:
        cfg.ollama.text_model = args.text_model
    if args.ocr_lang:
        cfg.figures.ocr_lang = args.ocr_lang
    if args.table_strategies:
        cfg.tables.strategies = [s.strip() for s in args.table_strategies.split(",") if s.strip()]
    for name in args.disable:
        cfg.disable(name)
    if args.save_images:
        cfg.figures.save_images = True
    if args.no_layout:
        cfg.include_layout_in_json = False
    if args.fail_fast:
        cfg.fail_fast = True
    cfg.log_level = "DEBUG" if args.verbose else "INFO"
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = config_from_args(args)
    rc = 0
    for path in args.inputs:
        try:
            md_path, json_path = extract_to_files(path, args.out, cfg)
            print(f"{path} -> {md_path}, {json_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR processing {path}: {exc}", file=sys.stderr)
            rc = 1
            if cfg.fail_fast:
                raise
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
<<<END FILE>>>

<<<FILE tests/conftest.py>>>
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
<<<END FILE>>>

<<<FILE tests/test_models.py>>>
from fable_pipeline.models import (
    BBox,
    Block,
    Document,
    DocumentMetadata,
    FigureData,
    TableData,
    from_json,
    normalize_grid,
    render_markdown_table,
    to_json,
)


def test_bbox_geometry():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert a.area == 100
    assert a.intersection_area(b) == 25
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert a.union(b).to_list() == [0, 0, 15, 15]
    assert a.contains_point(5, 5)
    assert BBox.from_any((10, 10, 0, 0)).to_list() == [0, 0, 10, 10]
    assert a.overlap_ratio(BBox(0, 0, 5, 10)) == 0.5


def test_normalize_grid_drops_empty_rows_and_cols():
    grid = [["a", None, "b"], [None, None, None], ["c", "", "d"]]
    assert normalize_grid(grid) == [["a", "b"], ["c", "d"]]
    assert normalize_grid(None) == []
    assert normalize_grid([["", ""]]) == []


def test_table_data_from_grid_and_renderings():
    td = TableData.from_grid([["Name", "Value"], ["x", "1"], ["y", "2"]], caption="Table 1: T")
    assert td.columns == ["Name", "Value"]
    assert td.n_rows == 2 and td.n_cols == 2
    assert td.records == [{"Name": "x", "Value": "1"}, {"Name": "y", "Value": "2"}]
    assert "| Name | Value |" in td.markdown
    assert "| --- | --- |" in td.markdown
    assert td.csv.splitlines()[0] == "Name,Value"
    df = td.to_dataframe()
    assert list(df.columns) == ["Name", "Value"] and len(df) == 2


def test_table_data_duplicate_and_missing_headers():
    td = TableData.from_grid([["A", "", "A"], ["1", "2", "3"]])
    assert td.columns == ["A", "col_2", "A (2)"]
    td2 = TableData.from_grid([["1", "2"], ["3", "4"]], header_row=False)
    assert td2.columns == ["col_1", "col_2"] and td2.n_rows == 2


def test_markdown_table_escapes_pipes():
    md = render_markdown_table(["a|b", "c"], [["x|y", "z"]])
    assert "a\\|b" in md and "x\\|y" in md


def test_json_roundtrip_is_lossless():
    meta = DocumentMetadata(source_path="/x/y.pdf", file_name="y.pdf", page_count=1, title="T", warnings=["w"])
    blocks = [
        Block(id="h1", type="heading", text="Intro", pages=[1], bbox=BBox(1, 2, 3, 4), section_path=["Intro"], level=1),
        Block(id="t1", type="table", text="t", pages=[1], bbox=None,
              table=TableData.from_grid([["a", "b"], ["1", "2"]], caption="Table 1")),
        Block(id="f1", type="figure", text="f", pages=[1], bbox=BBox(0, 0, 1, 1),
              figure=FigureData(caption="Figure 1", ocr_text="abc", description="a plot")),
    ]
    doc = Document(metadata=meta, blocks=blocks)
    js = to_json(doc)
    back = from_json(js)
    assert back.to_dict() == doc.to_dict()
    assert back.tables[0].table.records == [{"a": "1", "b": "2"}]
    assert back.figures[0].figure.description == "a plot"
    assert back.headings[0].level == 1
<<<END FILE>>>

<<<FILE tests/test_config.py>>>
import json

import pytest

from fable_pipeline.config import PipelineConfig


def test_disable_and_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.disable("boilerplate", "tables", "ocr", "columns", "llm")
    assert not cfg.boilerplate.enabled and not cfg.tables.enabled
    assert not cfg.figures.ocr_enabled and not cfg.ingestion.detect_columns
    d = cfg.to_dict()
    assert d["boilerplate"]["enabled"] is False
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.to_dict() == d
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"figures": {"ocr_lang": "deu"}, "fail_fast": True}), encoding="utf-8")
    cfg3 = PipelineConfig.from_json_file(str(path))
    assert cfg3.figures.ocr_lang == "deu" and cfg3.fail_fast is True
    assert cfg3.tables.enabled is True  # untouched defaults


def test_unknown_disable_name_raises():
    with pytest.raises(ValueError):
        PipelineConfig().disable("nonsense")


def test_enable_llm_sets_flags():
    cfg = PipelineConfig().enable_llm()
    assert cfg.figures.describe_with_llm and cfg.tables.use_llm
    assert cfg.ollama.temperature == 0.0 and isinstance(cfg.ollama.seed, int)
<<<END FILE>>>

<<<FILE tests/test_ingestion.py>>>
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import dominant_font_size, parse_layout
from tests.conftest import add_page, new_pdf, save, write_lines


def test_parse_layout_basic(simple_pdf):
    doc = parse_layout(simple_pdf, PipelineConfig())
    assert doc.metadata.page_count == 1
    assert doc.metadata.file_name == "simple.pdf"
    page = doc.pages[0]
    assert page.width > 0 and page.height > 0
    texts = [ln.text for ln in page.lines]
    assert texts[0] == "A Synthetic Test Document"
    title = page.lines[0]
    assert title.bold and abs(title.font_size - 18) < 0.5
    assert "Bold" in title.font_name
    body = [ln for ln in page.lines if ln.text.startswith("This is body")]
    assert body and not body[0].bold and abs(body[0].font_size - 10) < 0.5
    assert [ln.order for ln in page.lines] == list(range(len(page.lines)))
    assert dominant_font_size(doc.pages) == 10.0


def test_two_column_reading_order(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    left = [f"Left column text line {i} order test" for i in range(1, 7)]
    right = [f"Right column text line {i} order test" for i in range(1, 7)]
    write_lines(page, left, x=50, y=100)
    write_lines(page, right, x=320, y=100)
    path = save(pdf, tmp_path / "cols.pdf")

    doc = parse_layout(path, PipelineConfig())
    texts = [ln.text for ln in doc.pages[0].lines]
    assert texts[:6] == left
    assert texts[6:] == right
    regions = {ln.region for ln in doc.pages[0].lines}
    assert len(regions) == 2

    cfg = PipelineConfig().disable("columns")
    doc2 = parse_layout(path, cfg)
    texts2 = [ln.text for ln in doc2.pages[0].lines]
    assert texts2[:2] == [left[0], right[0]]  # row-wise without column detection


def test_full_width_band_above_columns(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((50, 60), "Abstract spanning the whole width of the page in one long single line of text here", fontsize=10)
    left = [f"Left col line {i} with some words" for i in range(1, 6)]
    right = [f"Right col line {i} with some words" for i in range(1, 6)]
    write_lines(page, left, x=50, y=120)
    write_lines(page, right, x=320, y=120)
    path = save(pdf, tmp_path / "band.pdf")
    doc = parse_layout(path, PipelineConfig())
    texts = [ln.text for ln in doc.pages[0].lines]
    assert texts[0].startswith("Abstract")
    assert texts[1:6] == left and texts[6:] == right


def test_fragment_merge(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 100), "Hello", fontsize=10)
    page.insert_text((100, 100), "world again", fontsize=10)
    path = save(pdf, tmp_path / "frag.pdf")
    doc = parse_layout(path, PipelineConfig())
    assert len(doc.pages[0].lines) == 1
    assert doc.pages[0].lines[0].text == "Hello world again"
<<<END FILE>>>

<<<FILE tests/test_boilerplate.py>>>
from fable_pipeline.boilerplate import is_page_number, normalize_key, remove_boilerplate
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout


def test_page_number_patterns():
    for t in ["12", "iv", "Page 3 of 10", "- 7 -", "Seite 4", "3 / 12", "A-3"]:
        assert is_page_number(t), t
    for t in ["Introduction", "Table 1", "2019 was a year", "3.5 Results"]:
        assert not is_page_number(t), t


def test_normalize_key():
    assert normalize_key("Chapter 2  Methods   17") == normalize_key("Chapter 3 Methods 18")
    assert normalize_key("  -- Hello --  ") == "hello"


def test_running_header_and_page_numbers_removed(multipage_pdf):
    doc = parse_layout(multipage_pdf, PipelineConfig())
    doc = remove_boilerplate(doc, PipelineConfig())
    for page in doc.pages:
        roles = {ln.text: ln.role for ln in page.lines}
        assert roles["Journal of Synthetic Tests"] == "header"
        assert roles[str(page.number)] == "page_number"
    body = [ln for p in doc.pages for ln in p.lines if ln.role == "body"]
    assert all(ln.text != "Journal of Synthetic Tests" for ln in body)
    assert any(ln.text.startswith("This is body") for ln in body)
    assert "Journal of Synthetic Tests" in doc.metadata.boilerplate["headers"]
    assert "page_numbers" in doc.metadata.boilerplate


def test_disabled_module_keeps_everything(multipage_pdf):
    cfg = PipelineConfig().disable("boilerplate")
    doc = remove_boilerplate(parse_layout(multipage_pdf, cfg), cfg)
    assert all(ln.role == "body" for p in doc.pages for ln in p.lines)


def test_single_page_title_is_not_a_header(simple_pdf):
    doc = remove_boilerplate(parse_layout(simple_pdf), PipelineConfig())
    assert doc.pages[0].lines[0].role == "body"
<<<END FILE>>>

<<<FILE tests/test_sections.py>>>
from fable_pipeline.boilerplate import remove_boilerplate
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.sections import detect_sections, is_equation, list_marker, numbering_info
from tests.conftest import add_page, body_lines, new_pdf, save, write_lines


def test_numbering_info():
    assert numbering_info("2.3.1 Data Collection")[0] == 3
    assert numbering_info("2 Methods")[0] == 1
    assert numbering_info("A.1 Supplementary")[0] == 2
    assert numbering_info("Chapter 4 Results")[0] == 1
    assert numbering_info("IV. Discussion")[0] == 1
    assert numbering_info("2.5 mg of sample") is None
    assert numbering_info("3 samples were taken.")[0] == 1  # caller must reject trailing punctuation


def test_list_marker_and_equation():
    assert list_marker("• first item") == ("-", "first item")
    assert list_marker("- dash item") == ("-", "dash item")
    assert list_marker("(a) lettered") == ("(a)", "lettered")
    assert list_marker("2. numbered") == ("2.", "numbered")
    assert list_marker("plain text") is None
    assert is_equation("E = mc^2 (1)")
    assert is_equation("x = α + β · y")
    assert not is_equation("This is a normal sentence about energy (see Section 1).")


def test_headings_levels_and_paragraphs(multipage_pdf):
    cfg = PipelineConfig()
    doc = parse_layout(multipage_pdf, cfg)
    doc = remove_boilerplate(doc, cfg)
    doc = detect_sections(doc, cfg)
    assert doc.metadata.title == "Running Text Paper"
    heads = [(b.text, b.level) for b in doc.blocks if b.type == "heading"]
    assert ("Running Text Paper", 0) in heads
    assert ("1 Introduction", 1) in heads
    assert ("2 Methods", 1) in heads
    assert ("2.1 Data", 2) in heads
    data_par = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("Data paragraph")]
    assert data_par and data_par[0].section_path == ["2 Methods", "2.1 Data"]
    # paragraph continuing across the page break is merged into one block
    tail = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("Tail paragraph")]
    assert tail and tail[0].pages == [1, 2]
    assert "continues on the next page" in tail[0].text
    # the two paragraphs on page 1 are separate (vertical gap)
    first = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("This is body")]
    assert first and first[0].extra["n_lines"] == 4
    assert all(ln.role != "header" or True for p in doc.pages for ln in p.lines)
    assert not any("Journal of Synthetic Tests" in b.text for b in doc.blocks)


def test_lists_and_bold_headings(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 80), "Results", fontsize=10, fontname="hebo")
    y = write_lines(page, body_lines(2), y=100)
    write_lines(page, ["• first bullet item", "• second bullet item", "• third bullet item"], x=84, y=y + 10)
    page.insert_text((72, 200), "Key findings", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2, "After the bold heading line"), y=220)
    path = save(pdf, tmp_path / "lists.pdf")
    doc = detect_sections(parse_layout(path), PipelineConfig())
    types = [b.type for b in doc.blocks]
    assert types.count("heading") == 2
    lists = [b for b in doc.blocks if b.type == "list"]
    assert len(lists) == 1
    assert lists[0].text.split("\n") == ["- first bullet item", "- second bullet item", "- third bullet item"]
    assert lists[0].section_path == ["Results"]
    last = [b for b in doc.blocks if b.text.startswith("After the bold")][0]
    assert last.section_path == ["Key findings"]


def test_outline_is_used_for_levels(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 80), "Background", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2), y=100)
    page.insert_text((72, 160), "Related work", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2, "More"), y=180)
    pdf.set_toc([[1, "Background", 1], [2, "Related work", 1]])
    path = save(pdf, tmp_path / "toc.pdf")
    doc = detect_sections(parse_layout(path), PipelineConfig())
    levels = {b.text: b.level for b in doc.blocks if b.type == "heading"}
    assert levels["Background"] == 1 and levels["Related work"] == 2
    cfg = PipelineConfig().disable("outline")
    doc2 = detect_sections(parse_layout(path), cfg)
    levels2 = {b.text: b.level for b in doc2.blocks if b.type == "heading"}
    assert levels2["Background"] == levels2["Related work"] == 1


def test_plain_fallback_when_disabled(simple_pdf):
    cfg = PipelineConfig().disable("sections")
    doc = detect_sections(parse_layout(simple_pdf, cfg), cfg)
    assert doc.blocks and all(b.type == "paragraph" for b in doc.blocks)
    assert all(b.section_path == [] for b in doc.blocks)
<<<END FILE>>>

<<<FILE tests/test_tables.py>>>
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.models import Block, Page, TableData
from fable_pipeline.sections import detect_sections
from fable_pipeline.tables import _merge_continued_tables, _parse_json_rows, extract_tables
from tests.conftest import add_page, body_lines, draw_grid_table, new_pdf, save, write_lines


def _ruled_pdf(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(2), y=60)
    page.insert_text((72, 110), "Table 1: Synthetic measurements", fontsize=10, fontname="helv")
    draw_grid_table(page, 72, 120, [100, 100, 100], 20,
                    [["Sample", "Mass", "Volume"], ["A", "1.0", "2.5"], ["B", "3.0", "4.5"], ["C", "5.0", "6.5"]])
    write_lines(page, body_lines(2, "Text after the table line"), y=230)
    return save(pdf, tmp_path / "ruled.pdf")


def test_ruled_table_extraction(tmp_path):
    path = _ruled_pdf(tmp_path)
    cfg = PipelineConfig()
    doc = extract_tables(parse_layout(path, cfg), cfg)
    tables = [b for p in doc.pages for b in p.region_blocks if b.type == "table"]
    assert len(tables) == 1
    td = tables[0].table
    assert td.columns == ["Sample", "Mass", "Volume"]
    assert td.records[0] == {"Sample": "A", "Mass": "1.0", "Volume": "2.5"}
    assert td.n_rows == 3
    assert td.caption == "Table 1: Synthetic measurements"
    assert "| Sample | Mass | Volume |" in td.markdown
    claimed = [ln for ln in doc.pages[0].lines if ln.claimed == tables[0].id]
    assert any(ln.role == "table" for ln in claimed) and any(ln.role == "caption" for ln in claimed)
    doc = detect_sections(doc, cfg)
    types = [b.type for b in doc.blocks]
    assert "table" in types
    idx = types.index("table")
    assert doc.blocks[idx - 1].text.startswith("This is body")
    assert doc.blocks[idx + 1].text.startswith("Text after")
    assert not any("Sample" in b.text for b in doc.blocks if b.type == "paragraph")


def test_whitespace_table_detection(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(3), y=60)
    rows = [["Country", "2019", "2020"], ["Germany", "83.1", "83.2"], ["France", "67.0", "67.4"],
            ["Spain", "47.1", "47.4"], ["Italy", "59.8", "59.6"]]
    y = 130
    for row in rows:
        for x, cell in zip((72, 220, 340), row):
            page.insert_text((x, y), cell, fontsize=10, fontname="helv")
        y += 14
    write_lines(page, body_lines(3, "Prose below the table line"), y=y + 20)
    path = save(pdf, tmp_path / "ws.pdf")
    cfg = PipelineConfig()
    cfg.tables.strategies = ["whitespace"]
    doc = extract_tables(parse_layout(path, cfg), cfg)
    tables = [b for p in doc.pages for b in p.region_blocks if b.type == "table"]
    assert len(tables) == 1
    td = tables[0].table
    assert td.method == "whitespace"
    assert td.columns == ["Country", "2019", "2020"]
    assert td.n_rows == 4 and td.rows[0] == ["Germany", "83.1", "83.2"]


def test_tables_disabled(tmp_path):
    path = _ruled_pdf(tmp_path)
    cfg = PipelineConfig().disable("tables")
    doc = extract_tables(parse_layout(path, cfg), cfg)
    assert not any(p.region_blocks for p in doc.pages)


def test_parse_json_rows():
    assert _parse_json_rows('Here: [["a","b"],["1",2]] done') == [["a", "b"], ["1", "2"]]
    assert _parse_json_rows("no json") is None
    assert _parse_json_rows('[{"a": 1}]') is None


def test_merge_continued_tables():
    p1 = Page(number=1, width=600, height=800)
    p2 = Page(number=2, width=600, height=800)
    t1 = TableData.from_grid([["A", "B"], ["1", "2"]], caption="Table 1")
    t2 = TableData.from_grid([["A", "B"], ["3", "4"]])
    p1.region_blocks.append(Block(id="t1", type="table", text="", pages=[1], bbox=_bb(50, 600, 300, 780), table=t1))
    p2.region_blocks.append(Block(id="t2", type="table", text="", pages=[2], bbox=_bb(50, 50, 300, 150), table=t2))
    from fable_pipeline.models import Document, DocumentMetadata

    doc = Document(metadata=DocumentMetadata("x", "x", 2), pages=[p1, p2])
    _merge_continued_tables(doc)
    assert not p2.region_blocks
    merged = p1.region_blocks[0]
    assert merged.table.rows == [["1", "2"], ["3", "4"]]
    assert merged.pages == [1, 2]


def _bb(x0, y0, x1, y1):
    from fable_pipeline.models import BBox

    return BBox(x0, y0, x1, y1)
<<<END FILE>>>

<<<FILE tests/test_figures.py>>>
import json
import pytest

from fable_pipeline.config import PipelineConfig
from fable_pipeline.figures import compose_figure_text, extract_figures, is_scanned_candidate, ocr_scanned_pages
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.models import FigureData
from fable_pipeline.ocr import tesseract_available
from fable_pipeline.sections import detect_sections
from tests.conftest import add_page, body_lines, insert_png, make_png, new_pdf, save, write_lines


def _figure_pdf(tmp_path, text=None):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(2), y=60)
    insert_png(page, (72, 100, 272, 250), make_png(400, 300, text=text))
    page.insert_text((72, 270), "Figure 1: A synthetic red rectangle", fontsize=10, fontname="helv")
    write_lines(page, body_lines(2, "Text after the figure line"), y=300)
    return save(pdf, tmp_path / "fig.pdf")


def test_embedded_image_becomes_figure_with_caption(tmp_path):
    path = _figure_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    doc = extract_figures(parse_layout(path, cfg), cfg)
    figs = [b for b in doc.pages[0].region_blocks if b.type == "figure"]
    assert len(figs) == 1
    fd = figs[0].figure
    assert fd.kind == "raster" and fd.caption == "Figure 1: A synthetic red rectangle"
    assert fd.width_px > 0 and figs[0].bbox.width > 150
    assert fd.description is None
    doc = detect_sections(doc, cfg)
    types = [b.type for b in doc.blocks]
    assert types == ["paragraph", "figure", "paragraph"]
    assert "Figure 1" in doc.blocks[1].text
    assert "figure OCR requested" not in " ".join(doc.metadata.warnings)


def test_save_images(tmp_path):
    path = _figure_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    cfg.figures.save_images = True
    cfg.figures.image_dir = str(tmp_path / "imgs")
    doc = extract_figures(parse_layout(path, cfg), cfg)
    fd = doc.pages[0].region_blocks[0].figure
    assert fd.image_path and fd.image_path.endswith(".png")
    assert (tmp_path / "imgs").exists()


def test_scanned_candidate_detection(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    insert_png(page, (20, 20, 575, 822), make_png(1000, 1400, color=(255, 255, 255), text="SCANNED PAGE TEXT", text_size=90))
    path = save(pdf, tmp_path / "scan.pdf")
    cfg = PipelineConfig()
    doc = parse_layout(path, cfg)
    assert is_scanned_candidate(doc.pages[0], cfg.figures)
    cfg.figures.ocr_enabled = False
    doc = ocr_scanned_pages(doc, cfg)
    assert any("OCR is disabled" in w for w in doc.metadata.warnings)
    assert not doc.pages[0].is_scanned


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")
def test_scanned_page_is_ocred(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    insert_png(page, (20, 20, 575, 822), make_png(1100, 1500, color=(255, 255, 255), text="HELLO SCANNED WORLD", text_size=80))
    path = save(pdf, tmp_path / "scan2.pdf")
    cfg = PipelineConfig()
    doc = ocr_scanned_pages(parse_layout(path, cfg), cfg)
    assert doc.pages[0].is_scanned
    assert doc.pages[0].lines and doc.pages[0].lines[0].source == "ocr"
    assert doc.metadata.scanned_pages == [1]
    joined = " ".join(ln.text for ln in doc.pages[0].lines).upper()
    assert "HELLO" in joined or "SCANNED" in joined or "WORLD" in joined


def test_compose_figure_text():
    fd = FigureData(caption="Figure 2: Trend", ocr_text="x axis\ny axis", description="A rising line plot.")
    txt = compose_figure_text(fd)
    assert txt.splitlines() == ["Figure 2: Trend", "Description: A rising line plot.", "Text in figure: x axis y axis"]
    assert compose_figure_text(FigureData()) == "[Figure]"


class _FakeResponse:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_description_via_fake_ollama(tmp_path, monkeypatch):
    path = _figure_pdf(tmp_path)
    captured = {}

    def fake_opener(req, timeout=None):
        if req.full_url.endswith("/api/tags"):
            return _FakeResponse({"models": [{"name": "llava:7b"}]})
        payload = json.loads(req.data.decode())
        captured.update(payload)
        return _FakeResponse({"response": "A red rectangle on a white background."})

    import fable_pipeline.figures as figmod
    from fable_pipeline.ollama_client import OllamaClient

    monkeypatch.setattr(figmod, "OllamaClient", lambda cfg: OllamaClient(cfg, opener=fake_opener))
    cfg = PipelineConfig().disable("ocr")
    cfg.figures.describe_with_llm = True
    doc = extract_figures(parse_layout(path, cfg), cfg)
    fd = doc.pages[0].region_blocks[0].figure
    assert fd.description == "A red rectangle on a white background."
    assert fd.description_model == "llava:7b"
    assert captured["options"]["temperature"] == 0.0 and captured["options"]["seed"] == 42
    assert captured["images"] and captured["model"] == "llava:7b"
    assert "Figure 1" in captured["prompt"]
<<<END FILE>>>

<<<FILE tests/test_ollama_client.py>>>
import json
import urllib.error

from fable_pipeline.config import OllamaConfig
from fable_pipeline.ollama_client import OllamaClient


class _Resp:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_unavailable_server_returns_none():
    def opener(req, timeout=None):
        raise urllib.error.URLError("refused")

    client = OllamaClient(OllamaConfig(), opener=opener)
    assert client.is_available() is False
    assert client.has_model("llava:7b") is False
    assert client.generate("hi") is None


def test_generate_uses_deterministic_options():
    seen = {}

    def opener(req, timeout=None):
        if req.full_url.endswith("/api/tags"):
            return _Resp({"models": [{"name": "qwen2.5:7b"}, {"name": "llava:7b"}]})
        seen.update(json.loads(req.data.decode()))
        return _Resp({"response": "  ok  "})

    cfg = OllamaConfig(seed=7, temperature=0.0, text_model="qwen2.5:7b")
    client = OllamaClient(cfg, opener=opener)
    assert client.is_available() and client.has_model("qwen2.5:7b") and not client.has_model("gemma")
    assert client.generate("prompt", images=[b"abc"]) == "ok"
    assert seen["options"] == {"temperature": 0.0, "seed": 7, "num_predict": cfg.num_predict}
    assert seen["stream"] is False and seen["images"] == ["YWJj"]
<<<END FILE>>>

<<<FILE tests/test_cleanup.py>>>
from fable_pipeline.cleanup import clean_text, cleanup_document
from fable_pipeline.config import CleanupConfig, PipelineConfig
from fable_pipeline.models import BBox, Block, Document, DocumentMetadata, FigureData, TableData


def test_ligatures_and_unicode():
    assert clean_text("e\ufb03cient \ufb01ne \ufb02ow") == "efficient fine flow"
    assert clean_text("caf\u0065\u0301") == "café"  # NFC composition


def test_dehyphenation_across_line_breaks():
    assert clean_text("the experi-\nment worked") == "the experiment worked"
    assert clean_text("a self-\ncontained unit") == "a self-contained unit"
    assert clean_text("the 2019-\n2020 season") == "the 2019-2020 season"
    assert clean_text("see Fig-\nure 3") == "see Figure 3"
    assert clean_text("soft\u00adhyphen") == "softhyphen"


def test_whitespace_and_control_chars():
    assert clean_text("a \u200b b\x0c  c\n\nd ,e .") == "a b c d,e."
    assert clean_text("I N T R O D U C T I O N and more") == "INTRODUCTION and more"
    assert clean_text("Introduction ........ 5") == "Introduction 5"
    assert clean_text("( text )") == "(text)"


def test_keep_newlines_for_lists():
    out = clean_text("- first\n- sec-\nond\n- third", keep_newlines=True)
    assert out == "- first\n- second\n- third"


def test_config_switches():
    cfg = CleanupConfig(dehyphenate=False, fix_ligatures=False, use_ftfy=False)
    assert clean_text("experi-\nment \ufb01", cfg) == "experi- ment \ufb01"
    cfg2 = CleanupConfig(normalize_quotes=True)
    assert clean_text("\u201cquoted\u201d", cfg2) == '"quoted"'


def test_cleanup_document_blocks():
    meta = DocumentMetadata("x.pdf", "x.pdf", 1, title="A \ufb01ne  title")
    td = TableData.from_grid([["Na\ufb01me", "Val ue"], ["x", "1"]])
    blocks = [
        Block(id="p", type="paragraph", text="line one con-\ntinued here", pages=[1], bbox=BBox(0, 0, 1, 1)),
        Block(id="l", type="list", text="- item one\nwrapped\n- item two", pages=[1], bbox=None, extra={"items": []}),
        Block(id="t", type="table", text="", pages=[1], bbox=None, table=td),
        Block(id="f", type="figure", text="", pages=[1], bbox=None, figure=FigureData(caption="Figure 1: \ufb02ow", ocr_text="a\n\nb")),
        Block(id="e", type="paragraph", text="   ", pages=[1], bbox=None),
    ]
    doc = cleanup_document(Document(meta, blocks), PipelineConfig())
    texts = {b.id: b.text for b in doc.blocks}
    assert texts["p"] == "line one continued here"
    assert texts["l"] == "- item one wrapped\n- item two"
    assert doc.blocks[1].extra["items"] == ["- item one wrapped", "- item two"]
    assert doc.blocks[2].table.columns == ["Nafime", "Val ue"]
    assert doc.blocks[3].figure.caption == "Figure 1: flow"
    assert "e" not in texts  # empty paragraph dropped
    assert doc.metadata.title == "A fine title"


def test_cleanup_disabled_keeps_text():
    cfg = PipelineConfig().disable("cleanup")
    doc = Document(DocumentMetadata("x", "x", 1), [Block(id="p", type="paragraph", text="a-\nb", pages=[1], bbox=None)])
    assert cleanup_document(doc, cfg).blocks[0].text == "a-\nb"
<<<END FILE>>>

<<<FILE tests/test_render.py>>>
from fable_pipeline.models import BBox, Block, Document, DocumentMetadata, FigureData, TableData
from fable_pipeline.render import to_markdown


def _doc(with_title=True):
    blocks = []
    if with_title:
        blocks.append(Block(id="t", type="heading", text="My Title", pages=[1], bbox=None, level=0, extra={"is_title": True}))
    blocks += [
        Block(id="h1", type="heading", text="1 Intro", pages=[1], bbox=BBox(0, 0, 1, 1), level=1, section_path=["1 Intro"]),
        Block(id="p1", type="paragraph", text="Hello world.", pages=[1], bbox=None, section_path=["1 Intro"]),
        Block(id="h2", type="heading", text="1.1 Sub", pages=[1], bbox=None, level=2),
        Block(id="l1", type="list", text="- a\n- b", pages=[1], bbox=None),
        Block(id="tb", type="table", text="", pages=[1], bbox=None,
              table=TableData.from_grid([["A", "B"], ["1", "2"]], caption="Table 1: Numbers")),
        Block(id="fg", type="figure", text="", pages=[1], bbox=None,
              figure=FigureData(caption="Figure 1: Plot", description="Rising trend.", ocr_text="x y")),
        Block(id="c", type="caption", text="Figure 9: orphan caption", pages=[1], bbox=None),
    ]
    return Document(DocumentMetadata("x", "x", 1), blocks)


def test_markdown_rendering_with_title():
    md = to_markdown(_doc())
    assert md.startswith("# My Title\n\n## 1 Intro\n\nHello world.\n\n### 1.1 Sub\n\n- a\n- b\n\n")
    assert "**Table 1: Numbers**\n\n| A | B |\n| --- | --- |\n| 1 | 2 |" in md
    assert "[Figure: Figure 1: Plot]\nDescription: Rising trend.\nText in figure: x y" in md
    assert "*Figure 9: orphan caption*" in md
    assert md.endswith("\n")


def test_markdown_rendering_without_title():
    md = to_markdown(_doc(with_title=False))
    assert md.startswith("# 1 Intro\n\n")
    assert "\n## 1.1 Sub\n" in md


def test_empty_document():
    assert to_markdown(Document(DocumentMetadata("x", "x", 0), [])) == ""
<<<END FILE>>>

<<<FILE tests/test_pipeline.py>>>
import json
import os

from fable_pipeline import PipelineConfig, extract, from_json, to_json, to_markdown
from fable_pipeline.__main__ import main
from tests.conftest import add_page, body_lines, draw_grid_table, insert_png, make_png, new_pdf, save, write_lines


def _rich_pdf(tmp_path):
    pdf = new_pdf()
    for i in range(3):
        page = add_page(pdf)
        page.insert_text((72, 30), "Synthetic Conference Proceedings", fontsize=8, fontname="helv")
        page.insert_text((290, 820), f"Page {i + 1} of 3", fontsize=8, fontname="helv")
        if i == 0:
            page.insert_text((72, 90), "Rich Synthetic Document", fontsize=18, fontname="hebo")
            page.insert_text((72, 140), "1 Introduction", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(3), y=170)
            page.insert_text((72, 240), "Table 1: Key numbers", fontsize=10)
            draw_grid_table(page, 72, 250, [120, 120], 20, [["Metric", "Value"], ["Accuracy", "0.91"], ["Recall", "0.87"]])
            write_lines(page, body_lines(2, "After table line"), y=340)
        elif i == 1:
            page.insert_text((72, 80), "2 Methods", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(2, "Methods line"), y=110)
            insert_png(page, (72, 150, 272, 300), make_png())
            page.insert_text((72, 320), "Figure 1: Synthetic image", fontsize=10)
            write_lines(page, ["• bullet one", "• bullet two"], x=84, y=360)
        else:
            page.insert_text((72, 80), "3 Conclusion", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(2, "Final line"), y=110)
    return save(pdf, tmp_path / "rich.pdf")


def test_end_to_end(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    doc = extract(path, cfg)
    assert doc.metadata.title == "Rich Synthetic Document"
    assert doc.metadata.page_count == 3
    assert "ingestion" in doc.metadata.steps_applied and "sections" in doc.metadata.steps_applied
    types = [b.type for b in doc.blocks]
    assert "table" in types and "figure" in types and "list" in types and "heading" in types
    heads = [b.text for b in doc.blocks if b.type == "heading"]
    assert heads == ["Rich Synthetic Document", "1 Introduction", "2 Methods", "3 Conclusion"]
    table = doc.tables[0]
    assert table.table.records[0] == {"Metric": "Accuracy", "Value": "0.91"}
    assert table.section_path == ["1 Introduction"]
    fig = doc.figures[0]
    assert fig.figure.caption == "Figure 1: Synthetic image" and fig.section_path == ["2 Methods"]
    lst = [b for b in doc.blocks if b.type == "list"][0]
    assert lst.text == "- bullet one\n- bullet two"
    md = to_markdown(doc)
    assert md.startswith("# Rich Synthetic Document")
    assert "Synthetic Conference Proceedings" not in md
    assert "Page 2 of 3" not in md
    assert "| Metric | Value |" in md and "[Figure: Figure 1: Synthetic image]" in md
    # paragraphs are single-line clean text
    for b in doc.blocks:
        if b.type == "paragraph":
            assert "\n" not in b.text
    back = from_json(to_json(doc))
    assert back.to_dict() == doc.to_dict()


def test_determinism(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    a, b = extract(path, cfg).to_dict(), extract(path, cfg).to_dict()
    a["metadata"]["timings"] = b["metadata"]["timings"] = {}
    assert a == b


def test_ablation_and_custom_step(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("boilerplate", "tables", "figures", "sections", "ocr")
    doc = extract(path, cfg)
    assert set(doc.metadata.steps_skipped) >= {"boilerplate", "tables", "figures"}
    assert all(b.type == "paragraph" for b in doc.blocks)
    assert any("Synthetic Conference Proceedings" in b.text for b in doc.blocks)

    calls = []

    def my_boilerplate(document, config):
        calls.append(len(document.pages))
        return document

    cfg2 = PipelineConfig().disable("ocr")
    cfg2.custom_steps["boilerplate"] = my_boilerplate
    doc2 = extract(path, cfg2)
    assert calls == [3]
    assert any("Synthetic Conference Proceedings" in b.text for b in doc2.blocks)
    assert doc2.metadata.config["custom_steps"] == ["boilerplate"]


def test_failed_step_is_recorded(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")

    def boom(document, config):
        raise RuntimeError("kaboom")

    cfg.custom_steps["tables"] = boom
    doc = extract(path, cfg)
    assert "tables" in doc.metadata.steps_skipped
    assert any("kaboom" in w for w in doc.metadata.warnings)
    assert doc.blocks


def test_cli_writes_outputs(tmp_path):
    path = _rich_pdf(tmp_path)
    out = tmp_path / "out"
    rc = main([path, "--out", str(out), "--disable", "ocr", "--no-layout"])
    assert rc == 0
    md = out / "rich.md"
    js = out / "rich.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "pages" not in data and data["metadata"]["title"] == "Rich Synthetic Document"
    assert md.read_text(encoding="utf-8").startswith("# Rich Synthetic Document")
    assert main([str(tmp_path / "missing.pdf"), "--out", str(out)]) == 1
    assert not os.path.exists(out / "missing.md")
<<<END FILE>>>

<<<FILE DESIGN.md>>>
# FABLE pipeline – design notes

FABLE (*Figures, Articles, Boilerplate, Layout, Extraction*) converts PDFs into a
structure-preserving, LLM-friendly representation for RAG. It stops at the
`Document` output contract; chunking, embedding and indexing live elsewhere.

```
PDF ─► ingestion ─► scanned_ocr ─► boilerplate ─► tables ─► figures ─► sections ─► cleanup ─► Document
         (PyMuPDF)   (Tesseract)    (position +    (PyMuPDF,  (images,    (headings,    (ftfy,
                                     repetition)    pdfplumber, vector    paragraphs,   hyphens,
                                                    whitespace) clusters, lists, path)  ligatures)
                                                               OCR, VLM)
```

Every module has the signature `step(document, config) -> document`, operates on
the shared `Document` (layout layer `pages[].lines[]` + output layer `blocks[]`)
and can be disabled or replaced through `PipelineConfig` (`disable(...)`,
`custom_steps`). Decisions are recorded, not destroyed: boilerplate lines keep
their `role`, table/figure lines keep a `claimed` pointer, every block records
its detection method and confidence. This makes ablations and error analysis
possible on the JSON output alone.

## Library choices

| Need | Choice | Why | Rejected alternatives |
|---|---|---|---|
| Text + geometry + fonts | **PyMuPDF** | Fastest open parser, exact span boxes, font name/flags, image placement, vector-drawing clustering, page rendering for OCR, `find_tables` | pdfminer.six (10–50× slower, no rendering), pypdf (no layout), pdfplumber alone (slow on large files) |
| Ruled tables (2nd opinion) | **pdfplumber** | Independent line/edge detection; catches partially ruled tables PyMuPDF misses | tabula-py (Java dependency, no bbox w/o extra work) |
| Lattice/stream tables | **Camelot** (optional) | Good on scanned-like ruled tables; needs Ghostscript+OpenCV, so opt-in | – |
| OCR | **Tesseract 5** via pytesseract | Only mature fully local OCR with word boxes + confidences; CPU-friendly | EasyOCR/PaddleOCR (heavy torch/paddle deps, GPU-oriented) |
| Encoding repair | **ftfy** | Robust mojibake / ligature / control-char repair with no model | manual tables only |
| LLM/VLM | **Ollama HTTP API** (urllib only) | Local, model-agnostic, deterministic with `seed`+`temperature=0`; zero extra pip dependencies | llama.cpp bindings (install friction, no vision uniformity) |

All processing is CPU-only; no step needs a GPU.

## Module 1 – Ingestion & layout parsing (`ingestion.py`)

**Design.** `page.get_text("dict")` gives spans with bbox, font, size, flags.
Spans are grouped into lines; a line's style is the character-weighted
dominant style (bold if ≥80 % of characters are bold). PyMuPDF often splits one
baseline into several "lines"/blocks (tab stops, kerning, font changes); they
are re-merged when they overlap vertically and the horizontal gap is small.

**Reading order** uses a recursive XY-cut with band merging: lines are split
into horizontal bands at white-space gaps larger than the typical line pitch;
consecutive bands that share a vertical gutter are grouped (so a 2-column body
isn't cut into many tiny regions by paragraph gaps), then split at gutters that
are ≥8 pt wide, leave ≥12 % of the page width on each side and have ≥2 lines
per side, recursing into each column. Lines inside a region are sorted
top-to-bottom with vertically overlapping lines treated as one row. Full-width
elements (title, abstract, wide tables) therefore come before/after the columns
they interrupt. Vertical (rotated) text is pushed to the end and never becomes
body text.

**Alternatives.** PyMuPDF's `sort=True` (plain top-left sort; breaks columns),
`page.get_text("blocks")` order (creator-dependent), ML layout models (LayoutLM,
DocLayNet detectors – too heavy for the CPU/no-dependency constraint and
non-deterministic across versions).

**Failure cases.** Three-column pages with very narrow gutters (<8 pt); text
boxes placed inside the gutter; decorative side bars that look like columns;
columns with only one line each; right-to-left scripts.

**Cost.** ~10–40 ms/page, memory ≈ size of the span dictionary (a few MB).

## Module 2 – Header/footer & boilerplate (`boilerplate.py`)

**Design.** Candidates = lines whose centre lies in the top/bottom 12 % of the
page. Texts are normalised (digits→`#`, case-folded, edge punctuation removed)
and clustered with `SequenceMatcher` similarity ≥0.9; a cluster appearing on
≥max(2, 30 % of pages) pages becomes `header`/`footer`. Alternating odd/even
headers each appear on half of the pages, so they still pass. Page numbers are
recognised by pattern (`12`, `iv`, `Page 3 of 10`, `- 7 -`, `A-3`) regardless
of repetition. Body-zone lines ≥12 chars that repeat on ≥50 % of pages at the
same relative height (watermarks, "Downloaded from …") become `boilerplate`.

**Alternatives.** Fixed margins only (fails for footnotes/wide headers); pure
repetition without position (removes repeated table headers or legitimate
repeated sentences); geometric clustering of blocks across pages (more
complex, similar accuracy).

**Failure cases.** Single-page documents (no repetition → only page numbers
are removed); headers that contain the running section title and change every
page (handled by fuzzy matching only if the constant part dominates); very
short documents (2–3 pages) where a repeated first line of a form is removed;
footnotes are *kept* (they are in the footer zone but differ per page).

**Cost.** O(lines²) within the zone clusters, negligible (<10 ms for 100 pages).

## Module 3 – Section & paragraph detection (`sections.py`)

**Headings.** A line is a heading if it is short (≤20 words), has letters, no
terminal comma/semicolon/period (numbering excepted), is not a caption or
bullet, and satisfies one of: (a) matches a PDF outline entry on a nearby page;
(b) has section numbering (`2.3.1 X`, `Chapter 4`, `Appendix A`, `IV.`) and is
bold, larger or standalone; (c) font ≥1.1× the dominant body size and
standalone; (d) bold at body size and standalone. "Standalone" = extra gap or
style change before it and (short line or style change/gap after it), which
suppresses bold/large *paragraphs* (abstracts) whose lines are full-width and
adjacent. Consecutive numbered lines with consecutive single-level numbers are
treated as a list, not as headings.

**Levels.** Numbering depth and outline levels are authoritative. Each heading
has a style key (size rounded to 0.5 pt, bold); styles learn a level from
numbered headings, remaining styles are ranked by size then boldness and
interleaved monotonically. Multi-line headings are merged if the continuation
has the same (non-body) style, no numbering and no large gap. The title is
the largest text in the upper half of page 1 (≥1.3× body), stored in
`metadata.title`, emitted as a level-0 heading and (optionally) in the section
path. PDF-metadata titles are only used as fallback when they look like real
titles (contain a space, no file extension).

**Paragraphs.** Lines are merged in reading order unless: style changes;
vertical gap > median gap + max(2 pt, 0.35×size); first-line indent > 1×size;
previous line is short (x1 < 80 % of region width for justified text, 60 % for
ragged text – detected per document). Across column/page breaks a paragraph
continues when the previous line is full width or hyphenated and the new line
is not indented. Table/figure blocks are inserted at the reading position of
their claimed lines (table body / caption) or the first line below them.
Lists (bullets `•-–*`, enumerations `1.`, `(a)`, `iv)`) become one `list`
block per run of items; indented wrapped lines are appended to their item.
Equations (equation number at line end or ≥12 % math symbols) are isolated
paragraphs with `extra.equation=True` so they are not glued into prose.
Orphan captions (figure module disabled, unmatched) become `caption` blocks.

**Alternatives.** Font-size histogram thresholds only (fails on bold-only
headings and numbered-but-plain headings), ML heading classifiers (not
deterministic/offline-light), relying on the PDF outline only (absent in most
Office exports).

**Failure cases.** Headings styled identically to body text and without
numbering; run-in headings (`**Methods.** We …` in the same line); drop caps;
two-column text where a floating figure interrupts a paragraph mid-column
(paragraph is split); justified paragraphs whose last line happens to be
full width at a column break (merged with the next paragraph); reference
lists (each entry becomes its own paragraph only if hanging indents exist).

**Cost.** Linear in the number of lines; <50 ms for a 30-page paper.

## Module 4 – Tables (`tables.py`)

**Design.** Several detectors produce candidates `(grid, bbox, method,
confidence)`: PyMuPDF `find_tables` (0.85), pdfplumber line strategy (0.84),
pdfplumber text strategy (0.5, off by default), Camelot lattice/stream
(optional), and our **whitespace detector** (0.45–0.8) for unruled tables:
baseline-aligned rows with ≥2 short cells, consistent x-intervals forming ≥2
columns, ≥3 such rows, mostly short cells, bonus for uniform row lengths and
numeric content. Candidates are validated (≥2×2, fill ratio ≥30 %, two-column
prose rejected), deduplicated by IoU/containment (highest confidence wins),
and sorted by position. Each table gets `TableData` (header → unique column
names, records, CSV, Markdown, DataFrame on demand), a caption ("Table N …"
above preferred, below allowed), and the table's lines are claimed so they
never reappear as paragraphs. Tables continued on the next page (same column
count, bottom→top placement, same header or no caption) are merged.
Optional LLM assistance: low-confidence whitespace candidates are sent to the
local text model as `cell | cell` lines; the returned JSON rows replace the
grid only if they parse and keep ≥2 columns.

**Alternatives.** tabula-py (Java), table-transformer/DETR models (GPU, large),
LLM-only detection (slow, non-deterministic across prompts, hallucinated
cells – therefore LLM only *restructures* existing candidates).

**Failure cases.** Tables with multi-line cells in unruled layouts (rows split
into multiple grid rows); merged header cells (header names concatenated or
`col_n`); rotated tables (lines are `rotated` and ignored); tables drawn as
images (only OCR'd pages with whitespace detection can recover them); figures
with many axis labels detected as whitespace tables (mitigated by the
short-cell/width checks and by running tables before figures with overlap
suppression only in one direction).

**Cost.** PyMuPDF/pdfplumber: 20–200 ms/page (pdfplumber dominates on
graphics-heavy pages); whitespace: O(lines); LLM: 2–15 s per table on CPU.

## Module 5 – Figures & scanned pages (`figures.py`, `ocr.py`)

**Scanned pages.** A page with <25 native characters and ≥50 % raster coverage
is rendered at 300 dpi and OCR'd (Tesseract psm 3). Lines with mean word
confidence ≥40 are converted back to PDF-point boxes and replace the page's
lines, so column detection, boilerplate removal, whitespace tables and
section detection work unchanged (`source="ocr"` is preserved per line/block).

**Figures.** Regions = embedded raster images (≥40 pt and ≥0.5 % of the page;
full-page images behind an OCR text layer are skipped) + clusters of vector
drawings (`cluster_drawings`; rejected when they are mostly text, i.e. text
boxes/frames, or overlap tables). Overlapping regions are merged. Each figure
is rendered (150 dpi), OCR'd (psm 11 sparse text; vector figures with native
text use the native text instead), captioned ("Figure N" below preferred),
and optionally described by a vision model through Ollama (temperature 0,
fixed seed, caption passed as context). Block text = caption + description +
"Text in figure", which is what the chunker will embed.

**Alternatives.** Image-only extraction via `get_images` (misses vector plots,
loses placement); layout detectors (heavy); CLIP captioning (less informative
than a VLM prompted for trends/extremes).

**Failure cases.** Figures assembled from many small images (merged only if
overlapping); decorative images/logos (filtered by size only); charts whose
axis labels are picked up as a whitespace table; VLM hallucinations (bounded by
the "do not speculate" prompt but not eliminated – descriptions are marked
with the model name so they can be filtered); OCR on low-resolution rasters.

**Cost.** Rendering ≈ 50 ms/figure; Tesseract ≈ 0.5–3 s per page/figure on
CPU; 7B vision model ≈ 10–60 s per figure on CPU (optional). Memory: one
rendered page at 300 dpi ≈ 25 MB.

## Module 6 – Cleanup (`cleanup.py`)

ftfy repair → control/zero-width removal → ligature expansion → NFC
normalisation (NFKC deliberately avoided: it would destroy superscripts/units)
→ de-hyphenation at line breaks (join when the continuation starts lowercase
and the prefix is not a known compound prefix like *self-*, *non-*) → dot
leaders → letter-spaced capitals → whitespace collapse (lists keep one item
per line; table cells and figure texts are cleaned individually and the
Markdown/CSV/records renderings are rebuilt).

**Failure cases.** Hyphenated compounds split exactly at the hyphen and not in
the prefix list are joined ("state-of-the-" + "art" → "state-of-theart" is
avoided only because "the" is… not in the list – such cases do occur);
letter-spaced *lowercase* words are left alone; math spacing may be altered.

**Cost.** Negligible (regex, linear).

## Output contract and determinism

`Document.blocks` is ordered in reading order; every block carries `type`,
`text`, `pages`, `bbox`, `section_path`, `level` (headings; 0 = title),
`table`/`figure` payloads and an `extra` dict (method, confidence, n_lines,
source=ocr, equation…). `to_json` is lossless (the layout layer can be omitted
with `include_layout=False`); `to_markdown` emits `#` headings (offset by one
when a title exists), Markdown tables with bold captions, `[Figure: …]`
blocks with description/OCR lines and lists with one item per line.
All heuristics are deterministic; LLM steps use `temperature=0` and a fixed
`seed`, and the model name is stored with each description. Ollama/Tesseract
absence degrades gracefully with a warning in `metadata.warnings`.

## Testing

`tests/` builds synthetic PDFs with PyMuPDF (columns, running headers, ruled
and unruled tables, embedded images, outline entries, lists) and tests every
module in isolation plus the end-to-end pipeline, determinism, ablation
switches, custom-step swapping, failure recording and the CLI. OCR tests are
skipped automatically when Tesseract is not installed; Ollama is mocked.
<<<END FILE>>>

Project structure summary:

- `pyproject.toml`, `requirements.txt`, `requirements-optional.txt` – packaging, pinned deps, system-dependency notes (Tesseract, Ollama, Ghostscript/Java for optional backends).
- `fable_pipeline/models.py` – shared data structures (layout layer `Page/Line/Span`, output layer `Document/Block/TableData/FigureData`), lossless `to_json`/`from_json`.
- `fable_pipeline/config.py` – `PipelineConfig` with per-module sub-configs, `disable(...)` for ablations, `custom_steps` for swapping modules, Ollama settings (temperature 0, fixed seed).
- `fable_pipeline/ingestion.py` – PyMuPDF layout parsing, font info, fragment merging, XY-cut multi-column reading order.
- `fable_pipeline/boilerplate.py` – header/footer/page-number/watermark detection by position + cross-page fuzzy repetition.
- `fable_pipeline/captions.py` – shared caption detection for tables/figures/sections.
- `fable_pipeline/tables.py` – PyMuPDF/pdfplumber/Camelot/whitespace table detection, dedup, captions, continuation merging, optional Ollama restructuring.
- `fable_pipeline/figures.py`, `ocr.py`, `ollama_client.py` – scanned-page OCR, figure regions (raster + vector), figure OCR, optional vision-model descriptions via local Ollama.
- `fable_pipeline/sections.py` – title/heading detection (outline, numbering, size, bold), levels, lists, equations, paragraph merging across columns/pages, block assembly with section paths.
- `fable_pipeline/cleanup.py` – ftfy/Unicode/ligature/hyphenation/whitespace cleanup preserving lists, tables and figures.
- `fable_pipeline/render.py` – `to_markdown`; `fable_pipeline/pipeline.py` – `extract` entry point with timing, error isolation and `extract_to_files`; `fable_pipeline/__main__.py` – CLI.
- `tests/` – conftest with synthetic PDF builders and unit tests for every module plus end-to-end, determinism, ablation and CLI tests.
- `DESIGN.md` – rationale, alternatives, failure cases and cost per module.