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
