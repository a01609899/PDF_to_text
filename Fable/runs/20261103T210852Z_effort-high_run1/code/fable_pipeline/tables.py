"""Module 4: table identification & extraction.

Detection strategies (configurable, ordered, deduplicated by IoU):

* ``pymupdf``       – ``Page.find_tables()`` (ruling lines / vector graphics).
* ``pdfplumber``    – ``Page.find_tables()`` with its line-based strategy;
                      complementary edge detection (handles thin/partial rules).
* ``pdfplumber_text`` – pdfplumber's text-alignment strategy (noisy; off by default).
* ``camelot``       – lattice+stream (optional dependency, needs Ghostscript).
* ``whitespace``    – our own layout-based detector for *unruled* tables: rows of
                      several short, baseline-aligned text cells whose x-intervals
                      form consistent columns.  Works on native and OCR'd pages.
* LLM assistance (optional, Ollama): low-confidence whitespace candidates are
  handed to a small local text model that returns the rows as JSON.

Every accepted table becomes a ``Block(type="table")`` with a
:class:`TableData` (records, CSV, Markdown), its caption and position.  The
lines inside the table region are marked ``role="table"`` so the section
module does not render them as paragraphs.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from .captions import TABLE_CAPTION_RE, caption_text, claim, find_caption
from .config import PipelineConfig, TableConfig
from .ingestion import dominant_font_size
from .layout_utils import lines_in_bbox, median, union_bbox
from .models import BBox, Block, Document, Line, Page, TableData, normalize_grid
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)

NUMERIC_RE = re.compile(r"^[\(\-+±~<>≤≥]?\s*\d[\d.,]*\s*(%|[a-zA-Zµ°]{0,4})?\)?$")

LLM_TABLE_PROMPT = (
    "The following lines were extracted from a PDF page and look like a table. Cells of a row are separated by ' | '. "
    "Reconstruct the table. Return ONLY a JSON array of rows, each row a JSON array of cell strings. "
    "The first row must be the header row (use short descriptive names if the header is missing). "
    "Keep all values exactly as written, do not add commentary.\n\n"
)


@dataclass
class _Candidate:
    grid: List[List[str]]
    bbox: BBox
    method: str
    confidence: float
    text_lines: List[str] = field(default_factory=list)


def compose_table_text(td: TableData) -> str:
    parts = []
    if td.caption:
        parts.append(td.caption)
    parts.append(td.markdown)
    return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_tables(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    config = config or PipelineConfig()
    cfg = config.tables
    if not cfg.enabled:
        return doc
    strategies = [s.lower() for s in cfg.strategies]
    body_size = dominant_font_size(doc.pages)
    path = doc.metadata.source_path

    pdf = fitz.open(path) if "pymupdf" in strategies else None
    plumber = None
    if "pdfplumber" in strategies or "pdfplumber_text" in strategies:
        try:
            import pdfplumber  # noqa: WPS433

            plumber = pdfplumber.open(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("pdfplumber unavailable (%s); strategy skipped", exc)
            doc.metadata.warnings.append(f"pdfplumber unavailable: {exc}")
    camelot_mod = None
    if "camelot" in strategies:
        try:
            import camelot  # noqa: WPS433

            camelot_mod = camelot
        except Exception as exc:  # noqa: BLE001
            log.warning("camelot unavailable (%s); strategy skipped", exc)
            doc.metadata.warnings.append(f"camelot unavailable: {exc}")
    client: Optional[OllamaClient] = None
    if cfg.use_llm:
        client = OllamaClient(config.ollama)
        if not client.is_available():
            doc.metadata.warnings.append("tables.use_llm requested but Ollama is not reachable")
            client = None

    counter = 0
    try:
        for page in doc.pages:
            cands: List[_Candidate] = []
            if not page.is_scanned:
                if pdf is not None:
                    cands.extend(_pymupdf_candidates(pdf.load_page(page.number - 1)))
                if plumber is not None and page.number - 1 < len(plumber.pages):
                    pl_page = plumber.pages[page.number - 1]
                    if "pdfplumber" in strategies:
                        cands.extend(_pdfplumber_candidates(pl_page, text_strategy=False))
                    if "pdfplumber_text" in strategies:
                        cands.extend(_pdfplumber_candidates(pl_page, text_strategy=True))
                if camelot_mod is not None:
                    cands.extend(_camelot_candidates(camelot_mod, path, page))
            if "whitespace" in strategies:
                cands.extend(_whitespace_candidates(page, cfg))
            cands = [c for c in cands if _plausible(c, cfg)]
            cands = _dedupe(cands, cfg)
            cands.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
            for cand in cands:
                if client is not None and cand.method.startswith("whitespace") and cand.confidence < cfg.llm_min_confidence:
                    _llm_refine(cand, client, config)
                counter += 1
                block = _make_table_block(page, cand, counter, cfg, body_size)
                page.region_blocks.append(block)
    finally:
        if pdf is not None:
            pdf.close()
        if plumber is not None:
            plumber.close()
    if cfg.merge_continued_tables:
        _merge_continued_tables(doc)
    log.info("tables: %d table(s) found", sum(1 for p in doc.pages for b in p.region_blocks if b.type == "table"))
    return doc


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def _pymupdf_candidates(fitz_page) -> List[_Candidate]:
    try:
        finder = fitz_page.find_tables()
    except Exception as exc:  # noqa: BLE001
        log.debug("pymupdf find_tables failed: %s", exc)
        return []
    out = []
    for t in getattr(finder, "tables", []):
        try:
            grid = t.extract()
            bbox = BBox.from_any(t.bbox)
        except Exception:  # noqa: BLE001
            continue
        out.append(_Candidate(grid=grid, bbox=bbox, method="pymupdf", confidence=0.85))
    return out


def _pdfplumber_candidates(pl_page, text_strategy: bool) -> List[_Candidate]:
    settings = {"vertical_strategy": "text", "horizontal_strategy": "text"} if text_strategy else None
    try:
        tables = pl_page.find_tables(settings) if settings else pl_page.find_tables()
    except Exception as exc:  # noqa: BLE001
        log.debug("pdfplumber find_tables failed: %s", exc)
        return []
    out = []
    for t in tables:
        try:
            grid = t.extract()
            bbox = BBox.from_any(t.bbox)
        except Exception:  # noqa: BLE001
            continue
        out.append(_Candidate(grid=grid, bbox=bbox, method="pdfplumber_text" if text_strategy else "pdfplumber",
                              confidence=0.5 if text_strategy else 0.84))
    return out


def _camelot_candidates(camelot_mod, path: str, page: Page) -> List[_Candidate]:
    out = []
    for flavor, base in (("lattice", 0.85), ("stream", 0.55)):
        try:
            tables = camelot_mod.read_pdf(path, pages=str(page.number), flavor=flavor)
        except Exception as exc:  # noqa: BLE001
            log.debug("camelot %s failed on page %d: %s", flavor, page.number, exc)
            continue
        for t in tables:
            try:
                x0, y0, x1, y1 = t._bbox  # PDF coords, origin bottom-left
                bbox = BBox.from_any((x0, page.height - y1, x1, page.height - y0))
                grid = t.df.values.tolist()
                acc = float(getattr(t, "accuracy", 80.0)) / 100.0
            except Exception:  # noqa: BLE001
                continue
            out.append(_Candidate(grid=grid, bbox=bbox, method=f"camelot_{flavor}", confidence=base * max(0.5, acc)))
        if out and flavor == "lattice":
            break  # stream only as fallback when lattice found nothing
    return out


def _group_rows(lines: List[Line]) -> List[List[Line]]:
    rows: List[List[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.bbox.y0, l.bbox.x0)):
        placed = False
        for row in rows[-3:]:
            ref = row[0]
            ov = ref.bbox.y_overlap(ln.bbox)
            if ov >= 0.5 * min(ref.bbox.height, ln.bbox.height):
                row.append(ln)
                placed = True
                break
        if not placed:
            rows.append([ln])
    for row in rows:
        row.sort(key=lambda l: l.bbox.x0)
    rows.sort(key=lambda r: min(l.bbox.y0 for l in r))
    return rows


def _column_intervals(rows: List[List[Line]]) -> List[List[float]]:
    intervals = sorted((c.bbox.x0, c.bbox.x1) for r in rows for c in r)
    cols: List[List[float]] = []
    for a, b in intervals:
        if cols and a <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], b)
        else:
            cols.append([a, b])
    return cols


def _rows_to_grid(rows: List[List[Line]], cols: List[List[float]]) -> List[List[str]]:
    grid = []
    for row in rows:
        cells = [""] * len(cols)
        for c in row:
            best, best_ov = 0, -1.0
            for j, (a, b) in enumerate(cols):
                ov = min(b, c.bbox.x1) - max(a, c.bbox.x0)
                if ov > best_ov:
                    best, best_ov = j, ov
            cells[best] = (cells[best] + " " + c.text.strip()).strip()
        grid.append(cells)
    return grid


def _whitespace_candidates(page: Page, cfg: TableConfig) -> List[_Candidate]:
    lines = [ln for ln in page.lines if ln.role == "body" and not ln.vertical]
    if len(lines) < cfg.whitespace_min_rows * 2:
        return []
    med_h = median([l.bbox.height for l in lines], 10.0) or 10.0
    rows = _group_rows(lines)
    sequences: List[List[List[Line]]] = []
    cur: List[List[Line]] = []
    for row in rows:
        multi = len(row) >= 2
        if cur:
            gap = min(l.bbox.y0 for l in row) - max(l.bbox.y1 for l in cur[-1])
            if gap > 2.5 * med_h:
                sequences.append(cur)
                cur = []
        if multi:
            cur.append(row)
        elif cur and len(row[0].text.split()) <= 4 and row[0].bbox.width < 0.5 * page.width:
            cur.append(row)  # spanning / single-cell row inside a table
        elif cur:
            sequences.append(cur)
            cur = []
    if cur:
        sequences.append(cur)

    out: List[_Candidate] = []
    for seq in sequences:
        while seq and len(seq[0]) < 2:
            seq.pop(0)
        while seq and len(seq[-1]) < 2:
            seq.pop()
        multi_rows = [r for r in seq if len(r) >= 2]
        if len(multi_rows) < cfg.whitespace_min_rows or len(multi_rows) < 0.6 * len(seq):
            continue
        cells = [c for r in multi_rows for c in r]
        n_words = [len(c.text.split()) for c in cells]
        short_ratio = sum(1 for n in n_words if n <= 3) / len(cells)
        long_ratio = sum(1 for n in n_words if n > 6) / len(cells)
        if short_ratio < cfg.whitespace_min_short_cell_ratio or long_ratio > 0.3:
            continue
        med_w = median([c.bbox.width for c in cells], 0.0)
        if med_w > cfg.whitespace_max_cell_width_ratio * page.width:
            continue
        cols = _column_intervals(multi_rows)
        if len(cols) < 2:
            continue
        grid = _rows_to_grid(seq, cols)
        bbox = union_bbox(c.bbox for r in seq for c in r)
        if bbox is None:
            continue
        counts = Counter(len(r) for r in multi_rows)
        uniform = counts.most_common(1)[0][1] / len(multi_rows)
        numeric = sum(1 for c in cells if NUMERIC_RE.match(c.text.strip())) / len(cells)
        conf = min(0.8, 0.45 + 0.2 * uniform + 0.15 * min(1.0, numeric * 2.0))
        out.append(
            _Candidate(
                grid=grid,
                bbox=bbox,
                method="whitespace",
                confidence=round(conf, 3),
                text_lines=[" | ".join(c.text.strip() for c in r) for r in seq],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Validation, deduplication, LLM refinement
# --------------------------------------------------------------------------- #
def _plausible(c: _Candidate, cfg: TableConfig) -> bool:
    grid = normalize_grid(c.grid)
    c.grid = grid
    if not grid:
        return False
    n_rows, n_cols = len(grid), len(grid[0])
    if n_rows < cfg.min_rows or n_cols < cfg.min_cols:
        return False
    cells = [cell for row in grid for cell in row]
    non_empty = [cell for cell in cells if cell]
    fill = len(non_empty) / max(1, len(cells))
    if fill < cfg.min_fill_ratio:
        return False
    avg_words = sum(len(cell.split()) for cell in non_empty) / max(1, len(non_empty))
    if n_cols <= 2 and avg_words > 20:  # two-column prose, not a table
        return False
    if c.bbox.area <= 0:
        return False
    c.confidence = round(min(0.99, c.confidence + 0.05 * fill), 3)
    return True


def _dedupe(cands: List[_Candidate], cfg: TableConfig) -> List[_Candidate]:
    ordered = sorted(cands, key=lambda c: (-c.confidence, c.bbox.y0, c.bbox.x0, c.method))
    accepted: List[_Candidate] = []
    for c in ordered:
        dup = False
        for a in accepted:
            if c.bbox.iou(a.bbox) >= cfg.dedupe_iou or c.bbox.overlap_ratio(a.bbox) >= 0.7 or a.bbox.overlap_ratio(c.bbox) >= 0.7:
                dup = True
                break
        if not dup:
            accepted.append(c)
    return accepted


def _parse_json_rows(text: Optional[str]) -> Optional[List[List[str]]]:
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    rows: List[List[str]] = []
    for row in data:
        if not isinstance(row, list):
            return None
        rows.append(["" if v is None else str(v) for v in row])
    return rows


def _llm_refine(cand: _Candidate, client: OllamaClient, config: PipelineConfig) -> None:
    lines = cand.text_lines[: config.tables.llm_max_rows]
    if not lines:
        return
    prompt = LLM_TABLE_PROMPT + "\n".join(lines)
    answer = client.generate(prompt, model=config.ollama.text_model, num_predict=max(512, 40 * len(lines)))
    rows = _parse_json_rows(answer)
    if rows and len(rows) >= 2:
        grid = normalize_grid(rows)
        if grid and len(grid[0]) >= 2:
            cand.grid = grid
            cand.method = "whitespace+llm"
            cand.confidence = max(cand.confidence, 0.7)


# --------------------------------------------------------------------------- #
# Block creation and continuation merging
# --------------------------------------------------------------------------- #
def _make_table_block(page: Page, cand: _Candidate, idx: int, cfg: TableConfig, body_size: float) -> Block:
    block_id = f"table-p{page.number}-{idx}"
    caption = None
    cap_lines = find_caption(page, cand.bbox, TABLE_CAPTION_RE, body_size, cfg.caption_search_distance, prefer="above")
    if cap_lines:
        caption = caption_text(cap_lines)
        claim(cap_lines, block_id)
    td = TableData.from_grid(cand.grid, caption=caption, method=cand.method, confidence=cand.confidence, header_row=cfg.header_row)
    for ln in lines_in_bbox(page, cand.bbox, pad=2.0):
        if ln.role == "body":
            ln.role = "table"
            ln.claimed = block_id
    return Block(
        id=block_id,
        type="table",
        text=compose_table_text(td),
        pages=[page.number],
        bbox=cand.bbox,
        table=td,
        extra={"method": cand.method, "confidence": cand.confidence, "source_id": block_id, "caption": caption},
    )


def _merge_continued_tables(doc: Document) -> None:
    """Merge a table at the bottom of page p with a same-shaped table at the top of p+1."""
    prev: Optional[tuple] = None  # (block, page_number, tail_bbox, page_height)
    for page in doc.pages:
        tables = sorted([b for b in page.region_blocks if b.type == "table" and b.bbox], key=lambda b: (b.bbox.y0, b.bbox.x0))
        if prev and tables:
            pb, pnum, tail, pheight = prev
            first = tables[0]
            same_shape = first.table is not None and pb.table is not None and first.table.n_cols == pb.table.n_cols
            continues = (
                pnum == page.number - 1
                and tail.y1 > 0.7 * pheight
                and first.bbox.y0 < 0.3 * page.height
                and same_shape
                and (first.table.columns == pb.table.columns or not first.table.caption)
            )
            if continues:
                if first.table.columns == pb.table.columns:
                    extra_rows = first.table.rows
                else:
                    extra_rows = [first.table.columns] + first.table.rows
                grid = [pb.table.columns] + pb.table.rows + extra_rows
                pb.table = TableData.from_grid(
                    grid, caption=pb.table.caption, method=pb.table.method + "+merged",
                    confidence=min(pb.table.confidence, first.table.confidence), header_row=True,
                )
                pb.text = compose_table_text(pb.table)
                pb.pages = sorted(set(pb.pages + [page.number]))
                pb.extra["continued_on_pages"] = pb.pages[1:]
                page.region_blocks.remove(first)
                for ln in page.lines:
                    if ln.claimed == first.id:
                        ln.claimed = pb.id
                tables = tables[1:]
                prev = (pb, page.number, first.bbox, page.height) if not tables else (tables[-1], page.number, tables[-1].bbox, page.height)
                continue
        prev = (tables[-1], page.number, tables[-1].bbox, page.height) if tables else None
