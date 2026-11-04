<<<FILE pyproject.toml>>>
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "fable-pipeline"
version = "1.0.0"
description = "FABLE: a local, structure-preserving PDF-to-text extraction pipeline for RAG with local LLMs"
requires-python = ">=3.11"
dependencies = [
    "PyMuPDF==1.24.9",
    "pdfplumber==0.11.4",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "Pillow==10.4.0",
    "pytesseract==0.3.13",
    "ftfy==6.2.3",
]

[project.scripts]
fable-pipeline = "fable_pipeline.__main__:main"

[tool.setuptools.packages.find]
include = ["fable_pipeline*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
<<<END FILE>>>

<<<FILE requirements.txt>>>
# Core dependencies (pinned). Python 3.11, CPU-only.
PyMuPDF==1.24.9
pdfplumber==0.11.4
pandas==2.2.2
numpy==1.26.4
Pillow==10.4.0
pytesseract==0.3.13
ftfy==6.2.3
pytest==8.3.2

# ---------------------------------------------------------------------------
# System dependencies (not installable via pip):
#   * Tesseract OCR >= 5.x  (binary "tesseract" must be on PATH; install the
#     language packs you need, e.g. "eng", "deu").  Required for OCR of scanned
#     pages and of text inside figures. The pipeline degrades gracefully
#     (warning, no OCR) when Tesseract is missing.
#   * Ollama (https://ollama.com) running locally, only if LLM-based steps are
#     enabled (figure descriptions, LLM-assisted table structuring). Pull the
#     models you configure, e.g. `ollama pull llava:7b` and
#     `ollama pull qwen2.5:7b`.
#   * Optional table backends (see requirements-optional.txt):
#       - Camelot needs Ghostscript (lattice flavour) and OpenCV.
#       - tabula-py needs a Java runtime (not used by default).
# ---------------------------------------------------------------------------
<<<END FILE>>>

<<<FILE requirements-optional.txt>>>
# Optional extras. Install on top of requirements.txt if you want the
# "camelot" table strategy (config.tables.strategies += ["camelot"]).
# Camelot additionally needs Ghostscript (system package) for the lattice flavour.
camelot-py==0.11.0
opencv-python-headless==4.10.0.84
ghostscript==0.7
# tabula-py is NOT used by the default pipeline (needs a Java runtime); it is
# listed here only for completeness of the evaluated alternatives.
# tabula-py==2.9.3
<<<END FILE>>>

<<<FILE fable_pipeline/__init__.py>>>
"""FABLE pipeline: structure-preserving PDF extraction for local RAG.

Public API::

    from fable_pipeline import extract, PipelineConfig, to_markdown, to_json

    doc = extract("paper.pdf", PipelineConfig())
    md = to_markdown(doc)
    js = to_json(doc)
"""
from .config import (
    BoilerplateConfig,
    CleanupConfig,
    FigureConfig,
    IngestionConfig,
    OllamaConfig,
    PipelineConfig,
    SectionConfig,
    TableConfig,
)
from .models import (
    PIPELINE_VERSION,
    BBox,
    Block,
    Document,
    DocumentMetadata,
    FigureData,
    Line,
    Page,
    Span,
    TableData,
    from_json,
    to_json,
)
from .pipeline import extract, extract_to_files
from .render import to_markdown

__all__ = [
    "extract",
    "extract_to_files",
    "to_markdown",
    "to_json",
    "from_json",
    "PipelineConfig",
    "IngestionConfig",
    "BoilerplateConfig",
    "SectionConfig",
    "TableConfig",
    "FigureConfig",
    "CleanupConfig",
    "OllamaConfig",
    "Document",
    "DocumentMetadata",
    "Block",
    "BBox",
    "Line",
    "Page",
    "Span",
    "TableData",
    "FigureData",
    "PIPELINE_VERSION",
]
<<<END FILE>>>

<<<FILE fable_pipeline/models.py>>>
"""Shared data structures of the FABLE pipeline.

Two layers live here:

* **Layout layer** (:class:`Span`, :class:`Line`, :class:`Page`): produced by the
  ingestion module and *mutated* by the other modules (roles are assigned to
  lines, tables/figures are attached to pages). This is the intermediate
  representation every module operates on.
* **Output layer** (:class:`Block`, :class:`TableData`, :class:`FigureData`,
  :class:`Document`): the output contract consumed by the shared chunker.

All classes are plain dataclasses with explicit ``to_dict``/``from_dict`` so the
JSON serialisation is lossless and does not depend on pickling.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

PIPELINE_VERSION = "1.0.0"

BLOCK_TYPES = ("heading", "paragraph", "list", "table", "figure", "caption")

#: Roles a layout line can take. Only ``body`` (and ``title``) lines are turned
#: into text blocks; all others are either removed (header/footer/...) or
#: consumed by a table/figure block.
LINE_ROLES = (
    "body",
    "title",
    "header",
    "footer",
    "page_number",
    "boilerplate",
    "table",
    "figure",
    "caption",
    "rotated",
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class BBox:
    """Axis-aligned bounding box in PDF points, origin top-left (PyMuPDF style)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_any(cls, seq: Sequence[float]) -> "BBox":
        x0, y0, x1, y1 = (float(v) for v in seq)
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersection_area(self, other: "BBox") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_ratio(self, other: "BBox") -> float:
        """Fraction of *this* box's area that lies inside ``other``."""
        return self.intersection_area(other) / self.area if self.area > 0 else 0.0

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        return (self.x0 - tol) <= x <= (self.x1 + tol) and (self.y0 - tol) <= y <= (self.y1 + tol)

    def x_overlap(self, other: "BBox") -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def y_overlap(self, other: "BBox") -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def expanded(self, dx: float, dy: Optional[float] = None) -> "BBox":
        dy = dx if dy is None else dy
        return BBox(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def to_list(self) -> List[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]


def _bbox_to(b: Optional[BBox]) -> Optional[List[float]]:
    return None if b is None else b.to_list()


def _bbox_from(v: Any) -> Optional[BBox]:
    return None if v is None else BBox.from_any(v)


# --------------------------------------------------------------------------- #
# Layout layer
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    text: str
    bbox: BBox
    font: str = ""
    size: float = 0.0
    flags: int = 0
    color: int = 0
    bold: bool = False
    italic: bool = False
    superscript: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "font": self.font,
            "size": round(self.size, 2),
            "flags": self.flags,
            "color": self.color,
            "bold": self.bold,
            "italic": self.italic,
            "superscript": self.superscript,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        return cls(
            text=d["text"],
            bbox=BBox.from_any(d["bbox"]),
            font=d.get("font", ""),
            size=float(d.get("size", 0.0)),
            flags=int(d.get("flags", 0)),
            color=int(d.get("color", 0)),
            bold=bool(d.get("bold", False)),
            italic=bool(d.get("italic", False)),
            superscript=bool(d.get("superscript", False)),
        )


@dataclass
class Line:
    """One text line with dominant font information and a layout role."""

    text: str
    bbox: BBox
    page: int
    spans: List[Span] = field(default_factory=list)
    font_size: float = 0.0
    font_name: str = ""
    bold: bool = False
    italic: bool = False
    layout_block: int = -1
    order: int = 0
    region: int = 0
    region_bbox: Optional[BBox] = None
    role: str = "body"
    source: str = "native"  # native | ocr
    index: int = 0
    vertical: bool = False
    claimed: Optional[str] = None  # id of the table/figure/caption block that consumed this line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "page": self.page,
            "font_size": round(self.font_size, 2),
            "font_name": self.font_name,
            "bold": self.bold,
            "italic": self.italic,
            "layout_block": self.layout_block,
            "order": self.order,
            "region": self.region,
            "region_bbox": _bbox_to(self.region_bbox),
            "role": self.role,
            "source": self.source,
            "index": self.index,
            "vertical": self.vertical,
            "claimed": self.claimed,
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Line":
        return cls(
            text=d["text"],
            bbox=BBox.from_any(d["bbox"]),
            page=int(d["page"]),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            font_size=float(d.get("font_size", 0.0)),
            font_name=d.get("font_name", ""),
            bold=bool(d.get("bold", False)),
            italic=bool(d.get("italic", False)),
            layout_block=int(d.get("layout_block", -1)),
            order=int(d.get("order", 0)),
            region=int(d.get("region", 0)),
            region_bbox=_bbox_from(d.get("region_bbox")),
            role=d.get("role", "body"),
            source=d.get("source", "native"),
            index=int(d.get("index", 0)),
            vertical=bool(d.get("vertical", False)),
            claimed=d.get("claimed"),
        )


@dataclass
class ImageInfo:
    bbox: BBox
    xref: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"bbox": self.bbox.to_list(), "xref": self.xref, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageInfo":
        return cls(BBox.from_any(d["bbox"]), int(d.get("xref", 0)), int(d.get("width", 0)), int(d.get("height", 0)))


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    rotation: int = 0
    lines: List[Line] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)
    is_scanned: bool = False
    char_count: int = 0
    #: table/figure blocks anchored on this page (also referenced from Document.blocks
    #: after section assembly; not serialised separately to avoid duplication).
    region_blocks: List["Block"] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "rotation": self.rotation,
            "is_scanned": self.is_scanned,
            "char_count": self.char_count,
            "lines": [ln.to_dict() for ln in self.lines],
            "images": [im.to_dict() for im in self.images],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Page":
        return cls(
            number=int(d["number"]),
            width=float(d["width"]),
            height=float(d["height"]),
            rotation=int(d.get("rotation", 0)),
            lines=[Line.from_dict(x) for x in d.get("lines", [])],
            images=[ImageInfo.from_dict(x) for x in d.get("images", [])],
            is_scanned=bool(d.get("is_scanned", False)),
            char_count=int(d.get("char_count", 0)),
        )


# --------------------------------------------------------------------------- #
# Output layer
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")


def normalize_grid(grid: Optional[Iterable[Optional[Iterable[Any]]]]) -> List[List[str]]:
    """Turn any nested table-ish structure into a rectangular list of stripped strings.

    Empty rows and empty columns are dropped; ``None`` becomes ``""``.
    """
    rows: List[List[str]] = []
    for row in grid or []:
        if row is None:
            continue
        cells = ["" if c is None else _WS_RE.sub(" ", str(c)).strip() for c in row]
        rows.append(cells)
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _md_cell(s: str) -> str:
    s = s.replace("\n", " ").replace("|", "\\|").strip()
    return s if s else " "


def render_markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    n = len(columns)
    if n == 0:
        return ""
    out = ["| " + " | ".join(_md_cell(c) for c in columns) + " |", "|" + "|".join([" --- "] * n) + "|"]
    for row in rows:
        cells = list(row)[:n] + [""] * max(0, n - len(row))
        out.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    return "\n".join(out)


def _unique_columns(header: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, name in enumerate(header):
        name = (name or "").strip() or f"col_{i + 1}"
        base = name
        k = seen.get(base, 0)
        if k:
            name = f"{base} ({k + 1})"
        seen[base] = k + 1
        out.append(name)
    return out


def _to_csv(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow(list(row))
    return buf.getvalue()


@dataclass
class TableData:
    columns: List[str]
    rows: List[List[str]]
    records: List[Dict[str, str]]
    markdown: str
    csv: str
    caption: Optional[str] = None
    method: str = ""
    confidence: float = 0.0
    n_rows: int = 0
    n_cols: int = 0

    @classmethod
    def from_grid(
        cls,
        grid: Iterable[Optional[Iterable[Any]]],
        caption: Optional[str] = None,
        method: str = "",
        confidence: float = 0.0,
        header_row: bool = True,
    ) -> "TableData":
        g = normalize_grid(grid)
        if not g:
            return cls([], [], [], "", "", caption, method, confidence, 0, 0)
        n_cols = len(g[0])
        if header_row and len(g) >= 2 and any(g[0]):
            header, body = g[0], g[1:]
        else:
            header, body = [""] * n_cols, g
        columns = _unique_columns(header)
        records = [dict(zip(columns, row)) for row in body]
        return cls(
            columns=columns,
            rows=body,
            records=records,
            markdown=render_markdown_table(columns, body),
            csv=_to_csv(columns, body),
            caption=caption,
            method=method,
            confidence=confidence,
            n_rows=len(body),
            n_cols=len(columns),
        )

    def grid(self) -> List[List[str]]:
        return [list(self.columns)] + [list(r) for r in self.rows]

    def to_dataframe(self):
        """Return a pandas DataFrame (pandas is imported lazily)."""
        import pandas as pd  # noqa: WPS433

        return pd.DataFrame(self.rows, columns=self.columns)

    def to_json_records(self) -> str:
        return json.dumps(self.records, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "records": [dict(r) for r in self.records],
            "markdown": self.markdown,
            "csv": self.csv,
            "caption": self.caption,
            "method": self.method,
            "confidence": self.confidence,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableData":
        return cls(
            columns=list(d.get("columns", [])),
            rows=[list(r) for r in d.get("rows", [])],
            records=[dict(r) for r in d.get("records", [])],
            markdown=d.get("markdown", ""),
            csv=d.get("csv", ""),
            caption=d.get("caption"),
            method=d.get("method", ""),
            confidence=float(d.get("confidence", 0.0)),
            n_rows=int(d.get("n_rows", 0)),
            n_cols=int(d.get("n_cols", 0)),
        )


@dataclass
class FigureData:
    caption: Optional[str] = None
    ocr_text: str = ""
    native_text: str = ""
    description: Optional[str] = None
    description_model: Optional[str] = None
    image_path: Optional[str] = None
    width_px: int = 0
    height_px: int = 0
    kind: str = "raster"  # raster | vector | mixed
    xrefs: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "native_text": self.native_text,
            "description": self.description,
            "description_model": self.description_model,
            "image_path": self.image_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "kind": self.kind,
            "xrefs": list(self.xrefs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FigureData":
        return cls(
            caption=d.get("caption"),
            ocr_text=d.get("ocr_text", ""),
            native_text=d.get("native_text", ""),
            description=d.get("description"),
            description_model=d.get("description_model"),
            image_path=d.get("image_path"),
            width_px=int(d.get("width_px", 0)),
            height_px=int(d.get("height_px", 0)),
            kind=d.get("kind", "raster"),
            xrefs=[int(x) for x in d.get("xrefs", [])],
        )


@dataclass
class Block:
    id: str
    type: str
    text: str
    pages: List[int]
    bbox: Optional[BBox]
    section_path: List[str] = field(default_factory=list)
    level: Optional[int] = None
    table: Optional[TableData] = None
    figure: Optional[FigureData] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "pages": list(self.pages),
            "bbox": _bbox_to(self.bbox),
            "section_path": list(self.section_path),
            "level": self.level,
            "table": self.table.to_dict() if self.table else None,
            "figure": self.figure.to_dict() if self.figure else None,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Block":
        return cls(
            id=d["id"],
            type=d["type"],
            text=d.get("text", ""),
            pages=[int(p) for p in d.get("pages", [])],
            bbox=_bbox_from(d.get("bbox")),
            section_path=list(d.get("section_path", [])),
            level=d.get("level"),
            table=TableData.from_dict(d["table"]) if d.get("table") else None,
            figure=FigureData.from_dict(d["figure"]) if d.get("figure") else None,
            extra=dict(d.get("extra", {})),
        )


@dataclass
class DocumentMetadata:
    source_path: str
    file_name: str
    page_count: int
    title: Optional[str] = None
    pdf_title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None
    producer: Optional[str] = None
    creator: Optional[str] = None
    outline: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION
    config: Dict[str, Any] = field(default_factory=dict)
    steps_applied: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    scanned_pages: List[int] = field(default_factory=list)
    boilerplate: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "file_name": self.file_name,
            "page_count": self.page_count,
            "title": self.title,
            "pdf_title": self.pdf_title,
            "author": self.author,
            "subject": self.subject,
            "creation_date": self.creation_date,
            "producer": self.producer,
            "creator": self.creator,
            "outline": [dict(o) for o in self.outline],
            "pipeline_version": self.pipeline_version,
            "config": self.config,
            "steps_applied": list(self.steps_applied),
            "steps_skipped": list(self.steps_skipped),
            "timings": dict(self.timings),
            "warnings": list(self.warnings),
            "scanned_pages": list(self.scanned_pages),
            "boilerplate": {k: list(v) for k, v in self.boilerplate.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentMetadata":
        return cls(
            source_path=d.get("source_path", ""),
            file_name=d.get("file_name", ""),
            page_count=int(d.get("page_count", 0)),
            title=d.get("title"),
            pdf_title=d.get("pdf_title"),
            author=d.get("author"),
            subject=d.get("subject"),
            creation_date=d.get("creation_date"),
            producer=d.get("producer"),
            creator=d.get("creator"),
            outline=[dict(o) for o in d.get("outline", [])],
            pipeline_version=d.get("pipeline_version", PIPELINE_VERSION),
            config=dict(d.get("config", {})),
            steps_applied=list(d.get("steps_applied", [])),
            steps_skipped=list(d.get("steps_skipped", [])),
            timings=dict(d.get("timings", {})),
            warnings=list(d.get("warnings", [])),
            scanned_pages=[int(p) for p in d.get("scanned_pages", [])],
            boilerplate={k: list(v) for k, v in d.get("boilerplate", {}).items()},
        )


@dataclass
class Document:
    metadata: DocumentMetadata
    blocks: List[Block] = field(default_factory=list)
    pages: List[Page] = field(default_factory=list)

    def blocks_of_type(self, block_type: str) -> List[Block]:
        return [b for b in self.blocks if b.type == block_type]

    @property
    def headings(self) -> List[Block]:
        return self.blocks_of_type("heading")

    @property
    def tables(self) -> List[Block]:
        return self.blocks_of_type("table")

    @property
    def figures(self) -> List[Block]:
        return self.blocks_of_type("figure")

    def to_dict(self, include_layout: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "blocks": [b.to_dict() for b in self.blocks],
        }
        if include_layout:
            d["pages"] = [p.to_dict() for p in self.pages]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(
            metadata=DocumentMetadata.from_dict(d["metadata"]),
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            pages=[Page.from_dict(p) for p in d.get("pages", [])],
        )


def to_json(document: Document, include_layout: bool = True, indent: Optional[int] = 2) -> str:
    """Lossless JSON serialisation of the full :class:`Document`.

    ``include_layout=False`` omits the (large) per-line layout layer and keeps
    only metadata + blocks, which is all a chunker needs.
    """
    return json.dumps(document.to_dict(include_layout=include_layout), ensure_ascii=False, indent=indent)


def from_json(text: str) -> Document:
    return Document.from_dict(json.loads(text))
<<<END FILE>>>

<<<FILE fable_pipeline/config.py>>>
"""Configuration for the FABLE pipeline.

Every module has its own sub-config with an ``enabled`` flag (ingestion cannot
be disabled because every other module depends on its output).  Modules can be
*swapped* by registering a callable in :attr:`PipelineConfig.custom_steps`
under the step name; the callable receives ``(document, config)`` and must
return the (mutated) document.

Step names: ``ingestion`` (signature ``(pdf_path, config) -> Document``),
``scanned_ocr``, ``boilerplate``, ``tables``, ``figures``, ``sections``,
``cleanup``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    text_model: str = "qwen2.5:7b"
    vision_model: str = "llava:7b"
    timeout: float = 180.0
    seed: int = 42
    temperature: float = 0.0
    num_predict: int = 512
    keep_alive: str = "5m"


@dataclass
class IngestionConfig:
    detect_columns: bool = True
    min_gutter_width: float = 8.0
    min_column_width_ratio: float = 0.12
    min_lines_per_column: int = 2
    min_band_gap_factor: float = 0.5
    merge_line_gap_factor: float = 1.5
    merge_cross_block_gap_factor: float = 0.8


@dataclass
class BoilerplateConfig:
    enabled: bool = True
    header_zone: float = 0.12
    footer_zone: float = 0.12
    min_repeat_ratio: float = 0.3
    min_repeat_pages: int = 2
    fuzzy_threshold: float = 0.9
    detect_page_numbers: bool = True
    body_repeat_ratio: float = 0.5
    body_repeat_min_pages: int = 3
    body_min_chars: int = 12


@dataclass
class SectionConfig:
    enabled: bool = True
    use_outline: bool = True
    use_numbering: bool = True
    use_font_styles: bool = True
    heading_size_ratio: float = 1.1
    max_heading_words: int = 20
    max_levels: int = 4
    detect_title: bool = True
    title_size_ratio: float = 1.3
    title_in_section_path: bool = False
    detect_lists: bool = True
    detect_captions: bool = True
    detect_equations: bool = True
    paragraph_gap_factor: float = 0.35
    short_line_ratio: float = 0.80
    ragged_short_line_ratio: float = 0.60
    indent_factor: float = 1.0


@dataclass
class TableConfig:
    enabled: bool = True
    #: ordered list of detection strategies; available: pymupdf, pdfplumber,
    #: pdfplumber_text, camelot, whitespace
    strategies: List[str] = field(default_factory=lambda: ["pymupdf", "pdfplumber", "whitespace"])
    min_rows: int = 2
    min_cols: int = 2
    min_fill_ratio: float = 0.3
    dedupe_iou: float = 0.5
    header_row: bool = True
    caption_search_distance: float = 80.0
    merge_continued_tables: bool = True
    whitespace_min_rows: int = 3
    whitespace_min_short_cell_ratio: float = 0.5
    whitespace_max_cell_width_ratio: float = 0.4
    use_llm: bool = False
    llm_min_confidence: float = 0.6
    llm_max_rows: int = 60


@dataclass
class FigureConfig:
    enabled: bool = True
    detect_scanned_pages: bool = True
    scanned_max_chars: int = 25
    scanned_min_image_coverage: float = 0.5
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    ocr_min_confidence: float = 40.0
    page_ocr_psm: int = 3
    figure_ocr_psm: int = 11
    ocr_vector_with_native_text: bool = False
    extract_images: bool = True
    include_vector_graphics: bool = True
    min_figure_size: float = 40.0
    min_figure_area_ratio: float = 0.005
    max_text_lines_in_vector_figure: int = 30
    render_dpi: int = 150
    save_images: bool = False
    image_dir: Optional[str] = None
    caption_search_distance: float = 80.0
    describe_with_llm: bool = False
    description_prompt: str = (
        "You are describing a figure extracted from a document for a text-only search index. "
        "Describe in 2-5 sentences what the figure shows: the type of figure (line plot, bar chart, "
        "diagram, photo, screenshot, map, ...), axes and units if any, the main trend or comparison, and "
        "notable extremes (for example which category or x-value has the highest and lowest value). "
        "Transcribe short visible labels and legend entries. Do not speculate beyond what is visible."
    )
    max_description_tokens: int = 300


@dataclass
class CleanupConfig:
    enabled: bool = True
    use_ftfy: bool = True
    normalize_unicode: bool = True
    fix_ligatures: bool = True
    remove_control_chars: bool = True
    dehyphenate: bool = True
    fix_spaced_letters: bool = True
    collapse_whitespace: bool = True
    normalize_quotes: bool = False
    remove_dot_leaders: bool = True


@dataclass
class PipelineConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    boilerplate: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    sections: SectionConfig = field(default_factory=SectionConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    #: step name -> callable, used to swap a module implementation
    custom_steps: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    fail_fast: bool = False
    include_layout_in_json: bool = True
    log_level: str = "INFO"

    # ----------------------------------------------------------------- helpers
    def enable_llm(self, describe_figures: bool = True, refine_tables: bool = True) -> "PipelineConfig":
        self.figures.describe_with_llm = describe_figures
        self.tables.use_llm = refine_tables
        return self

    def disable(self, *names: str) -> "PipelineConfig":
        """Disable modules by name (for ablation): boilerplate, sections, tables,
        figures, cleanup, ocr, columns, scanned_ocr, lists, captions, outline."""
        for name in names:
            if name == "boilerplate":
                self.boilerplate.enabled = False
            elif name == "sections":
                self.sections.enabled = False
            elif name == "tables":
                self.tables.enabled = False
            elif name == "figures":
                self.figures.enabled = False
            elif name == "cleanup":
                self.cleanup.enabled = False
            elif name == "ocr":
                self.figures.ocr_enabled = False
            elif name == "scanned_ocr":
                self.figures.detect_scanned_pages = False
            elif name == "columns":
                self.ingestion.detect_columns = False
            elif name == "lists":
                self.sections.detect_lists = False
            elif name == "captions":
                self.sections.detect_captions = False
            elif name == "outline":
                self.sections.use_outline = False
            elif name == "llm":
                self.figures.describe_with_llm = False
                self.tables.use_llm = False
            else:
                raise ValueError(f"unknown module/feature name: {name}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "custom_steps":
                out[f.name] = sorted(value.keys())
            elif is_dataclass(value):
                out[f.name] = asdict(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        cfg = cls()
        cfg.update(data)
        return cfg

    def update(self, data: Dict[str, Any]) -> "PipelineConfig":
        """Recursively update from a (partial) dict, e.g. loaded from JSON."""
        for f in fields(self):
            if f.name not in data or f.name == "custom_steps":
                continue
            current = getattr(self, f.name)
            if is_dataclass(current):
                sub = data[f.name] or {}
                for sf in fields(current):
                    if sf.name in sub:
                        setattr(current, sf.name, sub[sf.name])
            else:
                setattr(self, f.name, data[f.name])
        return self

    @classmethod
    def from_json_file(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
<<<END FILE>>>

<<<FILE fable_pipeline/layout_utils.py>>>
"""Small geometry/statistics helpers shared by several modules."""
from __future__ import annotations

import statistics
from typing import Iterable, List, Optional, Sequence

from .models import BBox, Line, Page


def union_bbox(boxes: Iterable[BBox]) -> Optional[BBox]:
    out: Optional[BBox] = None
    for b in boxes:
        out = b if out is None else out.union(b)
    return out


def median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else default


def lines_in_bbox(page: Page, bbox: BBox, pad: float = 2.0, roles: Optional[Sequence[str]] = None) -> List[Line]:
    """Lines whose centre lies inside ``bbox`` (expanded by ``pad`` points)."""
    box = bbox.expanded(pad)
    out = []
    for ln in page.lines:
        if roles is not None and ln.role not in roles:
            continue
        if box.contains_point(ln.bbox.cx, ln.bbox.cy):
            out.append(ln)
    return out


def sorted_by_order(lines: Iterable[Line]) -> List[Line]:
    return sorted(lines, key=lambda ln: (ln.page, ln.order, ln.index))
<<<END FILE>>>

<<<FILE fable_pipeline/ingestion.py>>>
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
<<<END FILE>>>

<<<FILE fable_pipeline/boilerplate.py>>>
"""Module 2: header/footer, page number and boilerplate removal.

Strategy (position + cross-page repetition):

1. Lines whose vertical centre lies in the top/bottom zone of a page are
   *candidates*.
2. Candidate texts are normalised (lower-case, digits -> ``#``, punctuation at
   the ends stripped) and clustered with a fuzzy similarity so that
   "Chapter 2  Methods   17" and "Chapter 2  Methods   18" fall into one
   cluster.  Clusters that occur on at least ``max(min_repeat_pages,
   ceil(min_repeat_ratio * n_pages))`` pages are headers/footers.  Alternating
   (odd/even) running headers are handled naturally because each variant
   repeats on half of the pages.
3. Lines that *look* like page numbers ("12", "Page 3 of 10", "- 7 -", "iv")
   inside the zones are removed regardless of repetition.
4. Body-zone lines (>= ``body_min_chars``) repeated on many pages (e.g.
   watermarks, "Downloaded from ...") are marked as boilerplate.

Lines are not deleted; their ``role`` is set so the decision is auditable in
the JSON output and so ablation can simply skip this module.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .config import BoilerplateConfig, PipelineConfig
from .models import Document, Line, Page

log = logging.getLogger(__name__)

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^[ivxlcdm]{1,8}$"),
    re.compile(r"^(page|seite|pg\.?|p\.|s\.)\s*\d{1,4}(\s*(of|von|/|-)\s*\d{1,4})?$"),
    re.compile(r"^\d{1,4}\s*(of|von|/)\s*\d{1,4}$"),
    re.compile(r"^[-–—~]\s*\d{1,4}\s*[-–—~]$"),
    re.compile(r"^[a-z]\s?[-–]\s?\d{1,4}$"),
    re.compile(r"^\d{1,4}\s*[|•·]\s*$"),
    re.compile(r"^[|•·]\s*\d{1,4}$"),
]

_NORM_DIGITS = re.compile(r"\d+")
_NORM_WS = re.compile(r"\s+")
_NORM_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_key(text: str) -> str:
    key = text.lower()
    key = _NORM_DIGITS.sub("#", key)
    key = _NORM_WS.sub(" ", key).strip()
    key = _NORM_EDGE.sub("", key)
    return key


def is_page_number(text: str) -> bool:
    t = text.strip().lower()
    return any(p.match(t) for p in PAGE_NUMBER_PATTERNS)


def _zone(ln: Line, page: Page, cfg: BoilerplateConfig) -> Optional[str]:
    cy = ln.bbox.cy
    if cy <= page.height * cfg.header_zone:
        return "top"
    if cy >= page.height * (1.0 - cfg.footer_zone):
        return "bottom"
    return None


def _cluster(entries: List[Tuple[int, Line, str]], threshold: float) -> List[List[Tuple[int, Line, str]]]:
    """Group entries by (fuzzy) normalised text. Deterministic: insertion order."""
    clusters: List[List[Tuple[int, Line, str]]] = []
    reps: List[str] = []
    for entry in entries:
        key = entry[2]
        placed = False
        for i, rep in enumerate(reps):
            if key == rep or (len(key) >= 6 and len(rep) >= 6 and SequenceMatcher(None, key, rep).ratio() >= threshold):
                clusters[i].append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])
            reps.append(key)
    return clusters


def remove_boilerplate(doc: Document, config: Optional[PipelineConfig] = None) -> Document:
    config = config or PipelineConfig()
    cfg = config.boilerplate
    if not cfg.enabled:
        return doc
    n_pages = len(doc.pages)
    zone_entries: Dict[str, List[Tuple[int, Line, str]]] = {"top": [], "bottom": []}
    removed: Dict[str, List[str]] = defaultdict(list)

    for page in doc.pages:
        for ln in page.lines:
            if ln.role != "body":
                continue
            zone = _zone(ln, page, cfg)
            if zone is None:
                continue
            if cfg.detect_page_numbers and is_page_number(ln.text):
                ln.role = "page_number"
                removed["page_numbers"].append(ln.text.strip())
                continue
            zone_entries[zone].append((page.number, ln, normalize_key(ln.text)))

    threshold = max(cfg.min_repeat_pages, math.ceil(cfg.min_repeat_ratio * n_pages))
    if n_pages >= 2:
        for zone, entries in zone_entries.items():
            role = "header" if zone == "top" else "footer"
            for cluster in _cluster(entries, cfg.fuzzy_threshold):
                pages_hit = {p for p, _, _ in cluster}
                if len(pages_hit) >= threshold and cluster[0][2]:
                    for _, ln, _ in cluster:
                        ln.role = role
                    removed[role + "s"].append(cluster[0][1].text.strip())

        # body-zone repetition (watermarks, "downloaded from", confidentiality notes)
        body_threshold = max(cfg.body_repeat_min_pages, math.ceil(cfg.body_repeat_ratio * n_pages))
        body_entries: List[Tuple[int, Line, str]] = []
        for page in doc.pages:
            for ln in page.lines:
                if ln.role != "body":
                    continue
                key = normalize_key(ln.text)
                if len(key) >= cfg.body_min_chars:
                    body_entries.append((page.number, ln, key))
        for cluster in _cluster(body_entries, max(cfg.fuzzy_threshold, 0.95)):
            pages_hit = {p for p, _, _ in cluster}
            if len(pages_hit) >= body_threshold and len(cluster) >= body_threshold:
                # a repeated *sentence* could be legitimate; require it to be at a similar y position
                ys = sorted(ln.bbox.cy / max(1.0, doc.pages[p - 1].height) for p, ln, _ in cluster)
                spread = ys[-1] - ys[0]
                if spread <= 0.08:
                    for _, ln, _ in cluster:
                        ln.role = "boilerplate"
                    removed["boilerplate"].append(cluster[0][1].text.strip())

    doc.metadata.boilerplate = {k: sorted(set(v)) for k, v in removed.items()}
    total = sum(1 for p in doc.pages for ln in p.lines if ln.role in ("header", "footer", "page_number", "boilerplate"))
    log.info("boilerplate: marked %d lines on %d pages", total, n_pages)
    return doc
<<<END FILE>>>

<<<FILE fable_pipeline/captions.py>>>
"""Caption detection shared by the table, figure and section modules."""
from __future__ import annotations

import re
from typing import List, Optional, Pattern

from .models import BBox, Line, Page

TABLE_CAPTION_RE: Pattern[str] = re.compile(
    r"^(?P<label>(?:tabelle|table|tab\.|tabla|tableau)\s*(?P<num>[A-Z]?\d+(?:[.\-]\d+)*[a-z]?|[IVX]+))(?P<punct>\s*[.:\-–—|])?",
    re.I,
)
FIGURE_CAPTION_RE: Pattern[str] = re.compile(
    r"^(?P<label>(?:abbildung|abb\.|figure|fig\.|fig|figura|chart|plot|graph|scheme|image|photo)\s*"
    r"(?P<num>[A-Z]?\d+(?:[.\-]\d+)*[a-z]?|[IVX]+))(?P<punct>\s*[.:\-–—|])?",
    re.I,
)
ANY_CAPTION_RE: Pattern[str] = re.compile(
    r"^(?P<label>(?:tabelle|table|tab\.|tabla|tableau|abbildung|abb\.|figure|fig\.|fig|figura|chart|plot|graph|scheme|image|photo)\s*"
    r"(?P<num>[A-Z]?\d+(?:[.\-]\d+)*[a-z]?|[IVX]+))(?P<punct>\s*[.:\-–—|])?",
    re.I,
)
_VERB_RE = re.compile(
    r"^(shows?|presents?|lists?|summari[sz]es?|gives?|reports?|illustrates?|depicts?|displays?|contains?|"
    r"compares?|provides?|demonstrates?|indicates?|describes?|plots?|zeigt|enth[aä]lt|stellt|fasst)\b",
    re.I,
)


def looks_like_caption(line: Line, body_size: float, pattern: Pattern[str] = ANY_CAPTION_RE) -> bool:
    """True if a line starts a caption (as opposed to an in-text reference like 'Table 1 shows')."""
    text = line.text.strip()
    m = pattern.match(text)
    if not m:
        return False
    if m.group("punct"):
        return True
    rest = text[m.end():].strip()
    if not rest:
        return True
    if line.bold or abs(line.font_size - body_size) > 0.6:
        return True
    if rest[0].isupper() and not _VERB_RE.match(rest):
        return True
    return False


def group_caption_lines(page: Page, start: Line, body_size: float, max_lines: int = 8) -> List[Line]:
    """Extend a caption start line with its continuation lines."""
    ordered = sorted(page.lines, key=lambda l: l.order)
    group = [start]
    try:
        pos = ordered.index(start)
    except ValueError:
        return group
    region_x1 = start.region_bbox.x1 if start.region_bbox else start.bbox.x1
    region_w = (start.region_bbox.width if start.region_bbox else start.bbox.width) or 1.0
    # a short first line is a complete one-line caption
    if start.bbox.x1 < region_x1 - 0.15 * region_w:
        return group
    last = start
    for ln in ordered[pos + 1:]:
        if ln.role != "body" or ln.claimed:
            break
        if ln.region != start.region or ln.page != start.page:
            break
        gap = ln.bbox.y0 - last.bbox.y1
        if gap > 0.9 * max(last.font_size, 1.0) or gap < -2.0:
            break
        if abs(ln.font_size - start.font_size) > 0.6:
            break
        if looks_like_caption(ln, body_size):
            break
        group.append(ln)
        last = ln
        if len(group) >= max_lines:
            break
        if ln.bbox.x1 < region_x1 - 0.15 * region_w:  # short line ends the caption
            break
    return group


def caption_text(lines: List[Line]) -> str:
    return " ".join(ln.text.strip() for ln in lines).strip()


def claim(lines: List[Line], block_id: str) -> None:
    for ln in lines:
        ln.role = "caption"
        ln.claimed = block_id


def find_caption(
    page: Page,
    bbox: BBox,
    pattern: Pattern[str],
    body_size: float,
    max_distance: float = 80.0,
    prefer: str = "above",
) -> Optional[List[Line]]:
    """Find the nearest caption line above/below ``bbox`` and return its line group."""
    candidates = []
    for ln in page.lines:
        if ln.role != "body" or ln.claimed or ln.vertical:
            continue
        if not looks_like_caption(ln, body_size, pattern):
            continue
        if ln.bbox.x_overlap(bbox) <= 0 and not bbox.contains_point(ln.bbox.cx, bbox.cy):
            continue
        if ln.bbox.y1 <= bbox.y0 + 2:
            dist, pos = bbox.y0 - ln.bbox.y1, "above"
        elif ln.bbox.y0 >= bbox.y1 - 2:
            dist, pos = ln.bbox.y0 - bbox.y1, "below"
        else:
            continue  # inside the region
        if dist <= max_distance:
            penalty = 0.0 if pos == prefer else max_distance * 0.5
            candidates.append((dist + penalty, ln.order, ln))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    best = candidates[0][2]
    return group_caption_lines(page, best, body_size)
<<<END FILE>>>

<<<FILE fable_pipeline/ollama_client.py>>>
"""Minimal, dependency-free client for a local Ollama server.

Only ``/api/tags`` (availability check) and ``/api/generate`` are used. All
calls are made with ``temperature`` and ``seed`` from the config so results are
deterministic for the same model/prompt/image. Any failure returns ``None``
and is logged; callers must treat LLM output as optional.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .config import OllamaConfig

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, opener: Optional[Callable[..., Any]] = None):
        self.cfg = cfg
        self._opener = opener or urllib.request.urlopen
        self._available: Optional[bool] = None
        self._models: List[str] = []

    # ------------------------------------------------------------------ http
    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        url = self.cfg.url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with self._opener(req, timeout=timeout or self.cfg.timeout) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8")) if body else {}

    # ------------------------------------------------------------------- api
    def is_available(self) -> bool:
        if self._available is None:
            try:
                resp = self._request("/api/tags", timeout=5.0)
                self._models = sorted(m.get("name", "") for m in resp.get("models", []))
                self._available = True
            except Exception as exc:  # noqa: BLE001
                log.warning("Ollama not reachable at %s (%s); LLM steps are skipped", self.cfg.url, exc)
                self._available = False
        return bool(self._available)

    def has_model(self, name: str) -> bool:
        if not self.is_available():
            return False
        base = name.split(":")[0]
        return any(m == name or m.split(":")[0] == base for m in self._models) or not self._models

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        images: Optional[List[bytes]] = None,
        num_predict: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Optional[str]:
        model = model or self.cfg.text_model
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                "num_predict": num_predict or self.cfg.num_predict,
            },
        }
        if images:
            payload["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        if system:
            payload["system"] = system
        try:
            resp = self._request("/api/generate", payload)
        except urllib.error.URLError as exc:
            log.warning("Ollama request failed (%s)", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama request failed (%s)", exc)
            return None
        text = resp.get("response")
        return text.strip() if isinstance(text, str) else None
<<<END FILE>>>

<<<FILE fable_pipeline/ocr.py>>>
"""Thin wrapper around Tesseract (via pytesseract) returning line boxes."""
from __future__ import annotations

import functools
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .models import BBox

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    bbox: BBox  # pixel coordinates of the OCR'd image
    conf: float


@functools.lru_cache(maxsize=1)
def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def pixmap_to_image(pix):
    """Convert a PyMuPDF Pixmap to a PIL image via PNG bytes (handles alpha/colorspaces)."""
    from PIL import Image  # noqa: WPS433

    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def ocr_lines(image, lang: str = "eng", psm: int = 3, min_conf: float = 40.0) -> List[OcrLine]:
    """Run Tesseract and return text lines with pixel bounding boxes and mean confidence."""
    import pytesseract  # noqa: WPS433

    data: Dict[str, Any] = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {int(psm)}", output_type=pytesseract.Output.DICT
    )
    n = len(data.get("text", []))
    groups: Dict[tuple, list] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        left, top = int(data["left"][i]), int(data["top"][i])
        width, height = int(data["width"][i]), int(data["height"][i])
        groups.setdefault(key, []).append((left, top, left + width, top + height, txt, conf))
    out: List[OcrLine] = []
    for key in sorted(groups):
        words = sorted(groups[key], key=lambda w: w[0])
        bbox = BBox(min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words))
        text = " ".join(w[4] for w in words)
        conf = sum(w[5] for w in words) / len(words)
        out.append(OcrLine(text=text, bbox=bbox, conf=conf))
    out.sort(key=lambda l: (l.bbox.y0, l.bbox.x0))
    return out


def ocr_text(image, lang: str = "eng", psm: int = 11, min_conf: float = 40.0) -> str:
    try:
        lines = ocr_lines(image, lang=lang, psm=psm, min_conf=min_conf)
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR failed: %s", exc)
        return ""
    return "\n".join(l.text for l in lines)
<<<END FILE>>>

<<<FILE fable_pipeline/tables.py>>>
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
<<<END FILE>>>

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
    if cfg.include_vector_graphics:
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
                       body_size: float, client: Optional[OllamaClient], vision_model: str, do_ocr: bool) -> Block:
    block_id = f"figure-p{page.number}-{idx}"
    inside = sorted_by_order([ln for ln in lines_in_bbox(page, bbox, pad=0.