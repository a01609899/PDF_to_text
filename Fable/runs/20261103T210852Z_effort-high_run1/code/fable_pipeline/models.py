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
