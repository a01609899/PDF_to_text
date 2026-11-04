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
