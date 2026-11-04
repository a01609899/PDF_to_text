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
