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
