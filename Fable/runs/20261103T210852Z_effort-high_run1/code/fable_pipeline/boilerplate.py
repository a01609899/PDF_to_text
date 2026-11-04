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
