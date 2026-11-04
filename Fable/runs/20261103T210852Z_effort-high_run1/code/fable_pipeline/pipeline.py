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
