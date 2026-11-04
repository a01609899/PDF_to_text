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
