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
