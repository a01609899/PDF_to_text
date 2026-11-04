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
