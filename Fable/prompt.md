You are building a PDF-to-text extraction pipeline tailored for Retrieval-Augmented Generation (RAG) with local LLMs. It will serve as an independent baseline in a bachelor thesis and will be compared against (a) a separately developed pipeline and (b) the default PDF loaders of LlamaIndex and LangChain. Design it as a strong, well-reasoned solution in its own right. Do not aim for minimal effort.

## Goal
Turn PDFs into an LLM-friendly representation that preserves document structure (sections, headings, paragraphs, tables, figures) and improves over common PDF extractors by:
- reliably detecting paragraphs and sections,
- extracting tables into useful structured formats,
- handling images via OCR and textual description,
- removing headers, footers, page numbers and other boilerplate,
- cleaning the text so it is well suited for downstream retrieval and question answering.

## Scope: extraction only, NOT chunking or indexing
Chunking, embedding and indexing are done by a separate, shared component that is identical for all compared pipelines. Do NOT implement chunking, embeddings, vector stores, retrieval or any RAG/QA logic. Your pipeline ends at a clean, structured document representation (see "Output contract").

## Technical constraints
- Everything runs locally. No cloud APIs, no API keys, no paid services.
- Use open-source Python libraries that give low-level access to PDF layout, text and tables, e.g. PyMuPDF, pdfplumber, Camelot, tabula-py, and an open-source OCR engine (e.g. Tesseract via pytesseract). State which libraries you choose and why.
- Any LLM or vision-language model use (e.g. for image descriptions or LLM-assisted table detection) must go through a local Ollama server, using small open models (roughly 4–8B parameters; vision-capable models where needed). Make the model name and Ollama URL configurable, and make every LLM-based step optional so the pipeline still works without Ollama.
- Python 3.11. CPU-only must work; GPU optional.
- Deterministic output for the same input and config. For LLM-based steps, use temperature 0 and a fixed seed.
- Target documents: Various PDF files, including full text PDF's, scanned documents, scientific papers, Office Reports, files contating Images and data tables.

## Required modular components
Implement each as an independent, importable, separately testable module with a clear interface. Each module takes and returns the shared document representation (or a well-defined part of it), so modules can be swapped or disabled individually.

1. **Ingestion & layout parsing**: load the PDF and extract text with positional and font information (bounding boxes, font size, font name/weight, page). Determine correct reading order, including multi-column layouts.
2. **Header/footer & boilerplate removal**: systematically detect and remove or normalize running headers, footers, page numbers and other repeated boilerplate (e.g. using cross-page repetition and position).
3. **Section & paragraph detection**: recognize headings and build a section hierarchy using font size, margins, styling, numbering patterns (e.g. "2.3.1") and table-of-contents/outline information where available. Merge lines into paragraphs correctly, including across page and column breaks.
4. **Table identification & extraction**: robustly detect tables (existing table libraries and/or optional LLM-assisted detection) and extract them into structured formats (pandas DataFrame, plus CSV/JSON and a Markdown rendering). Keep each table's caption and position in the document.
5. **Image/figure handling**: extract embedded images and figures; run OCR for embedded text; generate a textual description of what is shown (e.g. trends in plots, "highest value at …") via an optional local vision model. Keep captions and position in the document. Also detect fully scanned pages and route them through OCR.
6. **Text cleanup**: fix whitespace, broken lines, hyphenation across line breaks, ligatures, Unicode/encoding issues, while preserving meaningful structure (lists, equations as text, numbering).

## Output contract
Expose one entry point:

    extract(pdf_path: str, config: PipelineConfig) -> Document

[Optional: paste your own Document/Block dataclass definitions here so the output plugs directly into your shared chunker. Otherwise keep the default below.]

Default representation: a `Document` holds metadata (source file, page count, title if detectable) and an ordered list of blocks. Each block has a type (`heading`, `paragraph`, `list`, `table`, `figure`, `caption`), its text, page number(s), bounding box, and section path (e.g. `["3 Methods", "3.2 Data"]`). Headings carry a level. Table blocks also carry the structured data (records as JSON, plus a Markdown rendering). Figure blocks carry the OCR text and the generated description.

Also provide:
- `to_markdown(document) -> str`: a single clean Markdown rendering (headings as `#`, tables as Markdown tables, figures as a description block) that a generic chunker can consume directly.
- `to_json(document) -> str`: lossless JSON serialization of the full Document.

## Deliverables
1. Complete, runnable Python code for every module, the shared data structures, and the `extract` entry point.
2. A `PipelineConfig` that lets each module be enabled, disabled or swapped, for ablation studies.
3. A small CLI: `python -m fable_pipeline input.pdf --out out_dir` writing the Markdown and JSON outputs.
4. Unit tests for each module (small synthetic PDFs generated in the tests are fine), and a `requirements.txt` with pinned versions plus a note on system dependencies (e.g. Tesseract, Ghostscript for Camelot, Java for tabula-py).
5. DESIGN.md: for each module, the design rationale, the main alternatives considered, known failure cases, and rough runtime/memory cost.

Do not include an evaluation framework.

## Output format (important)
Return every file of the project using exactly this format, one block per file, with no other code fences around it:

<<<FILE relative/path/to/file.py>>>
...full file contents...
<<<END FILE>>>

Use relative paths only (e.g. `fable_pipeline/headers_footers.py`, `tests/test_tables.py`, `requirements.txt`, `DESIGN.md`). Write every file in full: no placeholders, no "rest unchanged", no omitted sections. After the last file, add a short plain-text summary of the project structure.