# FABLE pipeline – design notes

FABLE (*Figures, Articles, Boilerplate, Layout, Extraction*) converts PDFs into a
structure-preserving, LLM-friendly representation for RAG. It stops at the
`Document` output contract; chunking, embedding and indexing live elsewhere.

```
PDF ─► ingestion ─► scanned_ocr ─► boilerplate ─► tables ─► figures ─► sections ─► cleanup ─► Document
         (PyMuPDF)   (Tesseract)    (position +    (PyMuPDF,  (images,    (headings,    (ftfy,
                                     repetition)    pdfplumber, vector    paragraphs,   hyphens,
                                                    whitespace) clusters, lists, path)  ligatures)
                                                               OCR, VLM)
```

Every module has the signature `step(document, config) -> document`, operates on
the shared `Document` (layout layer `pages[].lines[]` + output layer `blocks[]`)
and can be disabled or replaced through `PipelineConfig` (`disable(...)`,
`custom_steps`). Decisions are recorded, not destroyed: boilerplate lines keep
their `role`, table/figure lines keep a `claimed` pointer, every block records
its detection method and confidence. This makes ablations and error analysis
possible on the JSON output alone.

## Library choices

| Need | Choice | Why | Rejected alternatives |
|---|---|---|---|
| Text + geometry + fonts | **PyMuPDF** | Fastest open parser, exact span boxes, font name/flags, image placement, vector-drawing clustering, page rendering for OCR, `find_tables` | pdfminer.six (10–50× slower, no rendering), pypdf (no layout), pdfplumber alone (slow on large files) |
| Ruled tables (2nd opinion) | **pdfplumber** | Independent line/edge detection; catches partially ruled tables PyMuPDF misses | tabula-py (Java dependency, no bbox w/o extra work) |
| Lattice/stream tables | **Camelot** (optional) | Good on scanned-like ruled tables; needs Ghostscript+OpenCV, so opt-in | – |
| OCR | **Tesseract 5** via pytesseract | Only mature fully local OCR with word boxes + confidences; CPU-friendly | EasyOCR/PaddleOCR (heavy torch/paddle deps, GPU-oriented) |
| Encoding repair | **ftfy** | Robust mojibake / ligature / control-char repair with no model | manual tables only |
| LLM/VLM | **Ollama HTTP API** (urllib only) | Local, model-agnostic, deterministic with `seed`+`temperature=0`; zero extra pip dependencies | llama.cpp bindings (install friction, no vision uniformity) |

All processing is CPU-only; no step needs a GPU.

## Module 1 – Ingestion & layout parsing (`ingestion.py`)

**Design.** `page.get_text("dict")` gives spans with bbox, font, size, flags.
Spans are grouped into lines; a line's style is the character-weighted
dominant style (bold if ≥80 % of characters are bold). PyMuPDF often splits one
baseline into several "lines"/blocks (tab stops, kerning, font changes); they
are re-merged when they overlap vertically and the horizontal gap is small.

**Reading order** uses a recursive XY-cut with band merging: lines are split
into horizontal bands at white-space gaps larger than the typical line pitch;
consecutive bands that share a vertical gutter are grouped (so a 2-column body
isn't cut into many tiny regions by paragraph gaps), then split at gutters that
are ≥8 pt wide, leave ≥12 % of the page width on each side and have ≥2 lines
per side, recursing into each column. Lines inside a region are sorted
top-to-bottom with vertically overlapping lines treated as one row. Full-width
elements (title, abstract, wide tables) therefore come before/after the columns
they interrupt. Vertical (rotated) text is pushed to the end and never becomes
body text.

**Alternatives.** PyMuPDF's `sort=True` (plain top-left sort; breaks columns),
`page.get_text("blocks")` order (creator-dependent), ML layout models (LayoutLM,
DocLayNet detectors – too heavy for the CPU/no-dependency constraint and
non-deterministic across versions).

**Failure cases.** Three-column pages with very narrow gutters (<8 pt); text
boxes placed inside the gutter; decorative side bars that look like columns;
columns with only one line each; right-to-left scripts.

**Cost.** ~10–40 ms/page, memory ≈ size of the span dictionary (a few MB).

## Module 2 – Header/footer & boilerplate (`boilerplate.py`)

**Design.** Candidates = lines whose centre lies in the top/bottom 12 % of the
page. Texts are normalised (digits→`#`, case-folded, edge punctuation removed)
and clustered with `SequenceMatcher` similarity ≥0.9; a cluster appearing on
≥max(2, 30 % of pages) pages becomes `header`/`footer`. Alternating odd/even
headers each appear on half of the pages, so they still pass. Page numbers are
recognised by pattern (`12`, `iv`, `Page 3 of 10`, `- 7 -`, `A-3`) regardless
of repetition. Body-zone lines ≥12 chars that repeat on ≥50 % of pages at the
same relative height (watermarks, "Downloaded from …") become `boilerplate`.

**Alternatives.** Fixed margins only (fails for footnotes/wide headers); pure
repetition without position (removes repeated table headers or legitimate
repeated sentences); geometric clustering of blocks across pages (more
complex, similar accuracy).

**Failure cases.** Single-page documents (no repetition → only page numbers
are removed); headers that contain the running section title and change every
page (handled by fuzzy matching only if the constant part dominates); very
short documents (2–3 pages) where a repeated first line of a form is removed;
footnotes are *kept* (they are in the footer zone but differ per page).

**Cost.** O(lines²) within the zone clusters, negligible (<10 ms for 100 pages).

## Module 3 – Section & paragraph detection (`sections.py`)

**Headings.** A line is a heading if it is short (≤20 words), has letters, no
terminal comma/semicolon/period (numbering excepted), is not a caption or
bullet, and satisfies one of: (a) matches a PDF outline entry on a nearby page;
(b) has section numbering (`2.3.1 X`, `Chapter 4`, `Appendix A`, `IV.`) and is
bold, larger or standalone; (c) font ≥1.1× the dominant body size and
standalone; (d) bold at body size and standalone. "Standalone" = extra gap or
style change before it and (short line or style change/gap after it), which
suppresses bold/large *paragraphs* (abstracts) whose lines are full-width and
adjacent. Consecutive numbered lines with consecutive single-level numbers are
treated as a list, not as headings.

**Levels.** Numbering depth and outline levels are authoritative. Each heading
has a style key (size rounded to 0.5 pt, bold); styles learn a level from
numbered headings, remaining styles are ranked by size then boldness and
interleaved monotonically. Multi-line headings are merged if the continuation
has the same (non-body) style, no numbering and no large gap. The title is
the largest text in the upper half of page 1 (≥1.3× body), stored in
`metadata.title`, emitted as a level-0 heading and (optionally) in the section
path. PDF-metadata titles are only used as fallback when they look like real
titles (contain a space, no file extension).

**Paragraphs.** Lines are merged in reading order unless: style changes;
vertical gap > median gap + max(2 pt, 0.35×size); first-line indent > 1×size;
previous line is short (x1 < 80 % of region width for justified text, 60 % for
ragged text – detected per document). Across column/page breaks a paragraph
continues when the previous line is full width or hyphenated and the new line
is not indented. Table/figure blocks are inserted at the reading position of
their claimed lines (table body / caption) or the first line below them.
Lists (bullets `•-–*`, enumerations `1.`, `(a)`, `iv)`) become one `list`
block per run of items; indented wrapped lines are appended to their item.
Equations (equation number at line end or ≥12 % math symbols) are isolated
paragraphs with `extra.equation=True` so they are not glued into prose.
Orphan captions (figure module disabled, unmatched) become `caption` blocks.

**Alternatives.** Font-size histogram thresholds only (fails on bold-only
headings and numbered-but-plain headings), ML heading classifiers (not
deterministic/offline-light), relying on the PDF outline only (absent in most
Office exports).

**Failure cases.** Headings styled identically to body text and without
numbering; run-in headings (`**Methods.** We …` in the same line); drop caps;
two-column text where a floating figure interrupts a paragraph mid-column
(paragraph is split); justified paragraphs whose last line happens to be
full width at a column break (merged with the next paragraph); reference
lists (each entry becomes its own paragraph only if hanging indents exist).

**Cost.** Linear in the number of lines; <50 ms for a 30-page paper.

## Module 4 – Tables (`tables.py`)

**Design.** Several detectors produce candidates `(grid, bbox, method,
confidence)`: PyMuPDF `find_tables` (0.85), pdfplumber line strategy (0.84),
pdfplumber text strategy (0.5, off by default), Camelot lattice/stream
(optional), and our **whitespace detector** (0.45–0.8) for unruled tables:
baseline-aligned rows with ≥2 short cells, consistent x-intervals forming ≥2
columns, ≥3 such rows, mostly short cells, bonus for uniform row lengths and
numeric content. Candidates are validated (≥2×2, fill ratio ≥30 %, two-column
prose rejected), deduplicated by IoU/containment (highest confidence wins),
and sorted by position. Each table gets `TableData` (header → unique column
names, records, CSV, Markdown, DataFrame on demand), a caption ("Table N …"
above preferred, below allowed), and the table's lines are claimed so they
never reappear as paragraphs. Tables continued on the next page (same column
count, bottom→top placement, same header or no caption) are merged.
Optional LLM assistance: low-confidence whitespace candidates are sent to the
local text model as `cell | cell` lines; the returned JSON rows replace the
grid only if they parse and keep ≥2 columns.

**Alternatives.** tabula-py (Java), table-transformer/DETR models (GPU, large),
LLM-only detection (slow, non-deterministic across prompts, hallucinated
cells – therefore LLM only *restructures* existing candidates).

**Failure cases.** Tables with multi-line cells in unruled layouts (rows split
into multiple grid rows); merged header cells (header names concatenated or
`col_n`); rotated tables (lines are `rotated` and ignored); tables drawn as
images (only OCR'd pages with whitespace detection can recover them); figures
with many axis labels detected as whitespace tables (mitigated by the
short-cell/width checks and by running tables before figures with overlap
suppression only in one direction).

**Cost.** PyMuPDF/pdfplumber: 20–200 ms/page (pdfplumber dominates on
graphics-heavy pages); whitespace: O(lines); LLM: 2–15 s per table on CPU.

## Module 5 – Figures & scanned pages (`figures.py`, `ocr.py`)

**Scanned pages.** A page with <25 native characters and ≥50 % raster coverage
is rendered at 300 dpi and OCR'd (Tesseract psm 3). Lines with mean word
confidence ≥40 are converted back to PDF-point boxes and replace the page's
lines, so column detection, boilerplate removal, whitespace tables and
section detection work unchanged (`source="ocr"` is preserved per line/block).

**Figures.** Regions = embedded raster images (≥40 pt and ≥0.5 % of the page;
full-page images behind an OCR text layer are skipped) + clusters of vector
drawings (`cluster_drawings`; rejected when they are mostly text, i.e. text
boxes/frames, or overlap tables). Overlapping regions are merged. Each figure
is rendered (150 dpi), OCR'd (psm 11 sparse text; vector figures with native
text use the native text instead), captioned ("Figure N" below preferred),
and optionally described by a vision model through Ollama (temperature 0,
fixed seed, caption passed as context). Block text = caption + description +
"Text in figure", which is what the chunker will embed.

**Alternatives.** Image-only extraction via `get_images` (misses vector plots,
loses placement); layout detectors (heavy); CLIP captioning (less informative
than a VLM prompted for trends/extremes).

**Failure cases.** Figures assembled from many small images (merged only if
overlapping); decorative images/logos (filtered by size only); charts whose
axis labels are picked up as a whitespace table; VLM hallucinations (bounded by
the "do not speculate" prompt but not eliminated – descriptions are marked
with the model name so they can be filtered); OCR on low-resolution rasters.

**Cost.** Rendering ≈ 50 ms/figure; Tesseract ≈ 0.5–3 s per page/figure on
CPU; 7B vision model ≈ 10–60 s per figure on CPU (optional). Memory: one
rendered page at 300 dpi ≈ 25 MB.

## Module 6 – Cleanup (`cleanup.py`)

ftfy repair → control/zero-width removal → ligature expansion → NFC
normalisation (NFKC deliberately avoided: it would destroy superscripts/units)
→ de-hyphenation at line breaks (join when the continuation starts lowercase
and the prefix is not a known compound prefix like *self-*, *non-*) → dot
leaders → letter-spaced capitals → whitespace collapse (lists keep one item
per line; table cells and figure texts are cleaned individually and the
Markdown/CSV/records renderings are rebuilt).

**Failure cases.** Hyphenated compounds split exactly at the hyphen and not in
the prefix list are joined ("state-of-the-" + "art" → "state-of-theart" is
avoided only because "the" is… not in the list – such cases do occur);
letter-spaced *lowercase* words are left alone; math spacing may be altered.

**Cost.** Negligible (regex, linear).

## Output contract and determinism

`Document.blocks` is ordered in reading order; every block carries `type`,
`text`, `pages`, `bbox`, `section_path`, `level` (headings; 0 = title),
`table`/`figure` payloads and an `extra` dict (method, confidence, n_lines,
source=ocr, equation…). `to_json` is lossless (the layout layer can be omitted
with `include_layout=False`); `to_markdown` emits `#` headings (offset by one
when a title exists), Markdown tables with bold captions, `[Figure: …]`
blocks with description/OCR lines and lists with one item per line.
All heuristics are deterministic; LLM steps use `temperature=0` and a fixed
`seed`, and the model name is stored with each description. Ollama/Tesseract
absence degrades gracefully with a warning in `metadata.warnings`.

## Testing

`tests/` builds synthetic PDFs with PyMuPDF (columns, running headers, ruled
and unruled tables, embedded images, outline entries, lists) and tests every
module in isolation plus the end-to-end pipeline, determinism, ablation
switches, custom-step swapping, failure recording and the CLI. OCR tests are
skipped automatically when Tesseract is not installed; Ollama is mocked.
