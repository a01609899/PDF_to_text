from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import dominant_font_size, parse_layout
from tests.conftest import add_page, new_pdf, save, write_lines


def test_parse_layout_basic(simple_pdf):
    doc = parse_layout(simple_pdf, PipelineConfig())
    assert doc.metadata.page_count == 1
    assert doc.metadata.file_name == "simple.pdf"
    page = doc.pages[0]
    assert page.width > 0 and page.height > 0
    texts = [ln.text for ln in page.lines]
    assert texts[0] == "A Synthetic Test Document"
    title = page.lines[0]
    assert title.bold and abs(title.font_size - 18) < 0.5
    assert "Bold" in title.font_name
    body = [ln for ln in page.lines if ln.text.startswith("This is body")]
    assert body and not body[0].bold and abs(body[0].font_size - 10) < 0.5
    assert [ln.order for ln in page.lines] == list(range(len(page.lines)))
    assert dominant_font_size(doc.pages) == 10.0


def test_two_column_reading_order(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    left = [f"Left column text line {i} order test" for i in range(1, 7)]
    right = [f"Right column text line {i} order test" for i in range(1, 7)]
    write_lines(page, left, x=50, y=100)
    write_lines(page, right, x=320, y=100)
    path = save(pdf, tmp_path / "cols.pdf")

    doc = parse_layout(path, PipelineConfig())
    texts = [ln.text for ln in doc.pages[0].lines]
    assert texts[:6] == left
    assert texts[6:] == right
    regions = {ln.region for ln in doc.pages[0].lines}
    assert len(regions) == 2

    cfg = PipelineConfig().disable("columns")
    doc2 = parse_layout(path, cfg)
    texts2 = [ln.text for ln in doc2.pages[0].lines]
    assert texts2[:2] == [left[0], right[0]]  # row-wise without column detection


def test_full_width_band_above_columns(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((50, 60), "Abstract spanning the whole width of the page in one long single line of text here", fontsize=10)
    left = [f"Left col line {i} with some words" for i in range(1, 6)]
    right = [f"Right col line {i} with some words" for i in range(1, 6)]
    write_lines(page, left, x=50, y=120)
    write_lines(page, right, x=320, y=120)
    path = save(pdf, tmp_path / "band.pdf")
    doc = parse_layout(path, PipelineConfig())
    texts = [ln.text for ln in doc.pages[0].lines]
    assert texts[0].startswith("Abstract")
    assert texts[1:6] == left and texts[6:] == right


def test_fragment_merge(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 100), "Hello", fontsize=10)
    page.insert_text((100, 100), "world again", fontsize=10)
    path = save(pdf, tmp_path / "frag.pdf")
    doc = parse_layout(path, PipelineConfig())
    assert len(doc.pages[0].lines) == 1
    assert doc.pages[0].lines[0].text == "Hello world again"
