from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.models import Block, Page, TableData
from fable_pipeline.sections import detect_sections
from fable_pipeline.tables import _merge_continued_tables, _parse_json_rows, extract_tables
from tests.conftest import add_page, body_lines, draw_grid_table, new_pdf, save, write_lines


def _ruled_pdf(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(2), y=60)
    page.insert_text((72, 110), "Table 1: Synthetic measurements", fontsize=10, fontname="helv")
    draw_grid_table(page, 72, 120, [100, 100, 100], 20,
                    [["Sample", "Mass", "Volume"], ["A", "1.0", "2.5"], ["B", "3.0", "4.5"], ["C", "5.0", "6.5"]])
    write_lines(page, body_lines(2, "Text after the table line"), y=230)
    return save(pdf, tmp_path / "ruled.pdf")


def test_ruled_table_extraction(tmp_path):
    path = _ruled_pdf(tmp_path)
    cfg = PipelineConfig()
    doc = extract_tables(parse_layout(path, cfg), cfg)
    tables = [b for p in doc.pages for b in p.region_blocks if b.type == "table"]
    assert len(tables) == 1
    td = tables[0].table
    assert td.columns == ["Sample", "Mass", "Volume"]
    assert td.records[0] == {"Sample": "A", "Mass": "1.0", "Volume": "2.5"}
    assert td.n_rows == 3
    assert td.caption == "Table 1: Synthetic measurements"
    assert "| Sample | Mass | Volume |" in td.markdown
    claimed = [ln for ln in doc.pages[0].lines if ln.claimed == tables[0].id]
    assert any(ln.role == "table" for ln in claimed) and any(ln.role == "caption" for ln in claimed)
    doc = detect_sections(doc, cfg)
    types = [b.type for b in doc.blocks]
    assert "table" in types
    idx = types.index("table")
    assert doc.blocks[idx - 1].text.startswith("This is body")
    assert doc.blocks[idx + 1].text.startswith("Text after")
    assert not any("Sample" in b.text for b in doc.blocks if b.type == "paragraph")


def test_whitespace_table_detection(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(3), y=60)
    rows = [["Country", "2019", "2020"], ["Germany", "83.1", "83.2"], ["France", "67.0", "67.4"],
            ["Spain", "47.1", "47.4"], ["Italy", "59.8", "59.6"]]
    y = 130
    for row in rows:
        for x, cell in zip((72, 220, 340), row):
            page.insert_text((x, y), cell, fontsize=10, fontname="helv")
        y += 14
    write_lines(page, body_lines(3, "Prose below the table line"), y=y + 20)
    path = save(pdf, tmp_path / "ws.pdf")
    cfg = PipelineConfig()
    cfg.tables.strategies = ["whitespace"]
    doc = extract_tables(parse_layout(path, cfg), cfg)
    tables = [b for p in doc.pages for b in p.region_blocks if b.type == "table"]
    assert len(tables) == 1
    td = tables[0].table
    assert td.method == "whitespace"
    assert td.columns == ["Country", "2019", "2020"]
    assert td.n_rows == 4 and td.rows[0] == ["Germany", "83.1", "83.2"]


def test_tables_disabled(tmp_path):
    path = _ruled_pdf(tmp_path)
    cfg = PipelineConfig().disable("tables")
    doc = extract_tables(parse_layout(path, cfg), cfg)
    assert not any(p.region_blocks for p in doc.pages)


def test_parse_json_rows():
    assert _parse_json_rows('Here: [["a","b"],["1",2]] done') == [["a", "b"], ["1", "2"]]
    assert _parse_json_rows("no json") is None
    assert _parse_json_rows('[{"a": 1}]') is None


def test_merge_continued_tables():
    p1 = Page(number=1, width=600, height=800)
    p2 = Page(number=2, width=600, height=800)
    t1 = TableData.from_grid([["A", "B"], ["1", "2"]], caption="Table 1")
    t2 = TableData.from_grid([["A", "B"], ["3", "4"]])
    p1.region_blocks.append(Block(id="t1", type="table", text="", pages=[1], bbox=_bb(50, 600, 300, 780), table=t1))
    p2.region_blocks.append(Block(id="t2", type="table", text="", pages=[2], bbox=_bb(50, 50, 300, 150), table=t2))
    from fable_pipeline.models import Document, DocumentMetadata

    doc = Document(metadata=DocumentMetadata("x", "x", 2), pages=[p1, p2])
    _merge_continued_tables(doc)
    assert not p2.region_blocks
    merged = p1.region_blocks[0]
    assert merged.table.rows == [["1", "2"], ["3", "4"]]
    assert merged.pages == [1, 2]


def _bb(x0, y0, x1, y1):
    from fable_pipeline.models import BBox

    return BBox(x0, y0, x1, y1)
