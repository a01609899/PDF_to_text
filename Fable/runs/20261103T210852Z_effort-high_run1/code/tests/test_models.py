from fable_pipeline.models import (
    BBox,
    Block,
    Document,
    DocumentMetadata,
    FigureData,
    TableData,
    from_json,
    normalize_grid,
    render_markdown_table,
    to_json,
)


def test_bbox_geometry():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert a.area == 100
    assert a.intersection_area(b) == 25
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert a.union(b).to_list() == [0, 0, 15, 15]
    assert a.contains_point(5, 5)
    assert BBox.from_any((10, 10, 0, 0)).to_list() == [0, 0, 10, 10]
    assert a.overlap_ratio(BBox(0, 0, 5, 10)) == 0.5


def test_normalize_grid_drops_empty_rows_and_cols():
    grid = [["a", None, "b"], [None, None, None], ["c", "", "d"]]
    assert normalize_grid(grid) == [["a", "b"], ["c", "d"]]
    assert normalize_grid(None) == []
    assert normalize_grid([["", ""]]) == []


def test_table_data_from_grid_and_renderings():
    td = TableData.from_grid([["Name", "Value"], ["x", "1"], ["y", "2"]], caption="Table 1: T")
    assert td.columns == ["Name", "Value"]
    assert td.n_rows == 2 and td.n_cols == 2
    assert td.records == [{"Name": "x", "Value": "1"}, {"Name": "y", "Value": "2"}]
    assert "| Name | Value |" in td.markdown
    assert "| --- | --- |" in td.markdown
    assert td.csv.splitlines()[0] == "Name,Value"
    df = td.to_dataframe()
    assert list(df.columns) == ["Name", "Value"] and len(df) == 2


def test_table_data_duplicate_and_missing_headers():
    td = TableData.from_grid([["A", "", "A"], ["1", "2", "3"]])
    assert td.columns == ["A", "col_2", "A (2)"]
    td2 = TableData.from_grid([["1", "2"], ["3", "4"]], header_row=False)
    assert td2.columns == ["col_1", "col_2"] and td2.n_rows == 2


def test_markdown_table_escapes_pipes():
    md = render_markdown_table(["a|b", "c"], [["x|y", "z"]])
    assert "a\\|b" in md and "x\\|y" in md


def test_json_roundtrip_is_lossless():
    meta = DocumentMetadata(source_path="/x/y.pdf", file_name="y.pdf", page_count=1, title="T", warnings=["w"])
    blocks = [
        Block(id="h1", type="heading", text="Intro", pages=[1], bbox=BBox(1, 2, 3, 4), section_path=["Intro"], level=1),
        Block(id="t1", type="table", text="t", pages=[1], bbox=None,
              table=TableData.from_grid([["a", "b"], ["1", "2"]], caption="Table 1")),
        Block(id="f1", type="figure", text="f", pages=[1], bbox=BBox(0, 0, 1, 1),
              figure=FigureData(caption="Figure 1", ocr_text="abc", description="a plot")),
    ]
    doc = Document(metadata=meta, blocks=blocks)
    js = to_json(doc)
    back = from_json(js)
    assert back.to_dict() == doc.to_dict()
    assert back.tables[0].table.records == [{"a": "1", "b": "2"}]
    assert back.figures[0].figure.description == "a plot"
    assert back.headings[0].level == 1
