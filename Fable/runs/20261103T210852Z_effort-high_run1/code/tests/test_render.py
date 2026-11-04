from fable_pipeline.models import BBox, Block, Document, DocumentMetadata, FigureData, TableData
from fable_pipeline.render import to_markdown


def _doc(with_title=True):
    blocks = []
    if with_title:
        blocks.append(Block(id="t", type="heading", text="My Title", pages=[1], bbox=None, level=0, extra={"is_title": True}))
    blocks += [
        Block(id="h1", type="heading", text="1 Intro", pages=[1], bbox=BBox(0, 0, 1, 1), level=1, section_path=["1 Intro"]),
        Block(id="p1", type="paragraph", text="Hello world.", pages=[1], bbox=None, section_path=["1 Intro"]),
        Block(id="h2", type="heading", text="1.1 Sub", pages=[1], bbox=None, level=2),
        Block(id="l1", type="list", text="- a\n- b", pages=[1], bbox=None),
        Block(id="tb", type="table", text="", pages=[1], bbox=None,
              table=TableData.from_grid([["A", "B"], ["1", "2"]], caption="Table 1: Numbers")),
        Block(id="fg", type="figure", text="", pages=[1], bbox=None,
              figure=FigureData(caption="Figure 1: Plot", description="Rising trend.", ocr_text="x y")),
        Block(id="c", type="caption", text="Figure 9: orphan caption", pages=[1], bbox=None),
    ]
    return Document(DocumentMetadata("x", "x", 1), blocks)


def test_markdown_rendering_with_title():
    md = to_markdown(_doc())
    assert md.startswith("# My Title\n\n## 1 Intro\n\nHello world.\n\n### 1.1 Sub\n\n- a\n- b\n\n")
    assert "**Table 1: Numbers**\n\n| A | B |\n| --- | --- |\n| 1 | 2 |" in md
    assert "[Figure: Figure 1: Plot]\nDescription: Rising trend.\nText in figure: x y" in md
    assert "*Figure 9: orphan caption*" in md
    assert md.endswith("\n")


def test_markdown_rendering_without_title():
    md = to_markdown(_doc(with_title=False))
    assert md.startswith("# 1 Intro\n\n")
    assert "\n## 1.1 Sub\n" in md


def test_empty_document():
    assert to_markdown(Document(DocumentMetadata("x", "x", 0), [])) == ""
