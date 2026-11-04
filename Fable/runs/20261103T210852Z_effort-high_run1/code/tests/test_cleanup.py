from fable_pipeline.cleanup import clean_text, cleanup_document
from fable_pipeline.config import CleanupConfig, PipelineConfig
from fable_pipeline.models import BBox, Block, Document, DocumentMetadata, FigureData, TableData


def test_ligatures_and_unicode():
    assert clean_text("e\ufb03cient \ufb01ne \ufb02ow") == "efficient fine flow"
    assert clean_text("caf\u0065\u0301") == "café"  # NFC composition


def test_dehyphenation_across_line_breaks():
    assert clean_text("the experi-\nment worked") == "the experiment worked"
    assert clean_text("a self-\ncontained unit") == "a self-contained unit"
    assert clean_text("the 2019-\n2020 season") == "the 2019-2020 season"
    assert clean_text("see Fig-\nure 3") == "see Figure 3"
    assert clean_text("soft\u00adhyphen") == "softhyphen"


def test_whitespace_and_control_chars():
    assert clean_text("a \u200b b\x0c  c\n\nd ,e .") == "a b c d,e."
    assert clean_text("I N T R O D U C T I O N and more") == "INTRODUCTION and more"
    assert clean_text("Introduction ........ 5") == "Introduction 5"
    assert clean_text("( text )") == "(text)"


def test_keep_newlines_for_lists():
    out = clean_text("- first\n- sec-\nond\n- third", keep_newlines=True)
    assert out == "- first\n- second\n- third"


def test_config_switches():
    cfg = CleanupConfig(dehyphenate=False, fix_ligatures=False, use_ftfy=False)
    assert clean_text("experi-\nment \ufb01", cfg) == "experi- ment \ufb01"
    cfg2 = CleanupConfig(normalize_quotes=True)
    assert clean_text("\u201cquoted\u201d", cfg2) == '"quoted"'


def test_cleanup_document_blocks():
    meta = DocumentMetadata("x.pdf", "x.pdf", 1, title="A \ufb01ne  title")
    td = TableData.from_grid([["Na\ufb01me", "Val ue"], ["x", "1"]])
    blocks = [
        Block(id="p", type="paragraph", text="line one con-\ntinued here", pages=[1], bbox=BBox(0, 0, 1, 1)),
        Block(id="l", type="list", text="- item one\nwrapped\n- item two", pages=[1], bbox=None, extra={"items": []}),
        Block(id="t", type="table", text="", pages=[1], bbox=None, table=td),
        Block(id="f", type="figure", text="", pages=[1], bbox=None, figure=FigureData(caption="Figure 1: \ufb02ow", ocr_text="a\n\nb")),
        Block(id="e", type="paragraph", text="   ", pages=[1], bbox=None),
    ]
    doc = cleanup_document(Document(meta, blocks), PipelineConfig())
    texts = {b.id: b.text for b in doc.blocks}
    assert texts["p"] == "line one continued here"
    assert texts["l"] == "- item one wrapped\n- item two"
    assert doc.blocks[1].extra["items"] == ["- item one wrapped", "- item two"]
    assert doc.blocks[2].table.columns == ["Nafime", "Val ue"]
    assert doc.blocks[3].figure.caption == "Figure 1: flow"
    assert "e" not in texts  # empty paragraph dropped
    assert doc.metadata.title == "A fine title"


def test_cleanup_disabled_keeps_text():
    cfg = PipelineConfig().disable("cleanup")
    doc = Document(DocumentMetadata("x", "x", 1), [Block(id="p", type="paragraph", text="a-\nb", pages=[1], bbox=None)])
    assert cleanup_document(doc, cfg).blocks[0].text == "a-\nb"
