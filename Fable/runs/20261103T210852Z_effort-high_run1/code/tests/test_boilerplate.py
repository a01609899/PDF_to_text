from fable_pipeline.boilerplate import is_page_number, normalize_key, remove_boilerplate
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout


def test_page_number_patterns():
    for t in ["12", "iv", "Page 3 of 10", "- 7 -", "Seite 4", "3 / 12", "A-3"]:
        assert is_page_number(t), t
    for t in ["Introduction", "Table 1", "2019 was a year", "3.5 Results"]:
        assert not is_page_number(t), t


def test_normalize_key():
    assert normalize_key("Chapter 2  Methods   17") == normalize_key("Chapter 3 Methods 18")
    assert normalize_key("  -- Hello --  ") == "hello"


def test_running_header_and_page_numbers_removed(multipage_pdf):
    doc = parse_layout(multipage_pdf, PipelineConfig())
    doc = remove_boilerplate(doc, PipelineConfig())
    for page in doc.pages:
        roles = {ln.text: ln.role for ln in page.lines}
        assert roles["Journal of Synthetic Tests"] == "header"
        assert roles[str(page.number)] == "page_number"
    body = [ln for p in doc.pages for ln in p.lines if ln.role == "body"]
    assert all(ln.text != "Journal of Synthetic Tests" for ln in body)
    assert any(ln.text.startswith("This is body") for ln in body)
    assert "Journal of Synthetic Tests" in doc.metadata.boilerplate["headers"]
    assert "page_numbers" in doc.metadata.boilerplate


def test_disabled_module_keeps_everything(multipage_pdf):
    cfg = PipelineConfig().disable("boilerplate")
    doc = remove_boilerplate(parse_layout(multipage_pdf, cfg), cfg)
    assert all(ln.role == "body" for p in doc.pages for ln in p.lines)


def test_single_page_title_is_not_a_header(simple_pdf):
    doc = remove_boilerplate(parse_layout(simple_pdf), PipelineConfig())
    assert doc.pages[0].lines[0].role == "body"
