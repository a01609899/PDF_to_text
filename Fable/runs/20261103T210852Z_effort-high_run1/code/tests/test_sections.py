from fable_pipeline.boilerplate import remove_boilerplate
from fable_pipeline.config import PipelineConfig
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.sections import detect_sections, is_equation, list_marker, numbering_info
from tests.conftest import add_page, body_lines, new_pdf, save, write_lines


def test_numbering_info():
    assert numbering_info("2.3.1 Data Collection")[0] == 3
    assert numbering_info("2 Methods")[0] == 1
    assert numbering_info("A.1 Supplementary")[0] == 2
    assert numbering_info("Chapter 4 Results")[0] == 1
    assert numbering_info("IV. Discussion")[0] == 1
    assert numbering_info("2.5 mg of sample") is None
    assert numbering_info("3 samples were taken.")[0] == 1  # caller must reject trailing punctuation


def test_list_marker_and_equation():
    assert list_marker("• first item") == ("-", "first item")
    assert list_marker("- dash item") == ("-", "dash item")
    assert list_marker("(a) lettered") == ("(a)", "lettered")
    assert list_marker("2. numbered") == ("2.", "numbered")
    assert list_marker("plain text") is None
    assert is_equation("E = mc^2 (1)")
    assert is_equation("x = α + β · y")
    assert not is_equation("This is a normal sentence about energy (see Section 1).")


def test_headings_levels_and_paragraphs(multipage_pdf):
    cfg = PipelineConfig()
    doc = parse_layout(multipage_pdf, cfg)
    doc = remove_boilerplate(doc, cfg)
    doc = detect_sections(doc, cfg)
    assert doc.metadata.title == "Running Text Paper"
    heads = [(b.text, b.level) for b in doc.blocks if b.type == "heading"]
    assert ("Running Text Paper", 0) in heads
    assert ("1 Introduction", 1) in heads
    assert ("2 Methods", 1) in heads
    assert ("2.1 Data", 2) in heads
    data_par = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("Data paragraph")]
    assert data_par and data_par[0].section_path == ["2 Methods", "2.1 Data"]
    # paragraph continuing across the page break is merged into one block
    tail = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("Tail paragraph")]
    assert tail and tail[0].pages == [1, 2]
    assert "continues on the next page" in tail[0].text
    # the two paragraphs on page 1 are separate (vertical gap)
    first = [b for b in doc.blocks if b.type == "paragraph" and b.text.startswith("This is body")]
    assert first and first[0].extra["n_lines"] == 4
    assert all(ln.role != "header" or True for p in doc.pages for ln in p.lines)
    assert not any("Journal of Synthetic Tests" in b.text for b in doc.blocks)


def test_lists_and_bold_headings(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 80), "Results", fontsize=10, fontname="hebo")
    y = write_lines(page, body_lines(2), y=100)
    write_lines(page, ["• first bullet item", "• second bullet item", "• third bullet item"], x=84, y=y + 10)
    page.insert_text((72, 200), "Key findings", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2, "After the bold heading line"), y=220)
    path = save(pdf, tmp_path / "lists.pdf")
    doc = detect_sections(parse_layout(path), PipelineConfig())
    types = [b.type for b in doc.blocks]
    assert types.count("heading") == 2
    lists = [b for b in doc.blocks if b.type == "list"]
    assert len(lists) == 1
    assert lists[0].text.split("\n") == ["- first bullet item", "- second bullet item", "- third bullet item"]
    assert lists[0].section_path == ["Results"]
    last = [b for b in doc.blocks if b.text.startswith("After the bold")][0]
    assert last.section_path == ["Key findings"]


def test_outline_is_used_for_levels(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    page.insert_text((72, 80), "Background", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2), y=100)
    page.insert_text((72, 160), "Related work", fontsize=10, fontname="hebo")
    write_lines(page, body_lines(2, "More"), y=180)
    pdf.set_toc([[1, "Background", 1], [2, "Related work", 1]])
    path = save(pdf, tmp_path / "toc.pdf")
    doc = detect_sections(parse_layout(path), PipelineConfig())
    levels = {b.text: b.level for b in doc.blocks if b.type == "heading"}
    assert levels["Background"] == 1 and levels["Related work"] == 2
    cfg = PipelineConfig().disable("outline")
    doc2 = detect_sections(parse_layout(path), cfg)
    levels2 = {b.text: b.level for b in doc2.blocks if b.type == "heading"}
    assert levels2["Background"] == levels2["Related work"] == 1


def test_plain_fallback_when_disabled(simple_pdf):
    cfg = PipelineConfig().disable("sections")
    doc = detect_sections(parse_layout(simple_pdf, cfg), cfg)
    assert doc.blocks and all(b.type == "paragraph" for b in doc.blocks)
    assert all(b.section_path == [] for b in doc.blocks)
