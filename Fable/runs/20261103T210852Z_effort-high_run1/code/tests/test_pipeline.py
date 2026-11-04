import json
import os

from fable_pipeline import PipelineConfig, extract, from_json, to_json, to_markdown
from fable_pipeline.__main__ import main
from tests.conftest import add_page, body_lines, draw_grid_table, insert_png, make_png, new_pdf, save, write_lines


def _rich_pdf(tmp_path):
    pdf = new_pdf()
    for i in range(3):
        page = add_page(pdf)
        page.insert_text((72, 30), "Synthetic Conference Proceedings", fontsize=8, fontname="helv")
        page.insert_text((290, 820), f"Page {i + 1} of 3", fontsize=8, fontname="helv")
        if i == 0:
            page.insert_text((72, 90), "Rich Synthetic Document", fontsize=18, fontname="hebo")
            page.insert_text((72, 140), "1 Introduction", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(3), y=170)
            page.insert_text((72, 240), "Table 1: Key numbers", fontsize=10)
            draw_grid_table(page, 72, 250, [120, 120], 20, [["Metric", "Value"], ["Accuracy", "0.91"], ["Recall", "0.87"]])
            write_lines(page, body_lines(2, "After table line"), y=340)
        elif i == 1:
            page.insert_text((72, 80), "2 Methods", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(2, "Methods line"), y=110)
            insert_png(page, (72, 150, 272, 300), make_png())
            page.insert_text((72, 320), "Figure 1: Synthetic image", fontsize=10)
            write_lines(page, ["• bullet one", "• bullet two"], x=84, y=360)
        else:
            page.insert_text((72, 80), "3 Conclusion", fontsize=14, fontname="hebo")
            write_lines(page, body_lines(2, "Final line"), y=110)
    return save(pdf, tmp_path / "rich.pdf")


def test_end_to_end(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    doc = extract(path, cfg)
    assert doc.metadata.title == "Rich Synthetic Document"
    assert doc.metadata.page_count == 3
    assert "ingestion" in doc.metadata.steps_applied and "sections" in doc.metadata.steps_applied
    types = [b.type for b in doc.blocks]
    assert "table" in types and "figure" in types and "list" in types and "heading" in types
    heads = [b.text for b in doc.blocks if b.type == "heading"]
    assert heads == ["Rich Synthetic Document", "1 Introduction", "2 Methods", "3 Conclusion"]
    table = doc.tables[0]
    assert table.table.records[0] == {"Metric": "Accuracy", "Value": "0.91"}
    assert table.section_path == ["1 Introduction"]
    fig = doc.figures[0]
    assert fig.figure.caption == "Figure 1: Synthetic image" and fig.section_path == ["2 Methods"]
    lst = [b for b in doc.blocks if b.type == "list"][0]
    assert lst.text == "- bullet one\n- bullet two"
    md = to_markdown(doc)
    assert md.startswith("# Rich Synthetic Document")
    assert "Synthetic Conference Proceedings" not in md
    assert "Page 2 of 3" not in md
    assert "| Metric | Value |" in md and "[Figure: Figure 1: Synthetic image]" in md
    # paragraphs are single-line clean text
    for b in doc.blocks:
        if b.type == "paragraph":
            assert "\n" not in b.text
    back = from_json(to_json(doc))
    assert back.to_dict() == doc.to_dict()


def test_determinism(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    a, b = extract(path, cfg).to_dict(), extract(path, cfg).to_dict()
    a["metadata"]["timings"] = b["metadata"]["timings"] = {}
    assert a == b


def test_ablation_and_custom_step(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("boilerplate", "tables", "figures", "sections", "ocr")
    doc = extract(path, cfg)
    assert set(doc.metadata.steps_skipped) >= {"boilerplate", "tables", "figures"}
    assert all(b.type == "paragraph" for b in doc.blocks)
    assert any("Synthetic Conference Proceedings" in b.text for b in doc.blocks)

    calls = []

    def my_boilerplate(document, config):
        calls.append(len(document.pages))
        return document

    cfg2 = PipelineConfig().disable("ocr")
    cfg2.custom_steps["boilerplate"] = my_boilerplate
    doc2 = extract(path, cfg2)
    assert calls == [3]
    assert any("Synthetic Conference Proceedings" in b.text for b in doc2.blocks)
    assert doc2.metadata.config["custom_steps"] == ["boilerplate"]


def test_failed_step_is_recorded(tmp_path):
    path = _rich_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")

    def boom(document, config):
        raise RuntimeError("kaboom")

    cfg.custom_steps["tables"] = boom
    doc = extract(path, cfg)
    assert "tables" in doc.metadata.steps_skipped
    assert any("kaboom" in w for w in doc.metadata.warnings)
    assert doc.blocks


def test_cli_writes_outputs(tmp_path):
    path = _rich_pdf(tmp_path)
    out = tmp_path / "out"
    rc = main([path, "--out", str(out), "--disable", "ocr", "--no-layout"])
    assert rc == 0
    md = out / "rich.md"
    js = out / "rich.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "pages" not in data and data["metadata"]["title"] == "Rich Synthetic Document"
    assert md.read_text(encoding="utf-8").startswith("# Rich Synthetic Document")
    assert main([str(tmp_path / "missing.pdf"), "--out", str(out)]) == 1
    assert not os.path.exists(out / "missing.md")
