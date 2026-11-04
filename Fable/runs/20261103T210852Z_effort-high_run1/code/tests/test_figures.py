import json
import pytest

from fable_pipeline.config import PipelineConfig
from fable_pipeline.figures import compose_figure_text, extract_figures, is_scanned_candidate, ocr_scanned_pages
from fable_pipeline.ingestion import parse_layout
from fable_pipeline.models import FigureData
from fable_pipeline.ocr import tesseract_available
from fable_pipeline.sections import detect_sections
from tests.conftest import add_page, body_lines, insert_png, make_png, new_pdf, save, write_lines


def _figure_pdf(tmp_path, text=None):
    pdf = new_pdf()
    page = add_page(pdf)
    write_lines(page, body_lines(2), y=60)
    insert_png(page, (72, 100, 272, 250), make_png(400, 300, text=text))
    page.insert_text((72, 270), "Figure 1: A synthetic red rectangle", fontsize=10, fontname="helv")
    write_lines(page, body_lines(2, "Text after the figure line"), y=300)
    return save(pdf, tmp_path / "fig.pdf")


def test_embedded_image_becomes_figure_with_caption(tmp_path):
    path = _figure_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    doc = extract_figures(parse_layout(path, cfg), cfg)
    figs = [b for b in doc.pages[0].region_blocks if b.type == "figure"]
    assert len(figs) == 1
    fd = figs[0].figure
    assert fd.kind == "raster" and fd.caption == "Figure 1: A synthetic red rectangle"
    assert fd.width_px > 0 and figs[0].bbox.width > 150
    assert fd.description is None
    doc = detect_sections(doc, cfg)
    types = [b.type for b in doc.blocks]
    assert types == ["paragraph", "figure", "paragraph"]
    assert "Figure 1" in doc.blocks[1].text
    assert "figure OCR requested" not in " ".join(doc.metadata.warnings)


def test_save_images(tmp_path):
    path = _figure_pdf(tmp_path)
    cfg = PipelineConfig().disable("ocr")
    cfg.figures.save_images = True
    cfg.figures.image_dir = str(tmp_path / "imgs")
    doc = extract_figures(parse_layout(path, cfg), cfg)
    fd = doc.pages[0].region_blocks[0].figure
    assert fd.image_path and fd.image_path.endswith(".png")
    assert (tmp_path / "imgs").exists()


def test_scanned_candidate_detection(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    insert_png(page, (20, 20, 575, 822), make_png(1000, 1400, color=(255, 255, 255), text="SCANNED PAGE TEXT", text_size=90))
    path = save(pdf, tmp_path / "scan.pdf")
    cfg = PipelineConfig()
    doc = parse_layout(path, cfg)
    assert is_scanned_candidate(doc.pages[0], cfg.figures)
    cfg.figures.ocr_enabled = False
    doc = ocr_scanned_pages(doc, cfg)
    assert any("OCR is disabled" in w for w in doc.metadata.warnings)
    assert not doc.pages[0].is_scanned


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")
def test_scanned_page_is_ocred(tmp_path):
    pdf = new_pdf()
    page = add_page(pdf)
    insert_png(page, (20, 20, 575, 822), make_png(1100, 1500, color=(255, 255, 255), text="HELLO SCANNED WORLD", text_size=80))
    path = save(pdf, tmp_path / "scan2.pdf")
    cfg = PipelineConfig()
    doc = ocr_scanned_pages(parse_layout(path, cfg), cfg)
    assert doc.pages[0].is_scanned
    assert doc.pages[0].lines and doc.pages[0].lines[0].source == "ocr"
    assert doc.metadata.scanned_pages == [1]
    joined = " ".join(ln.text for ln in doc.pages[0].lines).upper()
    assert "HELLO" in joined or "SCANNED" in joined or "WORLD" in joined


def test_compose_figure_text():
    fd = FigureData(caption="Figure 2: Trend", ocr_text="x axis\ny axis", description="A rising line plot.")
    txt = compose_figure_text(fd)
    assert txt.splitlines() == ["Figure 2: Trend", "Description: A rising line plot.", "Text in figure: x axis y axis"]
    assert compose_figure_text(FigureData()) == "[Figure]"


class _FakeResponse:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_description_via_fake_ollama(tmp_path, monkeypatch):
    path = _figure_pdf(tmp_path)
    captured = {}

    def fake_opener(req, timeout=None):
        if req.full_url.endswith("/api/tags"):
            return _FakeResponse({"models": [{"name": "llava:7b"}]})
        payload = json.loads(req.data.decode())
        captured.update(payload)
        return _FakeResponse({"response": "A red rectangle on a white background."})

    import fable_pipeline.figures as figmod
    from fable_pipeline.ollama_client import OllamaClient

    monkeypatch.setattr(figmod, "OllamaClient", lambda cfg: OllamaClient(cfg, opener=fake_opener))
    cfg = PipelineConfig().disable("ocr")
    cfg.figures.describe_with_llm = True
    doc = extract_figures(parse_layout(path, cfg), cfg)
    fd = doc.pages[0].region_blocks[0].figure
    assert fd.description == "A red rectangle on a white background."
    assert fd.description_model == "llava:7b"
    assert captured["options"]["temperature"] == 0.0 and captured["options"]["seed"] == 42
    assert captured["images"] and captured["model"] == "llava:7b"
    assert "Figure 1" in captured["prompt"]
