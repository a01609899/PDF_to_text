"""Thin wrapper around Tesseract (via pytesseract) returning line boxes."""
from __future__ import annotations

import functools
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .models import BBox

log = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    bbox: BBox  # pixel coordinates of the OCR'd image
    conf: float


@functools.lru_cache(maxsize=1)
def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def pixmap_to_image(pix):
    """Convert a PyMuPDF Pixmap to a PIL image via PNG bytes (handles alpha/colorspaces)."""
    from PIL import Image  # noqa: WPS433

    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def ocr_lines(image, lang: str = "eng", psm: int = 3, min_conf: float = 40.0) -> List[OcrLine]:
    """Run Tesseract and return text lines with pixel bounding boxes and mean confidence."""
    import pytesseract  # noqa: WPS433

    data: Dict[str, Any] = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {int(psm)}", output_type=pytesseract.Output.DICT
    )
    n = len(data.get("text", []))
    groups: Dict[tuple, list] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        left, top = int(data["left"][i]), int(data["top"][i])
        width, height = int(data["width"][i]), int(data["height"][i])
        groups.setdefault(key, []).append((left, top, left + width, top + height, txt, conf))
    out: List[OcrLine] = []
    for key in sorted(groups):
        words = sorted(groups[key], key=lambda w: w[0])
        bbox = BBox(min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words))
        text = " ".join(w[4] for w in words)
        conf = sum(w[5] for w in words) / len(words)
        out.append(OcrLine(text=text, bbox=bbox, conf=conf))
    out.sort(key=lambda l: (l.bbox.y0, l.bbox.x0))
    return out


def ocr_text(image, lang: str = "eng", psm: int = 11, min_conf: float = 40.0) -> str:
    try:
        lines = ocr_lines(image, lang=lang, psm=psm, min_conf=min_conf)
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR failed: %s", exc)
        return ""
    return "\n".join(l.text for l in lines)
