"""Turn an uploaded image or single-page PDF into a BGR array."""
from __future__ import annotations

import cv2
import numpy as np
import pymupdf

RENDER_DPI = 200
MAX_SIDE = 4000          # px, cap for rendered pages
SCAN_COVERAGE = 0.8      # a page whose single image covers this share of it is treated as a scan


def _decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def pdf_to_image(data: bytes) -> tuple[np.ndarray | None, str | None]:
    """(image, None) for a readable single-page PDF, else (None, reason).
    A page that is just one embedded scan / photo returns that image at its native resolution (keeps the compression
    traces the forensic checks rely on); anything else (text, vector art) is rendered at RENDER_DPI."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return None, "That PDF could not be opened. It may be damaged."
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            return None, "That PDF is password-protected. Remove the password and upload it again."
        if doc.page_count != 1:
            return None, f"Only single-page PDFs are supported, and this one has {doc.page_count} pages."
        page = doc[0]
        area = page.rect.width * page.rect.height
        imgs = page.get_images(full=True)
        if len(imgs) == 1 and area > 0:
            rects = page.get_image_rects(imgs[0][0])
            if rects and sum(r.width * r.height for r in rects) / area >= SCAN_COVERAGE:
                try:
                    img = _decode(doc.extract_image(imgs[0][0])["image"])
                    if img is not None:
                        return img, None
                except Exception:
                    pass  # fall through to rendering
        zoom = min(RENDER_DPI / 72, MAX_SIDE / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None


def load_image(data: bytes) -> tuple[np.ndarray | None, str | None]:
    """(BGR image, None) or (None, reason) for JPG / PNG bytes or a single-page PDF."""
    if data[:5] == b"%PDF-":
        return pdf_to_image(data)
    img = _decode(data)
    return (img, None) if img is not None else (None, "That file could not be decoded as an image.")
