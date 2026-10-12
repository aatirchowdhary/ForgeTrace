"""Input gate: reject images that are not a readable Aadhaar card, PAN card or marksheet before any verdict is made."""
from __future__ import annotations

import cv2
import numpy as np

MIN_SIDE = 300          # px, shorter side
MIN_STD = 8.0           # grey-level std-dev; below this the image is (nearly) one flat colour
MIN_SHARPNESS = 50.0    # Laplacian variance at 1000 px width (real phone photos here: 400-6800; blurred / flat: ~0-2)
MIN_TEXT_BOXES = 6      # confidently read text boxes
MIN_EVIDENCE = 2        # independent document-type cues (header words, ID number pattern) needed to accept a type
SUPPORTED = "an Aadhaar card, a PAN card or a school marksheet"


def sharpness(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = 1000 / max(g.shape)
    return float(cv2.Laplacian(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.CV_64F).var())


def check_image(img_bgr: np.ndarray) -> str | None:
    """Cheap checks before OCR. Returns a rejection reason, or None when the image may proceed."""
    h, w = img_bgr.shape[:2]
    if min(h, w) < MIN_SIDE:
        return f"The image is too small ({w}x{h}px). Upload a photo or scan at least {MIN_SIDE}px on its short side."
    if float(img_bgr.std()) < MIN_STD:
        return "The image is blank or a single flat colour, so there is no document to check."
    if sharpness(img_bgr) < MIN_SHARPNESS:
        return "The image is too blurry or out of focus to read. Retake it in good light, holding the camera steady."
    return None


def check_document(ocr: dict) -> str | None:
    """Checks on the OCR result. Returns a rejection reason, or None when it looks like a supported document."""
    if ocr["doc_type"] == "unknown" or ocr.get("evidence", 0) < MIN_EVIDENCE:
        return (f"This does not look like {SUPPORTED}: the expected header text and ID number were not found. "
                "Upload a clear, uncropped image of the document.")
    if ocr.get("n_text", 0) < MIN_TEXT_BOXES:
        return "Too little text could be read from this image. Upload a clearer, uncropped picture of the document."
    return None
