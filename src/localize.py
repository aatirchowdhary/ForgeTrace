"""Training-free, per-text-box tamper localisation.

Edited text is usually re-rendered: its ink is cleaner / darker and its background patch is flatter than the text that
was left alone, which still carries the original scan / compression blur. For every OCR text box we measure a few
ink and background statistics and compare them with boxes of similar size and script in the same image (robust
z-score). Because the comparison is always against the document itself, it does not depend on the resolution or
compression history of the upload, which is what breaks the synthetic-trained U-Net on real documents.
"""
from __future__ import annotations

import cv2
import numpy as np

from ocr_check import _read

Z_FLAG = 4.0       # a box is reported as suspicious at or above this robust z-score
MIN_PEERS = 5      # fewer peers than this -> compare against every box in the image
K_NEAR = 10        # peers = this many nearest boxes of the same script and similar size
MIN_BOX_PX = 60    # ignore boxes smaller than this (too little signal)
# Verdict from the number of flagged fields (confident OCR reads only). Calibrated on a single genuine/forged pair
# (3 vs 12 flagged fields), so these are placeholders to re-tune once more real examples exist.
MIN_CONF, REVIEW_AT, TAMPERED_AT = 0.3, 4, 6
FEATURES = (("ink_dark", 4.0), ("sharp", 0.12), ("flat", 0.06))  # (feature, MAD floor)


def _is_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.8


def _box_features(gray: np.ndarray, sat: np.ndarray, b: dict) -> dict | None:
    pad = 3
    sl = (slice(max(0, int(b["y0"]) - pad), int(b["y1"]) + pad), slice(max(0, int(b["x0"]) - pad), int(b["x1"]) + pad))
    c, sc = gray[sl], sat[sl]
    if c.shape[0] < 8 or c.shape[1] < 20:
        return None
    ink, bg = c < 128, c > 200
    if ink.sum() < 10 or bg.sum() < 30 or bg.mean() < 0.5 or np.median(c[bg]) < 235:
        return None  # only plain dark-on-white text is comparable (coloured banners, logos and photos are skipped)
    if (sc[ink] > 80).mean() > 0.2:
        return None  # coloured (e.g. red) headings have a different ink model than black body text
    return {"ink_dark": float(np.percentile(c[ink], 10)),          # how black the ink core is (0 = pure black)
            "sharp": float(np.log1p(((c >= 60) & (c <= 200)).mean() / max(ink.mean(), 1e-3))),  # soft edge pixels per ink pixel
            "flat": float((c >= 253).mean())}                       # share of pure-white pixels (clean re-drawn patch)


def _robust_z(v: float, peers: np.ndarray, floor: float) -> float:
    med = float(np.median(peers))
    mad = max(1.4826 * float(np.median(np.abs(peers - med))), floor)
    return (v - med) / mad


def field_anomalies(img_bgr: np.ndarray, boxes: list[dict] | None = None) -> list[dict]:
    """One dict per usable OCR box: its text, bbox (x0, y0, x1, y1), anomaly score and the dominating feature."""
    boxes = _read(img_bgr) if boxes is None else boxes
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    items = []
    for b in boxes:
        f = _box_features(gray, sat, b)
        if f and b["h"] >= 6 and (b["x1"] - b["x0"]) * b["h"] >= MIN_BOX_PX:
            items.append({"text": b["text"], "bbox": (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])),
                          "h": b["h"], "conf": b["conf"], "latin": _is_latin(b["text"]), "f": f})
    for it in items:
        cx, cy = (it["bbox"][0] + it["bbox"][2]) / 2, (it["bbox"][1] + it["bbox"][3]) / 2
        near = sorted((o for o in items if o is not it and o["latin"] == it["latin"] and 0.7 <= o["h"] / it["h"] <= 1.4),
                      key=lambda o: ((o["bbox"][0] + o["bbox"][2]) / 2 - cx) ** 2 + ((o["bbox"][1] + o["bbox"][3]) / 2 - cy) ** 2)
        peers = near[:K_NEAR]  # the text around it was (mostly) left alone and shares its scan / compression history
        if len(peers) < MIN_PEERS:
            peers = [o for o in items if o is not it]
        best = (0.0, "")
        if len(peers) >= MIN_PEERS:
            for name, floor in FEATURES:
                sz = _robust_z(it["f"][name], np.array([p["f"][name] for p in peers]), floor)
                if abs(sz) > best[0]:
                    best, it["signed"] = (abs(sz), name), sz
        it["score"], it["reason"] = best
    return items


def anomaly_map(shape: tuple, items: list[dict], z_full: float = 8.0) -> np.ndarray:
    """HxW float32 in [0, 1]: each box filled with min(score / z_full, 1) when it is flagged, else 0."""
    out = np.zeros(shape[:2], np.float32)
    for it in items:
        if it["score"] >= Z_FLAG:
            x0, y0, x1, y1 = it["bbox"]
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], min(it["score"] / z_full, 1.0))
    return out


def suspicious_fields(items: list[dict]) -> list[dict]:
    return sorted((i for i in items if i["score"] >= Z_FLAG and i["conf"] >= MIN_CONF), key=lambda i: -i["score"])


def field_verdict(n_suspicious: int) -> str:
    return "TAMPERED" if n_suspicious >= TAMPERED_AT else "REVIEW" if n_suspicious >= REVIEW_AT else "GENUINE"
