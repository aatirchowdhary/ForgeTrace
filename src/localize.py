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
INK_NEUTRAL = 30   # max spread between the mean ink R, G, B values for text to count as black / grey
K_NEAR = 20        # peers = this many nearest boxes of the same script and similar size
MIN_BOX_PX = 60    # ignore boxes smaller than this (too little signal)
# Verdict from the number of flagged fields (confident OCR reads only). Calibrated on three genuine/forged pairs
# (Aadhaar letter, SSC and HSC marksheets; every forgery was a re-typed name): genuine documents flag 0 fields, forgeries
# 1-3. Placeholders to re-tune on more real examples.
MIN_CONF, REVIEW_AT, TAMPERED_AT = 0.3, 1, 1  # REVIEW_AT == TAMPERED_AT: no "review" band until more data shows one is needed
# (feature, MAD floor, direction of "re-drawn"): re-typed text is crisper / darker / cleaner than the scan around it, so
# only deviations in that direction count (blurrier-than-neighbours is normal for titles, stamps and stylised fonts)
FEATURES = (("edge", 0.03, +1), ("sharp", 0.12, -1))


def _is_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.8


def _box_features(gray: np.ndarray, bgr: np.ndarray, b: dict) -> dict | None:
    """Background-independent edge statistics, plus ink darkness when the box sits on plain white paper."""
    pad = 4
    sl = (slice(max(0, int(b["y0"]) - pad), int(b["y1"]) + pad), slice(max(0, int(b["x0"]) - pad), int(b["x1"]) + pad))
    c, col = gray[sl], bgr[sl].astype(np.float32)
    if c.shape[0] < 10 or c.shape[1] < 30:
        return None
    lo, hi = np.percentile(c, 3), np.percentile(c, 75)  # ink level, paper level
    if hi - lo < 40:
        return None
    t = (c - lo) / (hi - lo)                              # 0 = ink, 1 = paper, whatever the background colour
    ink, edge_px = t < 0.25, (t >= 0.25) & (t <= 0.75)
    if ink.sum() < 20 or edge_px.sum() < 10 or np.ptp(col[ink].mean(axis=0)) > INK_NEUTRAL:
        return None  # too little text, or coloured ink (red / blue headings, stylised titles): only neutral black-grey body text is comparable
    gm = np.hypot(cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1))
    f = {"edge": float(gm[edge_px].mean() / (hi - lo)),                         # edge gradient / contrast: crisp = high
         "sharp": float(np.log1p(edge_px.mean() / ink.mean()))}                  # soft edge pixels per ink pixel: crisp = low
    if np.median(c[c > 200]) >= 235 if (c > 200).sum() > 30 else False:        # plain white paper only
        f["ink_dark"] = float(np.percentile(c[c < 128], 10)) if (c < 128).sum() >= 10 else 255.0
        f["flat"] = float((c >= 253).mean())                                    # pure-white share: clean re-drawn patch
    return f


def _word_segments(gray: np.ndarray, b: dict) -> list[dict]:
    """Split an OCR box into word-level pieces at wide ink-free gaps. OCR often returns one box for 'label: value', and
    an edited value next to an untouched label would be averaged away at box level."""
    x0, x1, y0, y1 = int(b["x0"]), int(b["x1"]), int(b["y0"]), int(b["y1"])
    c = gray[max(0, y0):y1, max(0, x0):x1]
    if c.shape[0] < 8 or c.shape[1] < 40:
        return []
    lo, hi = np.percentile(c, 3), np.percentile(c, 75)
    if hi - lo < 40:
        return []
    cols = (c < (lo + hi) / 2).sum(axis=0) > 0
    gap, segs, start, run = max(4, int(0.25 * b["h"])), [], None, 0
    for i, on in enumerate(list(cols) + [False] * (gap + 1)):
        if on:
            start, run = (i if start is None else start), 0
        elif start is not None:
            run += 1
            if run >= gap:
                segs.append((start, i - run + 1))
                start, run = None, 0
    out = []
    for a, e in segs:
        if e - a >= 1.5 * b["h"]:  # shorter pieces carry too little text to measure
            out.append({**b, "x0": max(0, x0) + a, "x1": max(0, x0) + e})
    return out if len(out) > 1 else []


def _robust_z(v: float, peers: np.ndarray, floor: float) -> float:
    med = float(np.median(peers))
    mad = max(1.4826 * float(np.median(np.abs(peers - med))), floor)
    return (v - med) / mad


def field_anomalies(img_bgr: np.ndarray, boxes: list[dict] | None = None) -> list[dict]:
    """One dict per usable OCR box: its text, bbox (x0, y0, x1, y1), anomaly score and the dominating feature."""
    boxes = _read(img_bgr) if boxes is None else boxes
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    items = []
    for b in boxes:
        f = _box_features(gray, img_bgr, b)
        if f and b["h"] >= 6 and (b["x1"] - b["x0"]) * b["h"] >= MIN_BOX_PX:
            items.append({"text": b["text"], "bbox": (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])),
                          "h": b["h"], "conf": b["conf"], "latin": _is_latin(b["text"]), "f": f})
    segs = []
    for it, b in zip(list(items), [b for b in boxes if _box_features(gray, img_bgr, b) and b["h"] >= 6 and (b["x1"] - b["x0"]) * b["h"] >= MIN_BOX_PX]):
        for sb in _word_segments(gray, b):
            f = _box_features(gray, img_bgr, sb)
            if f:
                segs.append({"text": b["text"], "bbox": (int(sb["x0"]), int(sb["y0"]), int(sb["x1"]), int(sb["y1"])),
                             "h": b["h"], "conf": b["conf"], "latin": it["latin"], "f": f, "parent": it})
    for it in items + segs:
        cx, cy = (it["bbox"][0] + it["bbox"][2]) / 2, (it["bbox"][1] + it["bbox"][3]) / 2
        base = it.get("parent", it)
        near = sorted((o for o in items if o is not base and o["latin"] == it["latin"] and 0.7 <= o["h"] / it["h"] <= 1.4),
                      key=lambda o: ((o["bbox"][0] + o["bbox"][2]) / 2 - cx) ** 2 + ((o["bbox"][1] + o["bbox"][3]) / 2 - cy) ** 2)
        peers = near[:K_NEAR]  # the text around it was (mostly) left alone and shares its scan / compression history
        if len(peers) < MIN_PEERS:
            peers = [o for o in items if o is not base]
        best = (0.0, "")
        if len(peers) >= MIN_PEERS:
            for name, floor, sign in FEATURES:
                pv = np.array([p["f"][name] for p in peers if name in p["f"]])
                if name not in it["f"] or len(pv) < MIN_PEERS:
                    continue
                sz = sign * _robust_z(it["f"][name], pv, floor)
                if sz <= 0:
                    continue
                if abs(sz) > best[0]:
                    best, it["signed"] = (abs(sz), name), sz
        it["score"], it["reason"] = best
    flagged_parents = {id(sg["parent"]) for sg in segs if sg["score"] >= Z_FLAG}
    # a flagged word segment replaces its whole-box item, so the highlight lands on the edited words only
    return [i for i in items if id(i) not in flagged_parents] + [sg for sg in segs if sg["score"] >= Z_FLAG]


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
