"""OCR-based consistency checks for Aadhaar / PAN / marksheet (rules that pixel forensics cannot express)."""
from __future__ import annotations

import collections
import datetime
import difflib
import re
import statistics

import cv2
import numpy as np

from indian_data import verhoeff_valid

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
PAN_TYPES = "PCHFATBLJG"
_TO_LETTER = str.maketrans("01245678", "OIZASGTB")  # digit -> look-alike letter (PAN letter positions)
_TO_DIGIT = str.maketrans("OQDILZASGTB", "00011245678")  # letter -> look-alike digit (PAN digit positions)
HARD_FLAGS = {"aadhaar_checksum_fail", "invalid_pan_format", "pan_surname_mismatch", "total_mismatch",
              "percentage_mismatch", "result_mismatch"}
_reader = None


def get_reader():
    """easyocr.Reader, created once. Hindi is added when its model is available; GPU is used when CUDA exists."""
    global _reader
    if _reader is None:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        try:
            _reader = easyocr.Reader(["en", "hi"], gpu=gpu, verbose=False)
        except Exception as e:  # model download failed / unsupported
            print(f"WARNING: Hindi OCR unavailable ({type(e).__name__}); using English only")
            _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _reader


def _norm(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS)


def _read(img_bgr: np.ndarray) -> list[dict]:
    out = []
    for pts, text, conf in get_reader().readtext(img_bgr):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        t = _norm(text).strip()
        if t:
            out.append({"text": t, "conf": float(conf), "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                        "h": max(ys) - min(ys)})
    return sorted(out, key=lambda b: (b["y0"], b["x0"]))


def _rows(boxes: list[dict]) -> list[list[dict]]:
    """Group OCR boxes into text rows by y-centre."""
    if not boxes:
        return []
    tol = max(8.0, 0.5 * statistics.median(b["h"] for b in boxes))
    rows: list[list[dict]] = []
    for b in sorted(boxes, key=lambda b: (b["y0"] + b["y1"]) / 2):
        yc = (b["y0"] + b["y1"]) / 2
        if rows and abs(yc - np.mean([(r["y0"] + r["y1"]) / 2 for r in rows[-1]])) <= tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b["x0"]) for r in rows]


# ---------------------------------------------------------------- doc type
def detect_doc_type(text_lines: list[str]) -> str:
    t = " ".join(_norm(x) for x in text_lines).lower()
    compact = re.sub(r"\s+", "", t.upper())
    score = {
        "aadhaar": sum(k in t for k in ("government of india", "aadhaar", "aadhar", "unique identification"))
        + bool(re.search(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)", t)),
        "pan": sum(k in t for k in ("income tax", "permanent account")) + bool(re.search(r"[A-Z]{5}\d{4}[A-Z]", compact)),
        "marksheet": sum(k in t for k in ("marks", "roll", "total", "result")),
    }
    best = max(score, key=score.get)
    if score[best] == 0 or (best == "marksheet" and score[best] < 2):
        return "aadhaar" if score["aadhaar"] and score["aadhaar"] >= score["pan"] else ("pan" if score["pan"] else "unknown")
    return best


# ---------------------------------------------------------------- helpers
def _valid_dob(text: str) -> bool:
    for d, m, y in re.findall(r"(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)", text):
        try:
            dob = datetime.date(int(y), int(m), int(d))
        except ValueError:
            continue
        age = (datetime.date.today() - dob).days / 365.25
        if 0 <= age <= 120:
            return True
    return False


def _has_qr(img_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        det = cv2.QRCodeDetector()
        for g in (gray, cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):
            ok, _ = det.detect(g)
            if ok:
                return True
    except cv2.error:
        pass
    # fallback 1: a roughly square block of dense high-frequency texture (dense secure QR codes defeat the detectors)
    try:
        if cv2.QRCodeDetectorAruco().detect(gray)[0]:
            return True
    except cv2.error:
        pass
    edges = (np.abs(cv2.Laplacian(gray, cv2.CV_32F)) > 60).astype(np.float32)
    n, _, st, _ = cv2.connectedComponentsWithStats((cv2.blur(edges, (25, 25)) > 0.25).astype(np.uint8))
    if any(w > 0.1 * gray.shape[1] and h > 0.1 * gray.shape[1] and 0.8 < w / h < 1.25 for _, _, w, h, _ in st[1:n]):
        return True
    # fallback 2: dense bi-level block (black and white, almost no mid-grey) = QR-like
    h, w = gray.shape
    win, hits = max(32, min(h, w) // 8), 0
    for y in range(0, h - win, win // 2):
        for x in range(0, w - win, win // 2):
            p = gray[y:y + win, x:x + win]
            black, white, mid = (p < 80).mean(), (p > 180).mean(), ((p >= 80) & (p <= 180)).mean()
            hits += 0.3 <= black <= 0.7 and white >= 0.3 and mid < 0.15
    return hits >= 3


# ---------------------------------------------------------------- per-document rules
def _check_aadhaar(boxes, text, img_bgr, flags):
    num = None
    # not the tail of a 16-digit VID ("9154 2551 2829 5111"), which would otherwise match as a 12-digit number
    pat = r"(?<!\d)(?<!\d{4}\s)\d{4}\s?\d{4}\s?\d{4}(?!\s?\d)"
    cands = [t for b in boxes for t in re.findall(pat, b["text"])]
    for row in _rows(boxes):  # number split over several boxes
        cands += re.findall(pat, " ".join(b["text"] for b in row))
    if cands:  # the number is printed several times on a letter; the most common read wins (ties: a valid one)
        counts = collections.Counter(re.sub(r"\s", "", c) for c in cands)
        num = max(counts, key=lambda n: (counts[n], verhoeff_valid(n)))
    if num is None:
        flags.append("missing_id")
    else:
        if num[0] not in "23456789":
            flags.append("invalid_aadhaar_prefix")
        if not verhoeff_valid(num):
            flags.append("aadhaar_checksum_fail")
    if not _valid_dob(text):
        flags.append("invalid_dob")
    if not re.search(r"\b(male|female|transgender)\b", text, re.I):
        flags.append("invalid_gender")
    if not _has_qr(img_bgr):
        flags.append("missing_qr")


def _find_pan(boxes) -> str | None:
    toks = [re.sub(r"[^A-Za-z0-9]", "", b["text"]).upper() for b in boxes]
    toks = [t for t in toks if len(t) == 10]
    for t in toks:
        if PAN_RE.match(t):
            return t
    for t in toks:  # repair look-alike confusions (0/O, 1/I, 5/S, 8/B) by position
        fixed = t[:5].translate(_TO_LETTER) + t[5:9].translate(_TO_DIGIT) + t[9].translate(_TO_LETTER)
        if PAN_RE.match(fixed):
            return fixed
    return None


def _holder_name(boxes) -> str | None:
    for i, b in enumerate(boxes):
        t = b["text"].strip().lower().rstrip(":")
        if t.startswith("name") and "father" not in t:
            yc = (b["y0"] + b["y1"]) / 2
            for c in boxes[i + 1:]:
                if (c["y0"] + c["y1"]) / 2 > yc + 5 and abs(c["x0"] - b["x0"]) < 60:
                    return c["text"]
    return None


def _check_pan(boxes, text, flags):
    pan = _find_pan(boxes)
    if pan is None:
        flags.append("invalid_pan_format")
    else:
        if pan[3] not in PAN_TYPES:
            flags.append("invalid_pan_type")
        elif pan[3] == "P":
            name = _holder_name(boxes)
            if name and name.split()[-1][0].upper() != pan[4]:
                flags.append("pan_surname_mismatch")
    if not _valid_dob(text):
        flags.append("invalid_dob")
    lines = [b["text"].upper() for b in boxes]
    if not ("INCOME TAX" in " ".join(lines) or any(
            difflib.SequenceMatcher(None, l, "INCOME TAX DEPARTMENT").ratio() > 0.8 for l in lines)):
        flags.append("missing_header")


def parse_marksheet(boxes) -> dict:
    subjects, total, max_total = [], None, None
    rows = _rows(boxes)
    start = next((i for i, r in enumerate(rows) if re.search(r"subject|maximum", " ".join(b["text"] for b in r), re.I)), -1)
    for row in rows[start + 1:]:  # only rows below the table header
        txt = " ".join(b["text"] for b in row).strip(" :")
        m = re.match(r"^(.*?[A-Za-z].*?)\s+(\d{1,3})\s+(\d{1,3})$", txt)
        if not m:
            continue
        label, mx, ob = m.group(1), int(m.group(2)), int(m.group(3))
        if label.lower().startswith("total"):
            max_total, total = mx, ob
        else:
            subjects.append((label, mx, ob))
    t = " ".join(b["text"] for b in boxes)
    pct = re.search(r"(\d{1,3}(?:\.\d{1,2})?)\s*%", t)
    res = re.search(r"\b(PASS|FAIL)\b", t.upper())
    return {"subjects": subjects, "total": total, "max_total": max_total,
            "percent": float(pct.group(1)) if pct else None, "result": res.group(1) if res else None}


def _check_marksheet(boxes, flags):
    d = parse_marksheet(boxes)
    subj = d["subjects"]
    if subj and any(ob > mx for _, mx, ob in subj):
        flags.append("marks_exceed_max")
    if subj and d["total"] is not None and sum(ob for _, _, ob in subj) != d["total"]:
        flags.append("total_mismatch")
    if d["total"] is not None and d["percent"] is not None:
        mx = d["max_total"] or sum(m for _, m, _ in subj)
        if mx and abs(d["total"] / mx * 100 - d["percent"]) > 0.5:
            flags.append("percentage_mismatch")
    if subj and d["result"]:
        expected = "FAIL" if min(ob for _, _, ob in subj) < 33 else "PASS"
        if expected != d["result"]:
            flags.append("result_mismatch")


def _font_inconsistent(boxes) -> bool:
    for row in _rows([b for b in boxes if b["conf"] > 0.3 and len(b["text"]) >= 2]):
        if len(row) >= 3:
            med = statistics.median(b["h"] for b in row)
            if med > 0 and any(abs(b["h"] - med) / med > 0.35 for b in row):
                return True
    return False


# ---------------------------------------------------------------- public API
def _evaluate(img_bgr: np.ndarray) -> tuple[str, list[str], list[str], list[dict]]:
    boxes = _read(img_bgr)
    lines = [b["text"] for b in boxes]
    text = " ".join(lines)
    doc_type = detect_doc_type(lines)
    flags: list[str] = []
    if doc_type == "aadhaar":
        _check_aadhaar(boxes, text, img_bgr, flags)
    elif doc_type == "pan":
        _check_pan(boxes, text, flags)
    elif doc_type == "marksheet":
        _check_marksheet(boxes, flags)
    if _font_inconsistent(boxes):
        flags.append("font_inconsistency")
    if boxes and np.mean([b["conf"] for b in boxes]) < 0.5:
        flags.append("low_ocr_confidence")
    return doc_type, flags, lines, boxes


def ocr_check(img_bgr: np.ndarray) -> dict:
    """OCR rule checks. A hard-fail rule only stands if a second read (2x upscale) reproduces it, because a
    genuine inconsistency persists across reads while an OCR misread (e.g. '97' -> '997') usually does not."""
    doc_type, flags, lines, _ = _evaluate(img_bgr)
    if HARD_FLAGS & set(flags):
        big = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, flags2, lines2, _ = _evaluate(big)
        if HARD_FLAGS & set(flags) & set(flags2):
            flags = [f for f in flags if f not in HARD_FLAGS or f in flags2]
        else:
            flags, lines = flags2, lines2
    hard = bool(HARD_FLAGS & set(flags))
    score = 1.0 if hard else min(1.0, len(flags) / 3)
    return {"doc_type": doc_type, "flags": flags, "hard_fail": hard, "ocr_score": score, "text": lines}
