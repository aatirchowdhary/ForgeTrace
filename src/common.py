"""Shared paths, font lookup and text/face drawing helpers."""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GENUINE, TAMPERED, MASKS, META = (DATA / n for n in ("genuine", "tampered", "masks", "meta"))
SPLITS = ROOT / "splits"
SEED = 42
DOC_TYPES = ("aadhaar", "pan", "marksheet")
DOC_SIZE = {"aadhaar": (640, 404), "pan": (640, 404), "marksheet": (800, 1131)}
BASE_QUALITY = 98  # make_genuine writes this; make_tampered later recompresses


# ---------------------------------------------------------------- fonts
@lru_cache(maxsize=None)
def _font_index() -> dict[str, str]:
    home = Path.home()
    dirs = ["C:/Windows/Fonts", "/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts",
            "/System/Library/Fonts", str(home / ".fonts"), str(home / ".local/share/fonts")]
    idx: dict[str, str] = {}
    for d in map(Path, dirs):
        if d.is_dir():
            for p in d.rglob("*"):
                if p.suffix.lower() in (".ttf", ".ttc", ".otf"):
                    idx.setdefault(p.name.lower(), str(p))
    return idx


def find_font(*names: str) -> str | None:
    idx = _font_index()
    for n in names:
        if n.lower() in idx:
            return idx[n.lower()]
    return None


_FAMILIES = [("arial.ttf", "arialbd.ttf"), ("calibri.ttf", "calibrib.ttf"), ("verdana.ttf", "verdanab.ttf"),
             ("times.ttf", "timesbd.ttf"), ("dejavusans.ttf", "dejavusans-bold.ttf"),
             ("liberationsans-regular.ttf", "liberationsans-bold.ttf")]


def pick_family(rng: random.Random) -> tuple[str | None, str | None]:
    """(regular, bold) font paths; (None, None) means PIL's built-in default font."""
    avail = [(find_font(r), find_font(b)) for r, b in _FAMILIES]
    avail = [(r, b or r) for r, b in avail if r]
    return rng.choice(avail) if avail else (None, None)


def devanagari_font() -> str | None:
    p = find_font("nirmala.ttc", "mangal.ttf", "notosansdevanagari-regular.ttf", "lohit-devanagari.ttf")
    if p:
        return p
    return next((v for k, v in _font_index().items() if "devanagari" in k), None)


@lru_cache(maxsize=1024)
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def fit_size(text: str, path: str | None, size: int, max_w: int) -> int:
    while size > 10:
        l, _, r, _ = load_font(path, size).getbbox(text, anchor="la")
        if r - l <= max_w:
            break
        size -= 1
    return size


# ---------------------------------------------------------------- drawing
def draw_field(draw: ImageDraw.ImageDraw, fields: dict, key: str, text: str, x: int, y: int,
               font_path: str | None, size: int, fill: tuple, kind: str = "other",
               center_x: int | None = None) -> dict:
    """Draw left-anchored text (or centred on center_x) and record everything needed to redraw it later."""
    font = load_font(font_path, size)
    l, t, r, b = font.getbbox(text, anchor="la")
    if center_x is not None:
        x = round(center_x - (r - l) / 2 - l)
    draw.text((x, y), text, font=font, fill=fill, anchor="la")
    fields[key] = {"text": text, "xy": [x, y], "bbox": [x + l, y + t, x + r, y + b], "font": font_path,
                   "size": size, "fill": list(fill), "kind": kind, "cx": center_x}
    return fields[key]


def tint(base: tuple, rng: random.Random, jitter: int = 8) -> tuple:
    return tuple(max(0, min(255, c + rng.randint(-jitter, jitter))) for c in base)


def add_noise(img: Image.Image, sigma: float, nprng: np.random.Generator) -> Image.Image:
    a = np.asarray(img).astype(np.float32) + nprng.normal(0, sigma, (img.height, img.width, 1))
    a += nprng.normal(0, sigma * 0.3, (img.height, img.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_face(w: int, h: int, rng: random.Random) -> Image.Image:
    """Random face-like portrait (ellipses only)."""
    rc = lambda lo, hi: tuple(rng.randint(lo, hi) for _ in range(3))
    im = Image.new("RGB", (w, h), rc(140, 235))
    d = ImageDraw.Draw(im)
    skin = rng.choice([(224, 172, 105), (198, 134, 66), (141, 85, 36), (241, 194, 125), (255, 219, 172), (172, 112, 60)])
    d.ellipse([w * 0.05, h * 0.68, w * 0.95, h * 1.35], fill=rc(20, 200))  # shoulders
    hw, hh = w * rng.uniform(0.42, 0.52), h * rng.uniform(0.48, 0.58)
    cx, cy = w / 2 + rng.uniform(-3, 3), h * 0.42
    d.rectangle([cx - hw * 0.12, cy, cx + hw * 0.12, h * 0.75], fill=skin)  # neck
    d.ellipse([cx - hw / 2, cy - hh / 2, cx + hw / 2, cy + hh / 2], fill=skin)
    d.chord([cx - hw / 2 - 2, cy - hh / 2 - 5, cx + hw / 2 + 2, cy + hh * 0.1], 180, 360, fill=rc(5, 70))  # hair
    ey = cy - hh * 0.04
    for s in (-1, 1):
        ex = cx + s * hw * 0.2
        d.ellipse([ex - 3, ey - 2, ex + 3, ey + 2], fill=(25, 20, 20))
    d.arc([cx - hw * 0.15, cy + hh * 0.1, cx + hw * 0.15, cy + hh * 0.3], 20, 160, fill=(120, 50, 50), width=2)
    return im.filter(ImageFilter.GaussianBlur(0.6))


def signature_curve(draw: ImageDraw.ImageDraw, box: tuple, rng: random.Random, fill=(25, 25, 90)) -> None:
    x0, y0, x1, y1 = box
    n = 60
    ph = [rng.uniform(0, 6.3) for _ in range(3)]
    f = [rng.uniform(2, 5), rng.uniform(6, 12), rng.uniform(1, 3)]
    pts = []
    for i in range(n):
        u = i / (n - 1)
        v = 0.5 + 0.28 * np.sin(f[0] * 6.28 * u + ph[0]) + 0.15 * np.sin(f[1] * 6.28 * u + ph[1]) * (1 - u)
        pts.append((x0 + 8 + u * (x1 - x0 - 16), y0 + (y1 - y0) * float(np.clip(v, 0.1, 0.9))))
    draw.line(pts, fill=fill, width=2, joint="curve")
    draw.line([(x0 + 10, y1 - 10), (x1 - 15, y1 - 14 - rng.randint(0, 6))], fill=fill, width=1)
