"""Create one tampered copy (+ binary mask) per genuine image, then recompress JPEGs.

Order matters: all tampering reads the pristine q=98 genuine files first; only afterwards are ALL genuine files
recompressed with the same q 60-95 distribution, so compression history can't separate the classes.
Run make_genuine.py again before re-running this script.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from common import GENUINE, MASKS, META, SEED, TAMPERED, DATA, DOC_TYPES, load_font, make_face
from indian_data import generate_aadhaar, generate_pan, random_dob, random_full_name

PAD = 3
EDITABLE = {"aadhaar": ("name", "dob", "aadhaar"), "pan": ("name", "father", "dob", "pan"),
            "marksheet": ("name", "father", "dob", "roll")}
APPLICABLE = {"aadhaar": ["text_edit", "photo_swap", "copy_move", "splice"],
              "pan": ["text_edit", "photo_swap", "copy_move", "splice"],
              "marksheet": ["text_edit", "copy_move", "splice", "marks_edit_naive", "marks_edit_full"]}
DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")


# ---------------------------------------------------------------- helpers
def clip_box(box, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return [max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))]


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def local_bg(arr, box, ring=5):
    x0, y0, x1, y1 = box
    h, w = arr.shape[:2]
    ox0, oy0, ox1, oy1 = max(0, x0 - ring), max(0, y0 - ring), min(w, x1 + ring), min(h, y1 + ring)
    crop = arr[oy0:oy1, ox0:ox1]
    m = np.ones(crop.shape[:2], bool)
    m[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
    return np.median(crop[m], axis=0)


def new_value(kind: str, old: str, rng: random.Random) -> str:
    for _ in range(20):
        if kind == "name":
            v = random_full_name(rng)
        elif kind == "father":
            v = random_full_name(rng, "Male")
        elif kind == "dob":
            v = DATE_RE.sub(random_dob(rng), old)
        elif kind == "aadhaar":
            v = generate_aadhaar(rng)
        elif kind == "pan":
            v = generate_pan(rng, random_full_name(rng).split()[1])
        elif kind == "roll":
            v = str(rng.randint(10 ** (len(old) - 1), 10 ** len(old) - 1))
        else:
            raise ValueError(kind)
        if kind in ("name", "father") and old.isupper():
            v = v.upper()
        if v != old:
            return v
    raise RuntimeError("could not generate a new value")


def redraw(img, fld, text, rng, nprng, mask, noise):
    """Erase the old field with the local background colour and draw `text` in a similar font."""
    arr = np.array(img)
    size = fld["size"] + rng.choice([-1, 0, 0, 1])
    font = load_font(fld["font"], size)
    x, y = fld["xy"]
    l, t, r, b = font.getbbox(text, anchor="la")
    if fld.get("cx") is not None:
        x = round(fld["cx"] - (r - l) / 2 - l)
    nb = [x + l, y + t, x + r, y + b]
    u = clip_box([v + s * PAD for v, s in zip(union(fld["bbox"], nb), (-1, -1, 1, 1))], arr.shape)
    fill = local_bg(arr, u)
    sigma = noise if rng.random() < 0.5 else 0.0
    patch = fill + (nprng.normal(0, sigma, (u[3] - u[1], u[2] - u[0], 1)) if sigma else 0)
    arr[u[1]:u[3], u[0]:u[2]] = np.clip(patch, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).text((x, y), text, font=font, fill=tuple(fld["fill"]), anchor="la")
    mask[u[1]:u[3], u[0]:u[2]] = 255
    fld.update(text=text, bbox=nb, size=size, xy=[x, y])
    return img


# ---------------------------------------------------------------- tamper operations
def text_edit(img, meta, rng, nprng, mask, **_):
    kinds = EDITABLE[meta["doc_type"]]
    key = rng.choice([k for k, f in meta["fields"].items() if f["kind"] in kinds])
    f = meta["fields"][key]
    return redraw(img, f, new_value(f["kind"], f["text"], rng), rng, nprng, mask, meta["noise"])


def photo_swap(img, meta, rng, nprng, mask, **_):
    x0, y0, x1, y1 = meta["photo"]
    img = img.copy()
    img.paste(make_face(x1 - x0, y1 - y0, rng), (x0, y0))
    mask[y0:y1, x0:x1] = 255
    return img


def copy_move(img, meta, rng, nprng, mask, **_):
    arr = np.array(img)
    h, w = arr.shape[:2]
    for _ in range(200):
        bw, bh = rng.randint(40, 120), rng.randint(30, 90)
        sx, sy = rng.randint(0, w - bw), rng.randint(0, h - bh)
        if arr[sy:sy + bh, sx:sx + bw].std() < 15:  # skip blank paper
            continue
        tx, ty = rng.randint(0, w - bw), rng.randint(0, h - bh)
        if abs(tx - sx) < bw + 4 and abs(ty - sy) < bh + 4:
            continue
        arr[ty:ty + bh, tx:tx + bw] = arr[sy:sy + bh, sx:sx + bw].copy()
        mask[ty:ty + bh, tx:tx + bw] = 255
        return Image.fromarray(arr)
    raise RuntimeError("copy_move: no valid region")


def splice(img, meta, rng, nprng, mask, donors, **_):
    donor_stem = rng.choice([s for s in donors if s != meta["stem"]])
    donor = np.array(Image.open(GENUINE / f"{donor_stem}.jpg").convert("RGB"))
    dmeta = json.loads((META / f"{donor_stem}.json").read_text())
    arr = np.array(img)
    if rng.random() < 0.5:  # paste the donor's version of a text/photo field in the same place
        keys = [k for k, f in meta["fields"].items() if f["kind"] in EDITABLE[meta["doc_type"]]]
        key = rng.choice(keys)
        box = union(meta["fields"][key]["bbox"], dmeta["fields"][key]["bbox"])
        box = clip_box([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], arr.shape)
    else:  # arbitrary rectangle from the donor
        h, w = arr.shape[:2]
        bw, bh = rng.randint(60, 200), rng.randint(30, 100)
        x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
        box = [x, y, x + bw, y + bh]
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = donor[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] = 255
    return Image.fromarray(arr)


def _marks_edit(img, meta, rng, nprng, mask, full):
    f = meta["fields"]
    i = rng.randrange(6)
    old = int(f[f"marks_{i}"]["text"])
    new = rng.choice([m for m in range(35, 99) if abs(m - old) >= 8])
    img = redraw(img, f[f"marks_{i}"], str(new), rng, nprng, mask, meta["noise"])
    if full:
        marks = [int(f[f"marks_{j}"]["text"]) for j in range(6)]
        total = sum(marks)
        img = redraw(img, f["total"], str(total), rng, nprng, mask, meta["noise"])
        img = redraw(img, f["percent"], f"{total / 600 * 100:.2f}%", rng, nprng, mask, meta["noise"])
        result = "FAIL" if min(marks) < 33 else "PASS"
        if result != f["result"]["text"]:
            img = redraw(img, f["result"], result, rng, nprng, mask, meta["noise"])
    return img


def marks_edit_naive(img, meta, rng, nprng, mask, **_):
    return _marks_edit(img, meta, rng, nprng, mask, full=False)


def marks_edit_full(img, meta, rng, nprng, mask, **_):
    return _marks_edit(img, meta, rng, nprng, mask, full=True)


OPS = {"text_edit": text_edit, "photo_swap": photo_swap, "copy_move": copy_move, "splice": splice,
       "marks_edit_naive": marks_edit_naive, "marks_edit_full": marks_edit_full}


def tamper_one(stem, meta, rng, nprng, donors):
    """Returns (tamper_type, image, mask); retries until the mask is non-empty and covers every changed pixel."""
    orig = Image.open(GENUINE / f"{stem}.jpg").convert("RGB")
    ref = np.asarray(orig).astype(np.int16)
    for _ in range(10):
        ttype = rng.choice(APPLICABLE[meta["doc_type"]])
        m = json.loads(json.dumps(meta))  # fresh copy: ops mutate field boxes
        mask = np.zeros((orig.height, orig.width), np.uint8)
        try:
            out = OPS[ttype](orig, m, rng, nprng, mask, donors=donors)
        except RuntimeError:
            continue
        diff = np.abs(np.asarray(out).astype(np.int16) - ref).sum(2)
        if mask.any() and diff[mask == 0].max() == 0 and diff.any():
            return ttype, out, mask
    raise RuntimeError(f"could not tamper {stem}")


def main() -> None:
    stems = sorted(p.stem for p in GENUINE.glob("*.jpg"))
    metas = {s: json.loads((META / f"{s}.json").read_text()) for s in stems}
    if not stems or any("recompressed_q" in m for m in metas.values()):
        sys.exit("No pristine genuine images found - run make_genuine.py first (it was already recompressed).")
    TAMPERED.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    for old in list(TAMPERED.glob("*.jpg")) + list(MASKS.glob("*.png")):
        old.unlink()
    rng = random.Random(SEED + 1)
    nprng = np.random.default_rng(SEED + 1)
    donors = {dt: [s for s in stems if metas[s]["doc_type"] == dt] for dt in DOC_TYPES}
    rows = []
    for stem in tqdm(stems, desc="tamper"):
        ttype, img, mask = tamper_one(stem, metas[stem], rng, nprng, donors[metas[stem]["doc_type"]])
        q = rng.randint(60, 95)
        img.save(TAMPERED / f"{stem}.jpg", quality=q)
        Image.fromarray(mask).save(MASKS / f"{stem}.png")  # PNG: JPEG would corrupt the binary mask
        rows.append({"stem": stem, "tamper_type": ttype, "jpeg_quality": q})
    with open(DATA / "tamper_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "tamper_type", "jpeg_quality"])
        w.writeheader()
        w.writerows(rows)
    for stem in tqdm(stems, desc="recompress genuine"):  # every genuine image gets the same quality distribution
        q = rng.randint(60, 95)
        Image.open(GENUINE / f"{stem}.jpg").convert("RGB").save(GENUINE / f"{stem}.jpg", quality=q)
        metas[stem]["recompressed_q"] = q
        (META / f"{stem}.json").write_text(json.dumps(metas[stem]))


if __name__ == "__main__":
    main()
