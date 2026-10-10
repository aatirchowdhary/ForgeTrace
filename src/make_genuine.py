"""Generate fictional genuine documents: data/genuine/{doc_type}_{id}.jpg (+ field metadata in data/meta/)."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from common import (BASE_QUALITY, DOC_SIZE, DOC_TYPES, GENUINE, META, SEED, add_noise, devanagari_font,
                    draw_field, fit_size, load_font, make_face, pick_family, signature_curve, tint)
from indian_data import generate_aadhaar, generate_pan, make_person, STATES, SURNAMES

NAVY = (20, 30, 90)


def ink(rng: random.Random) -> tuple:
    g = rng.randint(10, 45)
    return (g, g, g + rng.randint(0, 15))


def make_qr(size: int, rng: random.Random) -> Image.Image:
    data = "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(60, 120)))
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=4, border=1)
        qr.add_data(data)
        return qr.make_image(fill_color="black", back_color="white").convert("RGB").resize((size, size), Image.NEAREST)
    except ImportError:
        n = 29
        cell = size // n
        im = Image.new("RGB", (cell * n, cell * n), "white")
        d = ImageDraw.Draw(im)
        for i in range(n):
            for j in range(n):
                if rng.random() < 0.5:
                    d.rectangle([i * cell, j * cell, (i + 1) * cell - 1, (j + 1) * cell - 1], fill="black")
        return im.resize((size, size), Image.NEAREST)


def photo_box(img: Image.Image, box: tuple, rng: random.Random) -> list:
    x0, y0, x1, y1 = box
    img.paste(make_face(x1 - x0, y1 - y0, rng), (x0, y0))
    ImageDraw.Draw(img).rectangle(box, outline=(60, 60, 60), width=1)
    return list(box)


def finish(img: Image.Image, nprng: np.random.Generator, rng: random.Random) -> tuple[Image.Image, float]:
    sigma = round(rng.uniform(1.5, 4.0), 2)
    return add_noise(img, sigma, nprng), sigma


# ---------------------------------------------------------------- aadhaar
def render_aadhaar(rng, nprng):
    W, H = DOC_SIZE["aadhaar"]
    p = make_person(rng)
    reg, bold = pick_family(rng)
    sz = rng.randint(-1, 2)
    img = Image.new("RGB", (W, H), tint((252, 250, 243), rng, 6))
    d = ImageDraw.Draw(img)
    f: dict = {}
    hs = 22
    for i, c in enumerate([(255, 153, 51), (255, 255, 255), (19, 136, 8)]):
        d.rectangle([0, i * hs, W, (i + 1) * hs], fill=c)
    hindi = devanagari_font()
    if hindi:
        d.text((W / 2, hs * 0.5), "भारत सरकार", font=load_font(hindi, 16), fill=NAVY, anchor="mm")
    d.text((W / 2, hs * 1.5), "Government of India", font=load_font(bold, 16 + sz // 2), fill=NAVY, anchor="mm")
    d.text((W / 2, 90), "Unique Identification Authority of India", font=load_font(reg, 14), fill=(90, 90, 90), anchor="mm")
    photo = photo_box(img, (30, 110, 170, 280), rng)
    col = ink(rng)
    nm = p["name"]
    draw_field(d, f, "name", nm, 195, 115, bold, fit_size(nm, bold, 24 + sz, 285), col, "name")
    draw_field(d, f, "dob", f"DOB: {p['dob']}", 195, 160, reg, 20 + sz, col, "dob")
    draw_field(d, f, "gender", p["gender"], 195, 195, reg, 20 + sz, col, "other")
    addr = f"{p['city']}, {p['state']} - {p['pin']}"
    draw_field(d, f, "address", addr, 195, 235, reg, fit_size(addr, reg, 15, 425), (70, 70, 70), "other")
    img.paste(make_qr(115, rng), (495, 108))
    d.line([(30, 312), (W - 30, 312)], fill=(200, 60, 60), width=2)
    num = generate_aadhaar(rng)
    draw_field(d, f, "aadhaar", num, 0, 326, bold, 40 + sz, col, "aadhaar", center_x=W // 2)
    d.text((W / 2, 385), "Mera Aadhaar, Meri Pehchaan", font=load_font(reg, 13), fill=(110, 110, 110), anchor="mm")
    return img, f, photo, p


# ---------------------------------------------------------------- pan
def render_pan(rng, nprng):
    W, H = DOC_SIZE["pan"]
    p = make_person(rng)
    reg, bold = pick_family(rng)
    sz = rng.randint(-1, 2)
    img = Image.new("RGB", (W, H), tint((214, 229, 244), rng, 7))
    d = ImageDraw.Draw(img)
    f: dict = {}
    d.text((W / 2, 28), "INCOME TAX DEPARTMENT", font=load_font(bold, 24 + sz), fill=NAVY, anchor="mm")
    d.text((W / 2, 58), "GOVT. OF INDIA", font=load_font(bold, 17 + sz), fill=NAVY, anchor="mm")
    d.line([(20, 78), (W - 20, 78)], fill=NAVY, width=2)
    photo = photo_box(img, (30, 100, 170, 250), rng)
    d.rectangle((30, 262, 170, 322), outline=(60, 60, 60), fill=(240, 244, 250))
    signature_curve(d, (30, 262, 170, 322), rng)
    d.text((100, 334), "Signature", font=load_font(reg, 12), fill=(90, 90, 90), anchor="mm")
    lab, col = (80, 80, 110), ink(rng)
    small = load_font(reg, 13)
    for text, y in [("Permanent Account Number", 90), ("Name", 156), ("Father's Name", 212), ("Date of Birth", 268)]:
        d.text((195, y), text, font=small, fill=lab, anchor="la")
    draw_field(d, f, "pan", generate_pan(rng, p["surname"]), 195, 106, bold, 32 + sz, col, "pan")
    draw_field(d, f, "name", p["name"].upper(), 195, 173, bold, 22 + sz, col, "name")
    draw_field(d, f, "father", p["father"].upper(), 195, 229, bold, 22 + sz, col, "father")
    draw_field(d, f, "dob", p["dob"], 195, 285, bold, 22 + sz, col, "dob")
    return img, f, photo, p


# ---------------------------------------------------------------- marksheet
SUBJECTS = ["English", "Hindi", "Mathematics", "Science", "Social Science"]
LAST_SUBJECT = ["Computer Applications", "Sanskrit", "Information Technology"]


def render_marksheet(rng, nprng):
    W, H = DOC_SIZE["marksheet"]
    p = make_person(rng, 2004, 2010)
    reg, bold = pick_family(rng)
    sz = rng.randint(-1, 2)
    img = Image.new("RGB", (W, H), tint((250, 246, 232), rng, 6))
    d = ImageDraw.Draw(img)
    f: dict = {}
    d.rectangle([20, 20, W - 20, H - 20], outline=NAVY, width=3)
    d.rectangle([28, 28, W - 28, H - 28], outline=NAVY, width=1)
    board = f"{rng.choice(list(STATES))} Board of Secondary Education"
    d.text((W / 2, 75), board, font=load_font(bold, fit_size(board, bold, 28 + sz, 700)), fill=NAVY, anchor="mm")
    school = f"{rng.choice(SURNAMES)} Public School, {p['city']}"
    d.text((W / 2, 120), school, font=load_font(reg, fit_size(school, reg, 22 + sz, 700)), fill=(50, 50, 50), anchor="mm")
    d.line([(50, 150), (W - 50, 150)], fill=NAVY, width=2)
    d.text((W / 2, 190), "STATEMENT OF MARKS", font=load_font(bold, 30 + sz), fill=NAVY, anchor="mm")
    d.text((W / 2, 227), f"Secondary School Examination, {rng.randint(2019, 2025)}", font=load_font(reg, 20), fill=(60, 60, 60), anchor="mm")

    col = ink(rng)
    lab = load_font(reg, 20)
    roll = str(rng.randint(100000, 9999999))
    for label, key, text, kind, y in [("Name of Student", "name", p["name"], "name", 270),
                                      ("Father's Name", "father", p["father"], "father", 312),
                                      ("Roll Number", "roll", roll, "roll", 354),
                                      ("Date of Birth", "dob", p["dob"], "dob", 396)]:
        d.text((60, y), f"{label} :", font=lab, fill=(70, 70, 70), anchor="la")
        draw_field(d, f, key, text, 280, y - 2, bold, 22 + sz, col, kind)

    subjects = SUBJECTS + [rng.choice(LAST_SUBJECT)]
    marks = [rng.randint(35, 98) for _ in subjects]
    if rng.random() < 0.05:  # a few FAIL sheets so the result rule is actually exercised
        marks[rng.randrange(6)] = rng.randint(20, 32)
    xs, top, rh = [60, 440, 580, 740], 450, 52
    d.rectangle([xs[0], top, xs[-1], top + rh * 8], outline=NAVY, width=2)
    for i in range(1, 8):
        d.line([(xs[0], top + i * rh), (xs[-1], top + i * rh)], fill=NAVY, width=1)
    for x in xs[1:3]:
        d.line([(x, top), (x, top + rh * 8)], fill=NAVY, width=1)
    for c, t in enumerate(["Subject", "Maximum Marks", "Marks Obtained"]):
        d.text(((xs[c] + xs[c + 1]) / 2 if c else xs[0] + 90, top + rh / 2), t, font=load_font(bold, 18), fill=NAVY, anchor="mm")
    cell_font = load_font(reg, 21)
    for i, (s, m) in enumerate(zip(subjects, marks)):
        cy = top + rh * (i + 1)
        d.text((xs[0] + 15, cy + rh / 2), s, font=cell_font, fill=col, anchor="lm")
        d.text(((xs[1] + xs[2]) / 2, cy + rh / 2), "100", font=cell_font, fill=col, anchor="mm")
        draw_field(d, f, f"marks_{i}", str(m), 0, cy + 13, bold, 22 + sz, col, "marks", center_x=(xs[2] + xs[3]) // 2)
    cy = top + rh * 7
    d.text((xs[0] + 15, cy + rh / 2), "Total", font=load_font(bold, 21), fill=col, anchor="lm")
    d.text(((xs[1] + xs[2]) / 2, cy + rh / 2), "600", font=load_font(bold, 21), fill=col, anchor="mm")
    total = sum(marks)
    draw_field(d, f, "total", str(total), 0, cy + 13, bold, 22 + sz, col, "total", center_x=(xs[2] + xs[3]) // 2)
    d.text((60, 900), "Percentage :", font=lab, fill=(70, 70, 70), anchor="la")
    draw_field(d, f, "percent", f"{total / 600 * 100:.2f}%", 280, 898, bold, 22 + sz, col, "percent")
    d.text((60, 945), "Result :", font=lab, fill=(70, 70, 70), anchor="la")
    draw_field(d, f, "result", "FAIL" if min(marks) < 33 else "PASS", 280, 943, bold, 22 + sz, col, "result")
    # seal + signature
    d.ellipse([90, 1000, 190, 1100], outline=(120, 40, 40), width=2)
    d.ellipse([98, 1008, 182, 1092], outline=(120, 40, 40), width=1)
    signature_curve(d, (540, 1000, 740, 1060), rng)
    d.text((640, 1080), "Controller of Examinations", font=load_font(reg, 15), fill=(70, 70, 70), anchor="mm")
    return img, f, None, p


RENDER = {"aadhaar": render_aadhaar, "pan": render_pan, "marksheet": render_marksheet}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="images per doc type")
    n = ap.parse_args().n
    GENUINE.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    for old in list(GENUINE.glob("*.jpg")) + list(META.glob("*.json")):
        old.unlink()
    for dt in DOC_TYPES:
        for i in tqdm(range(n), desc=dt):
            stem = f"{dt}_{i:04d}"
            rng = random.Random(f"{SEED}-{stem}")
            nprng = np.random.default_rng(rng.getrandbits(32))
            img, fields, photo, person = RENDER[dt](rng, nprng)
            img, sigma = finish(img, nprng, rng)
            img.save(GENUINE / f"{stem}.jpg", quality=BASE_QUALITY)
            (META / f"{stem}.json").write_text(json.dumps(
                {"stem": stem, "doc_type": dt, "size": list(img.size), "noise": sigma, "fields": fields,
                 "photo": photo, "person": person}))


if __name__ == "__main__":
    main()
