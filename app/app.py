"""Streamlit demo: upload a document image -> image model + OCR rule checks -> fused verdict."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion import fuse, load_threshold  # noqa: E402
from infer import load_model, predict  # noqa: E402
from pdf_input import load_image  # noqa: E402
from validate import check_document, check_image  # noqa: E402
from localize import anomaly_map, field_anomalies, field_verdict, suspicious_fields  # noqa: E402
from ocr_check import HARD_FLAGS, get_reader, ocr_check  # noqa: E402

EXPLAIN = {
    "missing_id": "No 12-digit Aadhaar number could be read from the card.",
    "invalid_aadhaar_prefix": "The Aadhaar number starts with 0 or 1, which a real Aadhaar number never does.",
    "aadhaar_checksum_fail": "The Aadhaar number fails the Verhoeff checksum, so it cannot be a valid number.",
    "impossible_dob": "A printed date is not a real calendar date (for example 29/02 in a year that is not a leap year).",
    "invalid_dob": "The date of birth is missing, is not a real DD/MM/YYYY date, or gives an age outside 0-120.",
    "invalid_gender": "The gender is not one of Male, Female or Transgender.",
    "missing_qr": "No QR code was found on the card.",
    "invalid_pan_format": "The PAN does not match the required format of five letters, four digits and a letter.",
    "invalid_pan_type": "The 4th PAN character is not a valid holder-type code (P, C, H, F, A, T, B, L, J or G).",
    "pan_surname_mismatch": "For an individual PAN, the 5th character must be the first letter of the surname, "
                            "but it does not match the printed name.",
    "missing_header": "The 'INCOME TAX DEPARTMENT' header could not be found.",
    "marks_exceed_max": "A subject's marks obtained are higher than its maximum marks.",
    "total_mismatch": "The printed total does not equal the sum of the subject marks.",
    "percentage_mismatch": "The printed percentage does not match the total divided by the maximum marks.",
    "result_mismatch": "The printed PASS/FAIL result contradicts the rule that a mark below 33 in any subject is a fail.",
    "font_inconsistency": "Some text lines have clearly different font heights from their neighbours, "
                          "which can happen when text has been overwritten.",
    "low_ocr_confidence": "The text is hard to read (low OCR confidence), so the rule checks may be unreliable.",
    "model_score_high": "The image model sees pixel-level patterns typical of edited regions.",
}
HEATMAP_MIN, HEATMAP_ALPHA = 0.3, 0.6

st.set_page_config(page_title="FORGETRACE", page_icon=":material/fact_check:", layout="wide")
st.html(f"<style>{(ROOT / 'app' / 'style.css').read_text(encoding='utf-8')}</style>")


# ---------------------------------------------------------------- cached resources and helpers
@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_resources():
    get_reader()  # warm up the OCR reader so the first upload is not slow
    return load_model(), load_threshold()


@st.cache_data(show_spinner=False)
def test_samples() -> pd.DataFrame:
    return pd.read_csv(ROOT / "splits" / "test.csv", keep_default_na=False)


@st.cache_data(max_entries=16, show_spinner=False)
def run_pipeline(img_bytes: bytes) -> dict:
    """Heavy part (model + OCR). Works on in-memory bytes only; nothing is written to disk."""
    img, err = load_image(img_bytes)
    if img is None:
        return {"error": err}
    reason = check_image(img)
    if reason:
        return {"rejected": reason}
    ocr = ocr_check(img)
    reason = check_document(ocr)
    if reason:
        return {"rejected": reason}
    model, _ = load_resources()
    pred = predict(img, model)
    fields = field_anomalies(img) + [  # fields a failed rule points at (e.g. an impossible date) are highlighted too
        {"text": r["text"], "bbox": r["bbox"], "score": 99.0, "conf": r["conf"], "reason": r["flag"].replace("_", " ")}
        for r in ocr["rule_boxes"]]
    return {"model_prob": pred["prob"], "mask_prob": pred["mask_prob"].astype(np.float16), "ocr": ocr,
            "fields": fields, "field_map": anomaly_map(img.shape, fields).astype(np.float16)}


_t = np.linspace(0, 1, 256, dtype=np.float32)[:, None]  # orange -> pale amber lookup table (BGR), no blue / purple
ORANGE_LUT = (np.array([10, 70, 255], np.float32) * (1 - _t) + np.array([190, 235, 255], np.float32) * _t).astype(np.uint8)[:, None, :]


def heatmap_overlay(img_bgr: np.ndarray, mask_prob: np.ndarray) -> np.ndarray:
    """Orange heatmap blended at alpha 0.45, only where mask_prob > 0.3. Returns RGB."""
    heat = cv2.applyColorMap((np.clip(mask_prob.astype(np.float32), 0, 1) * 255).astype(np.uint8), ORANGE_LUT)
    on = (mask_prob > HEATMAP_MIN)[..., None]
    blended = cv2.addWeighted(img_bgr, 1 - HEATMAP_ALPHA, heat, HEATMAP_ALPHA, 0)
    return cv2.cvtColor(np.where(on, blended, img_bgr).astype(np.uint8), cv2.COLOR_BGR2RGB)


def mask_aadhaar(lines: list[str]) -> list[str]:
    """Hide the first 8 digits of any 12-digit number (also when OCR split it into three 4-digit lines)."""
    pat = re.compile(r"(?<!\d)(\d{4})(\s?)(\d{4})(\s?)(\d{4})(?!\d)")
    out = [pat.sub(lambda m: f"XXXX{m.group(2)}XXXX{m.group(4)}{m.group(5)}", ln) for ln in lines]
    for i in range(len(out) - 2):
        if all(re.fullmatch(r"\d{4}", out[j]) for j in range(i, i + 3)):
            out[i], out[i + 1] = "XXXX", "XXXX"
    return out


def flag_label(flag: str) -> str:
    return flag.replace("_", " ")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    use_tuned = st.toggle("Use tuned thresholds per document type", value=True,
                          help="Thresholds tuned on the validation split (see models/threshold.json).")
    manual_thr = st.slider("Manual final-score threshold", 0.0, 1.0, 0.5, 0.01, disabled=use_tuned,
                           help="A document is TAMPERED when its final score is at or above this value. "
                                "OCR hard failures always give TAMPERED.")
    use_unet = st.toggle("Use the synthetic-trained U-Net", value=False,
                         help="The U-Net was trained only on the generated synthetic documents. On real scans and "
                              "screenshots it reports high tamper probability for genuine documents too, so it is off "
                              "by default. Turn it on for the 'Random test sample' images.")
    st.divider()
    st.subheader("Try a sample")
    if st.button("Random test sample", icon=":material/shuffle:", width="stretch"):
        row = test_samples().iloc[random.randrange(len(test_samples()))]
        st.session_state["sample"] = row.to_dict()
    st.caption("Picks an image from the held-out test split, with its ground truth shown for comparison.")
    st.divider()
    st.caption("Uploaded images are processed in memory and never saved to disk.")

# ---------------------------------------------------------------- main page
st.html("""
<div class="hero">
  <div class="nav"><b>FORGE<i>TRACE</i></b><span class="sep"></span><span>Document forgery check</span><span class="tag">Research prototype</span></div>
  <h1>Forge<span style="-webkit-text-fill-color:#FF6A1A">trace</span></h1>
  <div class="vs">Authentic vs forged</div>
  <p>Upload an Aadhaar card, PAN card or marksheet. Every text field is compared with the text around it and the
  printed numbers are checked against their own rules.</p>
</div>
""")

source = st.segmented_control("Image source", ["Upload", "Camera"], default="Upload", label_visibility="collapsed",
                              selection_mode="single", key="source") or "Upload"
if source == "Camera":
    uploaded = st.camera_input("Take a photo of the document", resolution="1080p")
    st.caption("Lay the document flat in good light, fill the frame and hold steady. Tap the camera-switch icon "
               "to use the rear camera.")
else:
    uploaded = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "pdf"])
    st.caption("JPG, PNG or a single-page PDF.")
sample = st.session_state.get("sample")

if uploaded is not None:
    img_bytes, truth = uploaded.getvalue(), None
elif sample is not None:
    img_bytes = (ROOT / sample["path"]).read_bytes()
    truth = ("tampered (" + sample["tamper_type"].replace("_", " ") + ")") if sample["label"] == 1 else "genuine"
else:
    st.info("Upload a JPG, PNG or single-page PDF, or take a photo, of an Aadhaar card, PAN card or marksheet, or use the sidebar to pick a test sample.",
            icon=":material/upload_file:")
    st.stop()

model, cfg = load_resources()
if cfg.get("model") and cfg["model"] != getattr(model, "file", cfg["model"]):
    st.warning(f"The saved thresholds were tuned for `{cfg['model']}`, but `{model.file}` is loaded. "
               "Re-run `python src/fusion.py` to re-tune them.", icon=":material/warning:")

with st.spinner("Analysing the document..."):
    res = run_pipeline(img_bytes)
if "error" in res:
    st.error(res["error"])
    st.stop()
if "rejected" in res:
    st.error(f"**Image rejected.** {res['rejected']}", icon=":material/block:")
    st.caption("Accepted: a clear photo or scan of an Aadhaar card or letter, a PAN card, or a school marksheet.")
    st.stop()

img, _ = load_image(img_bytes)  # run_pipeline already succeeded, so this cannot fail
ocr, prob = res["ocr"], res["model_prob"]
suspicious = suspicious_fields(res["fields"])
if use_unet:
    verdict = fuse(prob, ocr, cfg if use_tuned else manual_thr)
    threshold = (cfg["per_doc_type"].get(ocr["doc_type"], cfg["threshold"]) if use_tuned else manual_thr)
    tampered = verdict["verdict"] == "TAMPERED"
else:  # hard OCR rules, or many text fields that look re-drawn (see localize.py), give TAMPERED
    verdict = {"verdict": "TAMPERED" if ocr["hard_fail"] else field_verdict(len(suspicious)),
               "final_score": float(ocr["ocr_score"]), "reasons": list(ocr["flags"])}
    threshold, tampered = float("nan"), verdict["verdict"] == "TAMPERED"

# verdict + scores
with st.container(border=True):
    top = st.container(horizontal=True, vertical_alignment="center")
    with top:
        if verdict["verdict"] == "REVIEW":
            st.badge("REVIEW: suspicious fields", icon=":material/search:", color="orange")
        elif tampered:
            st.badge("TAMPERED", icon=":material/gpp_bad:", color="red")
        else:
            st.badge("GENUINE", icon=":material/verified:", color="green")
        st.markdown(f"Detected document type: **{ocr['doc_type']}**")
        if truth:
            st.markdown(f"Ground truth: **{truth}**")
    c = st.columns(4 if use_unet else 2)
    c[0].metric("OCR score", f"{ocr['ocr_score']:.2f}",
                help="Rule-check score from the text read by OCR. 1.00 if any hard rule failed (impossible date, "
                     "Aadhaar checksum, PAN format, marksheet totals). Otherwise the number of soft warnings divided by 3 "
                     "(0.33 = one warning). It adds context only: the TAMPERED verdict comes from hard rule failures or "
                     "suspicious text fields, not from this score.")
    c[1].metric("Suspicious text fields", len(suspicious),
                help="Text fields whose edges are crisper than the text around them (a sign of re-typed text), plus any "
                     "field a failed rule points at. One or more means TAMPERED.")
    if use_unet:
        c[2].metric("Model probability", f"{prob:.3f}")
        c[3].metric("Final / threshold", f"{verdict['final_score']:.2f} / {threshold:.2f}")
    if ocr["hard_fail"]:
        st.caption("An OCR hard failure overrides the threshold and always gives TAMPERED.")

# reasons
st.subheader("Why")
reasons = verdict["reasons"]
if reasons:
    for r in reasons:
        icon = ":material/error:" if r in HARD_FLAGS else ":material/warning:"
        (st.error if r in HARD_FLAGS else st.warning)(f"**{flag_label(r).capitalize()}.** {EXPLAIN.get(r, r)}", icon=icon)
else:
    st.success("No rule violations were found and the image model sees nothing suspicious.", icon=":material/check_circle:")

# images
left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alt="The uploaded document", width="stretch")
with right:
    st.subheader("Tamper heatmap")
    st.image(heatmap_overlay(img, (res["mask_prob"] if use_unet else res["field_map"])), width="stretch",
             alt="The document with the suspected edited regions highlighted in colour")
    st.caption("Text fields whose ink and background look different from the text around them (re-drawn text is "
               "usually crisper than the surrounding scan). Redder = stronger. Each field is compared only with its "
               "neighbours in the same image, so it does not depend on the upload's resolution or compression."
               if not use_unet else "Output of the synthetic-trained U-Net (reliable only on the generated documents).")
    if not use_unet and suspicious:
        st.dataframe(pd.DataFrame([{"text": f["text"], "score": round(f["score"], 1), "cue": f["reason"]}
                                   for f in suspicious[:12]]), hide_index=True, width="stretch")

with st.expander("Extracted text"):
    st.caption("The first 8 digits of any Aadhaar number are hidden.")
    st.code("\n".join(mask_aadhaar(ocr["text"])) or "(no text found)", language=None)
