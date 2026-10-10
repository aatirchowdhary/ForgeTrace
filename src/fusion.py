"""Fuse the U-Net classification probability with OCR rule checks into one verdict."""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from common import ROOT, SEED, SPLITS
from ocr_check import ocr_check

MODELS, RESULTS = ROOT / "models", ROOT / "results"
W_MODEL, W_OCR = 0.7, 0.3


def final_score(model_prob: float, ocr_score: float) -> float:
    return W_MODEL * model_prob + W_OCR * ocr_score


def pick_threshold(threshold: float | dict, doc_type: str) -> float:
    """`threshold` is a float, or the models/threshold.json dict (per-doc-type thresholds with a global fallback)."""
    if isinstance(threshold, dict):
        return threshold["per_doc_type"].get(doc_type, threshold["threshold"])
    return threshold


def fuse(model_prob: float, ocr: dict, threshold: float | dict) -> dict:
    score = final_score(model_prob, ocr["ocr_score"])
    threshold = pick_threshold(threshold, ocr["doc_type"])
    reasons = list(ocr["flags"]) + (["model_score_high"] if model_prob > 0.5 else [])
    return {"verdict": "TAMPERED" if ocr["hard_fail"] or score >= threshold else "GENUINE",
            "final_score": score, "reasons": reasons}


def load_threshold(path=MODELS / "threshold.json") -> dict:
    return json.loads(path.read_text())


def analyze(img_bgr: np.ndarray, model=None, threshold: float | dict | None = None) -> dict:
    """Full pipeline for one image: model prob + mask, OCR checks, fused verdict."""
    from infer import predict
    pred = predict(img_bgr, model)
    ocr = ocr_check(img_bgr)
    thr = load_threshold() if threshold is None else threshold
    return {**fuse(pred["prob"], ocr, thr), "model_prob": pred["prob"], "mask_prob": pred["mask_prob"],
            "ocr": ocr, "threshold": thr}


# ---------------------------------------------------------------- scoring a split (cached)
def score_split(name: str, model=None) -> pd.DataFrame:
    cache = RESULTS / f"scores_{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False)
    from infer import load_model, predict
    model = model or load_model()
    df = pd.read_csv(SPLITS / f"{name}.csv", keep_default_na=False)
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc=f"score/{name}"):
        img = cv2.imread(str(ROOT / r.path))
        ocr = ocr_check(img)
        rows.append({"path": r.path, "label": r.label, "doc_type": r.doc_type, "tamper_type": r.tamper_type,
                     "model_prob": predict(img, model)["prob"], "ocr_score": ocr["ocr_score"],
                     "hard_fail": ocr["hard_fail"], "ocr_flags": "|".join(ocr["flags"]), "ocr_doc_type": ocr["doc_type"]})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def predict_verdicts(df: pd.DataFrame, thr: float | dict) -> np.ndarray:
    score = W_MODEL * df.model_prob + W_OCR * df.ocr_score
    t = df.ocr_doc_type.map(lambda d: pick_threshold(thr, d)) if isinstance(thr, dict) else thr
    return ((df.hard_fail.astype(bool)) | (score >= t)).astype(int).to_numpy()


def tune_threshold(val: pd.DataFrame, max_fpr: float | None = None) -> tuple[float, float]:
    """Threshold maximising val F1 (middle of the plateau when several thresholds tie).
    With max_fpr, only thresholds whose false-positive rate on genuine images is <= max_fpr are allowed."""
    grid = np.append(np.linspace(0, 1, 401), 1.01)
    y = val.label.to_numpy()
    f1 = np.full(len(grid), -1.0)
    for i, t in enumerate(grid):
        pred = predict_verdicts(val, t)
        if max_fpr is None or pred[y == 0].mean() <= max_fpr:
            f1[i] = f1_score(y, pred, zero_division=0)
    best = grid[np.isclose(f1, f1.max())]
    return float(np.median(best)), float(f1.max())


def report(df: pd.DataFrame, thr: float, title: str) -> None:
    pred = predict_verdicts(df, thr)
    y = df.label.to_numpy()
    print(f"\n{title}: acc {np.mean(pred == y):.3f}  precision {precision_score(y, pred, zero_division=0):.3f}  "
          f"recall {recall_score(y, pred, zero_division=0):.3f}  F1 {f1_score(y, pred, zero_division=0):.3f}")
    g = df[df.label == 0]
    print(f"  genuine flagged TAMPERED: {pred[y == 0].mean():.1%}   genuine with a hard_fail: {g.hard_fail.astype(bool).mean():.1%}")
    print("  flagged by doc type (genuine -> tampered): " + ", ".join(
        f"{dt} {pred[(y == 0) & (df.doc_type == dt).to_numpy()].mean():.0%} -> {pred[(y == 1) & (df.doc_type == dt).to_numpy()].mean():.0%}"
        for dt in ("aadhaar", "pan", "marksheet")))
    t = df[df.label == 1].assign(hit=pred[y == 1])
    print("  recall by tamper type:\n" + t.groupby(["doc_type", "tamper_type"]).hit.agg(["mean", "size"])
          .rename(columns={"mean": "recall", "size": "n"}).round(2).to_string().replace("\n", "\n    "))
    print(f"  recall from OCR hard_fail alone: {t.hard_fail.astype(bool).mean():.1%}; model alone @0.5: "
          f"{(t.model_prob > 0.5).mean():.1%} (genuine FP {(g.model_prob > 0.5).mean():.1%})")


def demo_table(test: pd.DataFrame, thr: float) -> None:
    rows = []
    for dt in ("aadhaar", "pan", "marksheet"):
        d = test[test.doc_type == dt]
        rows.append(d[d.label == 0].sample(3, random_state=SEED))
        rows.append(d[d.label == 1].sample(3, random_state=SEED))
    t = pd.concat(rows).copy()
    t["final_score"] = W_MODEL * t.model_prob + W_OCR * t.ocr_score
    t["verdict"] = np.where(predict_verdicts(t, thr) == 1, "TAMPERED", "GENUINE")
    t["ocr_flags"] = t["ocr_flags"].replace("", "-")
    t["tamper_type"] = t.tamper_type.replace("none", "(genuine)")
    pd.set_option("display.width", 200, "display.max_colwidth", 48)
    print("\n" + t[["doc_type", "tamper_type", "ocr_flags", "model_prob", "final_score", "verdict"]]
          .round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="ignore cached scores_*.csv")
    ap.add_argument("--max-fpr", type=float, default=0.10, help="genuine false-positive cap for per-doc-type thresholds")
    a = ap.parse_args()
    if a.rescore:
        for f in RESULTS.glob("scores_*.csv"):
            f.unlink()
    from infer import load_model
    model = load_model()
    val, test = score_split("val", model), score_split("test", model)
    thr, f1 = tune_threshold(val)
    per = {dt: tune_threshold(val[val.ocr_doc_type == dt], a.max_fpr)[0] for dt in ("aadhaar", "pan", "marksheet")}
    cfg = {"model": model.file, "threshold": thr, "val_f1": f1, "per_doc_type": per, "max_fpr": a.max_fpr, "w_model": W_MODEL, "w_ocr": W_OCR}
    MODELS.mkdir(exist_ok=True)
    (MODELS / "threshold.json").write_text(json.dumps(cfg, indent=2))
    print(f"global threshold {thr:.3f} (val F1 {f1:.3f}); per-doc-type (FPR<={a.max_fpr:.0%}): "
          + ", ".join(f"{k} {v:.3f}" for k, v in per.items()) + " -> models/threshold.json")
    print(f"OCR doc-type detection accuracy (test): {(test.ocr_doc_type == test.doc_type).mean():.1%}")
    report(test, thr, "TEST, single global threshold (F1-optimal)")
    report(val, cfg, "VAL, per-doc-type thresholds")
    report(test, cfg, "TEST, per-doc-type thresholds  <- used for verdicts")
    demo_table(test, cfg)


if __name__ == "__main__":
    main()
