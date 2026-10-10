"""Phase 5: full evaluation on the TEST split only. Everything is written to results/."""
from __future__ import annotations

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from tqdm import tqdm

from common import DOC_TYPES, ROOT, SEED, SPLITS
from fusion import final_score, load_threshold, predict_verdicts, score_split
from infer import load_model, predict
from ocr_check import ocr_check

RESULTS = ROOT / "results"
TAMPER_ORDER = ["text_edit", "photo_swap", "copy_move", "splice", "marks_edit_naive", "marks_edit_full"]
VARIANTS = {"model_only": "model only", "ocr_hard_fail": "OCR rules (hard-fail)", "ocr_any_flag": "OCR rules (any flag)",
            "fused": "fused"}


# ---------------------------------------------------------------- helpers
def as_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: v is True or str(v) == "True")


def md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("-" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def tune_model_only(val: pd.DataFrame, max_fpr: float) -> dict:
    """Per-doc-type threshold on the model probability, same protocol as the fused thresholds (val, FPR cap)."""
    def tune(d):
        y, grid, res = d.label.to_numpy(), np.append(np.linspace(0, 1, 401), 1.01), []
        for t in grid:
            p = (d.model_prob >= t).astype(int).to_numpy()
            if p[y == 0].mean() <= max_fpr:
                res.append((f1_score(y, p, zero_division=0), t))
        best = max(f for f, _ in res)
        return float(np.median([t for f, t in res if np.isclose(f, best)]))
    out = {dt: tune(val[val.ocr_doc_type == dt]) for dt in DOC_TYPES}
    out["global"] = tune(val)
    return out


def all_verdicts(df: pd.DataFrame, cfg: dict, mthr: dict) -> dict[str, np.ndarray]:
    t = df.ocr_doc_type.map(lambda d: mthr.get(d, mthr["global"]))
    hard = as_bool(df.hard_fail)
    return {"model_only": (df.model_prob >= t).astype(int).to_numpy(),
            "ocr_hard_fail": hard.astype(int).to_numpy(),
            "ocr_any_flag": (hard | (df.ocr_flags != "")).astype(int).to_numpy(),
            "fused": predict_verdicts(df.assign(hard_fail=hard), cfg)}


def fused_score(df: pd.DataFrame) -> np.ndarray:
    """Continuous fused score for ROC; a hard fail ranks above everything else."""
    return (final_score(df.model_prob, df.ocr_score) + as_bool(df.hard_fail).astype(float)).to_numpy()


def bin_metrics(y, pred, score=None) -> dict:
    return {"n": len(y), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "fpr": float(np.mean(np.asarray(pred)[np.asarray(y) == 0])) if (np.asarray(y) == 0).any() else np.nan,
            "roc_auc": roc_auc_score(y, score) if score is not None and len(set(y)) > 1 else np.nan}


# ---------------------------------------------------------------- 1-3 classification, ablation, tamper recall
def classification(test, verd):
    rows = []
    for name, d in [("overall", test)] + [(dt, test[test.doc_type == dt]) for dt in DOC_TYPES]:
        idx = d.index.to_numpy()
        rows.append({"scope": name, **bin_metrics(d.label.to_numpy(), verd["fused"][idx], fused_score(d))})
    return pd.DataFrame(rows)


def plot_cm_roc(test, verd):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, (name, d) in zip(ax, [("overall", test)] + [(dt, test[test.doc_type == dt]) for dt in DOC_TYPES]):
        cm = confusion_matrix(d.label, verd["fused"][d.index.to_numpy()], labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        a.set_xticks([0, 1], ["pred genuine", "pred tampered"])
        a.set_yticks([0, 1], ["genuine", "tampered"])
        a.set_title(f"fused - {name}")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, (name, d) in zip(ax, [("overall", test)] + [(dt, test[test.doc_type == dt]) for dt in DOC_TYPES]):
        for label, score in [("model", d.model_prob.to_numpy()), ("fused", fused_score(d))]:
            fpr, tpr, _ = roc_curve(d.label, score)
            a.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(d.label, score):.2f})")
        a.plot([0, 1], [0, 1], "k:", lw=1)
        a.set_title(f"ROC - {name}")
        a.set_xlabel("false positive rate")
        a.set_ylabel("true positive rate")
        a.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "roc.png", dpi=110)
    plt.close(fig)


def ablation(test, verd):
    rows = []
    for key, label in VARIANTS.items():
        r = {"variant": label}
        for name, d in [("overall", test)] + [(dt, test[test.doc_type == dt]) for dt in DOC_TYPES]:
            r[f"F1 {name}"] = f1_score(d.label, verd[key][d.index.to_numpy()], zero_division=0)
        r["genuine flagged (FPR)"] = verd[key][(test.label == 0).to_numpy()].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def tamper_recall(test, verd):
    rows = []
    t = test[test.label == 1]
    groups = [(dt, tt, t[(t.doc_type == dt) & (t.tamper_type == tt)]) for dt in DOC_TYPES for tt in TAMPER_ORDER]
    groups += [("all", tt, t[t.tamper_type == tt]) for tt in TAMPER_ORDER]
    for dt, tt, d in groups:
        if len(d):
            idx = d.index.to_numpy()
            rows.append({"doc_type": dt, "tamper_type": tt, "n": len(d),
                         **{VARIANTS[k]: verd[k][idx].mean() for k in VARIANTS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4 localisation
def loc_metrics(prob: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    p, g = prob > thr, gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    return {"iou": inter / union if union else 1.0, "dice": 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0,
            "precision": inter / p.sum() if p.sum() else 0.0, "recall": inter / g.sum() if g.sum() else 0.0}


def localisation(test, model):
    rows = []
    t = test[test.label == 1]
    for r in tqdm(t.itertuples(), total=len(t), desc="localisation"):
        img = cv2.imread(str(ROOT / r.path))
        gt = cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127
        rows.append({"doc_type": r.doc_type, "tamper_type": r.tamper_type, **loc_metrics(predict(img, model)["mask_prob"], gt)})
    d = pd.DataFrame(rows)
    cols = ["iou", "dice", "precision", "recall"]
    overall = d[cols].mean().to_frame().T.assign(scope="overall", n=len(d))
    per_dt = d.groupby("doc_type")[cols].mean().reset_index().rename(columns={"doc_type": "scope"}).assign(
        n=d.groupby("doc_type").size().values)
    per_tt = d.groupby("tamper_type")[cols].mean().reindex([x for x in TAMPER_ORDER if x in set(d.tamper_type)]).reset_index().rename(
        columns={"tamper_type": "scope"}).assign(n=d.groupby("tamper_type").size().reindex(
            [x for x in TAMPER_ORDER if x in set(d.tamper_type)]).values)
    order = ["scope", "n"] + cols
    return pd.concat([overall, per_dt], ignore_index=True)[order], per_tt[order]


# ---------------------------------------------------------------- 5 robustness
def _jpeg(q):
    return lambda im, rng: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _blur(k):
    return lambda im, rng: cv2.GaussianBlur(im, (k, k), 0)


def _noise(s):
    return lambda im, rng: np.clip(im.astype(np.float32) + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)


def _down(im, rng):
    h, w = im.shape[:2]
    return cv2.resize(cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)


def _phone(im, rng):
    """Perspective warp + brightness gradient + blur."""
    h, w = im.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + (rng.uniform(-0.035, 0.035, (4, 2)) * [w, h]).astype(np.float32)
    out = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    a = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    ramp = 1 + 0.18 * (np.cos(a) * xx + np.sin(a) * yy)
    return cv2.GaussianBlur(np.clip(out * ramp[..., None], 0, 255).astype(np.uint8), (3, 3), 0)


CONDITIONS = {"clean": lambda im, rng: im, "jpeg90": _jpeg(90), "jpeg70": _jpeg(70), "jpeg50": _jpeg(50),
              "blur_k3": _blur(3), "blur_k5": _blur(5), "noise_s5": _noise(5), "noise_s10": _noise(10),
              "downscale50": _down, "phone": _phone}


def robustness(test, model, cfg, mthr, n_per_cell):
    sub = pd.concat([g.sample(min(n_per_cell, len(g)), random_state=SEED)
                     for _, g in test.groupby(["doc_type", "label"])]).sort_index()
    cache_dir = RESULTS / "robustness_cache"
    cache_dir.mkdir(exist_ok=True)
    rows = []
    for cond, fn in CONDITIONS.items():
        f = cache_dir / f"{cond}_n{n_per_cell}.csv"
        if f.exists():
            d = pd.read_csv(f, keep_default_na=False)
        else:
            recs = []
            for i, r in enumerate(tqdm(sub.itertuples(), total=len(sub), desc=f"robustness/{cond}")):
                im = fn(cv2.imread(str(ROOT / r.path)), np.random.default_rng(SEED + i))
                o = ocr_check(im)
                recs.append({"path": r.path, "label": r.label, "doc_type": r.doc_type, "model_prob": predict(im, model)["prob"],
                             "ocr_score": o["ocr_score"], "hard_fail": o["hard_fail"], "ocr_flags": "|".join(o["flags"]),
                             "ocr_doc_type": o["doc_type"]})
            d = pd.DataFrame(recs)
            d.to_csv(f, index=False)
        v = all_verdicts(d, cfg, mthr)
        y = d.label.to_numpy()
        row = {"condition": cond, "n": len(d), "fused_f1": f1_score(y, v["fused"], zero_division=0),
               "fused_recall": recall_score(y, v["fused"], zero_division=0), "fused_fpr": v["fused"][y == 0].mean(),
               "model_only_f1": f1_score(y, v["model_only"], zero_division=0),
               "genuine_hard_fail_rate": as_bool(d[d.label == 0].hard_fail).mean()}
        for dt in DOC_TYPES:
            m = (d.doc_type == dt).to_numpy()
            row[f"fused_f1_{dt}"] = f1_score(y[m], v["fused"][m], zero_division=0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_robustness(rob):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(rob))
    w = 0.27
    ax[0].bar(x - w, rob.fused_f1, w, label="fused F1")
    ax[0].bar(x, rob.fused_recall, w, label="fused recall")
    ax[0].bar(x + w, rob.fused_fpr, w, label="fused FPR (genuine flagged)")
    ax[0].set_xticks(x, rob.condition, rotation=40, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend()
    ax[0].set_title("fused detector under distortions")
    for dt in DOC_TYPES:
        ax[1].plot(x, rob[f"fused_f1_{dt}"], marker="o", label=dt)
    ax[1].plot(x, rob.model_only_f1, "k--", marker="s", label="model only (overall)")
    ax[1].set_xticks(x, rob.condition, rotation=40, ha="right")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("F1 per doc type")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 6 error analysis
def error_analysis(test, verd, model):
    score = fused_score(test)
    pred = verd["fused"]
    fp = test[(test.label == 0) & (pred == 1)].assign(score=score[(test.label == 0).to_numpy() & (pred == 1)]).nlargest(8, "score")
    fn = test[(test.label == 1) & (pred == 0)].assign(score=score[(test.label == 1).to_numpy() & (pred == 0)]).nsmallest(8, "score")
    fig, axes = plt.subplots(6, 8, figsize=(26, 19))
    for col_off, (kind, d) in enumerate([("FALSE POSITIVE", fp), ("FALSE NEGATIVE", fn)]):
        for c, r in enumerate(d.itertuples()):
            im = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB)
            gt = (cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127) if r.mask_path else np.zeros(im.shape[:2], bool)
            heat = predict(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), model)["mask_prob"]
            base = col_off * 3
            axes[base, c].imshow(im)
            axes[base, c].set_title(f"{kind}\n{r.doc_type}/{r.tamper_type}\nscore {r.score:.2f} model {r.model_prob:.2f}", fontsize=8)
            axes[base + 1, c].imshow(im)
            axes[base + 1, c].imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.8)
            axes[base + 1, c].set_title("ground-truth mask" + ("" if gt.any() else " (none)"), fontsize=8)
            axes[base + 2, c].imshow(im)
            axes[base + 2, c].imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[base + 2, c].set_title(f"model heatmap (max {heat.max():.2f})", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "errors.png", dpi=80)
    plt.close(fig)
    return fp, fn


# ---------------------------------------------------------------- 7 summary
def observations(cls, abl, tr, loc, rob, fp, fn) -> list[str]:
    g = lambda dt, tt, col: tr[(tr.doc_type == dt) & (tr.tamper_type == tt)][col]
    o = []
    o.append(f"Fused detector on the test split: F1 {cls.f1[0]:.3f}, recall {cls.recall[0]:.3f}, precision {cls.precision[0]:.3f}, "
             f"false-positive rate {cls.fpr[0]:.1%}, ROC-AUC {cls.roc_auc[0]:.3f}.")
    for tt in ("marks_edit_naive", "marks_edit_full"):
        d = tr[(tr.doc_type == "marksheet") & (tr.tamper_type == tt)]
        if len(d):
            d = d.iloc[0]
            o.append(f"`{tt}` (marksheet, n={int(d.n)}): OCR rules catch {d['OCR rules (hard-fail)']:.0%}, model alone "
                     f"{d['model only']:.0%}, fused {d['fused']:.0%}.")
    a = abl.set_index("variant")
    o.append("Overall F1 by variant: " + ", ".join(f"{k} {a.loc[k, 'F1 overall']:.3f}" for k in a.index) + ".")
    worst = tr[tr.doc_type == "all"].nsmallest(2, "fused")
    o.append("Weakest tamper types (fused recall, pooled): " + ", ".join(f"{r.tamper_type} {r.fused:.0%}" for r in worst.itertuples()) + ".")
    o.append(f"Localisation (tampered only, threshold 0.5): mean IoU {loc.iou[0]:.3f}, Dice {loc.dice[0]:.3f}.")
    c = rob.set_index("condition")
    drop = (c.fused_f1 - c.fused_f1["clean"]).drop("clean")
    o.append(f"Robustness (subset n={int(c.n['clean'])}): clean fused F1 {c.fused_f1['clean']:.3f}; largest drop under "
             f"{drop.idxmin()} ({c.fused_f1[drop.idxmin()]:.3f}); largest FPR {c.fused_fpr.idxmax()} ({c.fused_fpr.max():.1%}).")
    o.append(f"Error analysis: {len(fp)} false positives and {len(fn)} false negatives shown in `errors.png`.")
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-n", type=int, default=20, help="images per (doc_type, label) cell for the robustness subset")
    ap.add_argument("--skip-robustness", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cfg = load_threshold()
    model = load_model()
    val = score_split("val", model)
    masks = pd.read_csv(SPLITS / "test.csv", keep_default_na=False)[["path", "mask_path"]]
    test = score_split("test", model).merge(masks, on="path", how="left").reset_index(drop=True)
    test["hard_fail"] = as_bool(test.hard_fail)
    mthr = tune_model_only(val, cfg["max_fpr"])
    verd = all_verdicts(test, cfg, mthr)

    cls = classification(test, verd)
    plot_cm_roc(test, verd)
    abl = ablation(test, verd)
    tr = tamper_recall(test, verd)
    loc, loc_tt = localisation(test, model)
    fp, fn = error_analysis(test, verd, model)
    if a.skip_robustness:
        rob = pd.read_csv(RESULTS / "robustness.csv")
    else:
        rob = robustness(test, model, cfg, mthr, a.robust_n)
    rob.to_csv(RESULTS / "robustness.csv", index=False)
    plot_robustness(rob)
    for name, df in [("classification", cls), ("ablation", abl), ("tamper_type_recall", tr), ("localization", loc),
                     ("localization_by_tamper_type", loc_tt)]:
        df.to_csv(RESULTS / f"{name}.csv", index=False)

    err_tbl = lambda d: d[["path", "doc_type", "tamper_type", "model_prob", "ocr_flags", "score"]]
    s = ["# doc-forgery: test-split evaluation\n",
         f"Test split: {len(test)} images ({int((test.label == 0).sum())} genuine / {int((test.label == 1).sum())} tampered). "
         "Thresholds were tuned on the validation split only "
         f"(per-doc-type, genuine false-positive rate capped at {cfg['max_fpr']:.0%}): "
         + ", ".join(f"{k} {v:.3f}" for k, v in cfg["per_doc_type"].items()) + ".\n",
         "## Observations\n", *[f"- {x}" for x in observations(cls, abl, tr, loc, rob, fp, fn)], "",
         "## 1. Classification (fused verdict)\n", md(cls), "\nPlots: `confusion_matrix.png`, `roc.png`.\n",
         "## 2. Ablation (F1; last column = share of genuine images flagged)\n",
         "Model-only uses its own val-tuned per-doc-type threshold (same FPR cap). OCR (hard-fail) = only the decisive rules; "
         "OCR (any flag) = any flag raised.\n", md(abl),
         "\n## 3. Recall per tamper type\n", md(tr),
         "\n## 4. Localisation (tampered images, threshold 0.5)\n", md(loc), "\nBy tamper type:\n", md(loc_tt),
         f"\n## 5. Robustness (subset of {int(rob.n[0])} test images)\n", md(rob), "\nPlot: `robustness.png`.\n",
         "## 6. Error analysis\n", "`errors.png` (top: false positives, bottom: false negatives; image / GT mask / heatmap).\n",
         "Worst false positives:\n", md(err_tbl(fp)), "\nWorst false negatives:\n", md(err_tbl(fn)), ""]
    (RESULTS / "summary.md").write_text("\n".join(s), encoding="utf-8")
    print("\n".join(s))


if __name__ == "__main__":
    main()
