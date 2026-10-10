"""ELA-statistics baselines: RandomForest and RBF-SVM. Saves the one with the best val F1."""
from __future__ import annotations

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from common import DOC_TYPES, ROOT, SEED, SPLITS
from features import ela, ela_stats

MODELS = ROOT / "models"


def load_split(name: str):
    df = pd.read_csv(SPLITS / f"{name}.csv", keep_default_na=False)
    X = []
    for r in tqdm(df.itertuples(), total=len(df), desc=f"features/{name}"):
        stats = ela_stats(ela(cv2.imread(str(ROOT / r.path))))  # full-resolution image, not letterboxed
        X.append(np.concatenate([stats, np.eye(len(DOC_TYPES))[DOC_TYPES.index(r.doc_type)]]))
    return np.array(X), df.label.to_numpy(), df.doc_type.to_numpy()


def metrics(y, p) -> dict:
    return dict(acc=accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
                rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0))


def report(name, split, model, X, y, dt) -> float:
    pred = model.predict(X)
    rows = {"overall": metrics(y, pred)}
    rows.update({d: metrics(y[dt == d], pred[dt == d]) for d in DOC_TYPES})
    print(f"\n{name} / {split}\n{pd.DataFrame(rows).T.round(3).to_string()}")
    return rows["overall"]["f1"]


def main() -> None:
    Xtr, ytr, _ = load_split("train")
    Xva, yva, dva = load_split("val")
    Xte, yte, dte = load_split("test")
    models = {"random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
              "rbf_svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED))}
    best = (None, -1.0)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_f1 = report(name, "val", m, Xva, yva, dva)
        report(name, "test", m, Xte, yte, dte)
        if val_f1 > best[1]:
            best = (name, val_f1)
    name = best[0]
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"name": name, "model": models[name], "doc_types": DOC_TYPES}, MODELS / "baseline.joblib")
    test_f1 = f1_score(yte, models[name].predict(Xte))
    print(f"\nbest by val F1: {name} (val F1 {best[1]:.3f}) -> models/baseline.joblib")
    print(f"BASELINE TEST F1 ({name}): {test_f1:.3f}")


if __name__ == "__main__":
    main()
