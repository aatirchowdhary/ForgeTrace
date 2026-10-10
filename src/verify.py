"""Sanity checks: counts per split x doc_type x label, no base_id leakage, and preview.png."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from common import DOC_TYPES, ROOT, SPLITS


def main() -> None:
    df = pd.concat({n: pd.read_csv(SPLITS / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test")},
                   names=["split"]).reset_index(level=0).reset_index(drop=True)
    print(pd.crosstab([df.split, df.doc_type], df.label.map({0: "genuine", 1: "tampered"})).to_string(), "\n")
    print(df[df.label == 1].groupby(["doc_type", "tamper_type"]).size().to_string(), "\n")
    leaks = df.groupby("base_id").split.nunique().gt(1).sum()
    print(f"base_ids spanning more than one split: {leaks}")
    assert leaks == 0

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    test = df[df.split == "test"]
    for r, dt in enumerate(DOC_TYPES):
        t = test[(test.doc_type == dt) & (test.label == 1)].iloc[0]
        g = test[(test.base_id == t.base_id) & (test.label == 0)].iloc[0]
        for c, (title, p) in enumerate([("genuine", g.path), (f"tampered: {t.tamper_type}", t.path), ("mask", t.mask_path)]):
            im = Image.open(ROOT / p)
            axes[r, c].imshow(im, cmap="gray" if c == 2 else None)
            axes[r, c].set_title(f"{dt} - {title}", fontsize=10)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "preview.png", dpi=110)
    print("saved preview.png")


if __name__ == "__main__":
    main()
