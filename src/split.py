"""Write splits/{train,val,test}.csv: 70/15/15, per doc_type, grouped by base_id (genuine + its tampered copy together)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA, DOC_TYPES, GENUINE, ROOT, SEED, SPLITS

COLS = ["path", "label", "mask_path", "doc_type", "tamper_type", "base_id"]


def build_rows() -> pd.DataFrame:
    log = pd.read_csv(DATA / "tamper_log.csv").set_index("stem")["tamper_type"]
    rows = []
    for stem, ttype in log.items():
        dt = stem.rsplit("_", 1)[0]
        rows.append((f"data/genuine/{stem}.jpg", 0, "", dt, "none", stem))
        rows.append((f"data/tampered/{stem}.jpg", 1, f"data/masks/{stem}.png", dt, ttype, stem))
    return pd.DataFrame(rows, columns=COLS)


def main() -> None:
    df = build_rows()
    assign = {}
    for dt in DOC_TYPES:  # splitting ids within each doc type keeps the 70/15/15 ratio stratified by doc_type
        ids = sorted(df.loc[df.doc_type == dt, "base_id"].unique())
        train, rest = train_test_split(ids, test_size=0.30, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, random_state=SEED)
        assign.update({**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val"), **dict.fromkeys(test, "test")})
    df["split"] = df.base_id.map(assign)
    SPLITS.mkdir(exist_ok=True)
    for name in ("train", "val", "test"):
        part = df[df.split == name][COLS].sort_values("path")
        part.to_csv(SPLITS / f"{name}.csv", index=False)
        print(f"splits/{name}.csv: {len(part)} rows")


if __name__ == "__main__":
    main()
