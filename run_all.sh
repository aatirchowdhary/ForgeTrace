#!/usr/bin/env bash
# Full doc-forgery pipeline: synthetic data -> splits -> baseline -> training -> threshold tuning -> evaluation.
# Usage:  bash run_all.sh            (everything)
#         bash run_all.sh --app      (everything, then launch the Streamlit app)
# Needs a CUDA GPU for reasonable training time (about 50 minutes for the crop model); CPU works but is very slow.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
export PYTHONUTF8=1   # easyocr prints a progress bar that breaks on some Windows console encodings

step() { printf '\n==== %s ====\n' "$1"; }

step "1/9 make genuine documents (1200 images, fictional data)"
$PY src/make_genuine.py
step "2/9 make tampered copies + masks"
$PY src/make_tampered.py
step "3/9 train/val/test split"
$PY src/split.py
step "4/9 verify split + preview.png"
$PY src/verify.py
step "5/9 ELA-statistics baseline (RandomForest / SVM)"
$PY src/baseline.py
step "6/9 train the U-Net on native-resolution crops (tiled inference)"
$PY src/train_unet.py --crops --workers 3
step "7/9 score val/test (model + OCR) and tune fusion thresholds on val"
$PY src/fusion.py --rescore
step "8/9 evaluate on the test split -> results/summary.md"
rm -rf results/robustness_cache   # cached robustness scores belong to the previous model
$PY src/evaluate.py
step "9/9 done"
echo "Results: results/summary.md   Models: models/   Demo: streamlit run app/app.py"

if [[ "${1:-}" == "--app" ]]; then
  streamlit run app/app.py
fi
