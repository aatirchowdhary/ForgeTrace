# FORGETRACE (doc-forgery)

Detect and localise tampering in **Indian identity and academic documents** (Aadhaar card, PAN card, school marksheet).
Two complementary signals are fused into one verdict:

- **Pixel forensics:** a U-Net (ResNet-34 encoder) takes RGB + an Error Level Analysis (ELA) channel and predicts a
  per-pixel tamper map plus an image-level tamper probability.
- **Document logic:** OCR (EasyOCR) reads the text and checks rules that pixels cannot express: the Aadhaar Verhoeff
  checksum, the PAN format and surname letter, and marksheet total / percentage / result consistency.

> **All data is synthetic and fictional.** The documents are drawn by `src/make_genuine.py` with randomly generated
> names and numbers. This is a research prototype, **not for production use** (see [Limitations](#limitations)).

## Folder structure

```
doc-forgery/
├── app/app.py            Streamlit demo
├── data/                 genuine/  tampered/  masks/  meta/ (field boxes)   <- generated
├── splits/               train.csv  val.csv  test.csv                       <- generated
├── models/               unet_crop.pt (final)  unet_best.pt (whole-page model)  threshold.json  baseline.joblib
├── results/              summary.md and all tables/plots from src/evaluate.py
│   └── letterbox_baseline/   the same evaluation for the earlier whole-page model, for comparison
├── src/
│   ├── indian_data.py    fictional names, Verhoeff, Aadhaar / PAN generators
│   ├── make_genuine.py   render 1200 genuine documents
│   ├── make_tampered.py  one tampered copy + mask per document (6 tamper types)
│   ├── split.py          70/15/15 split, grouped by base document
│   ├── features.py       ELA map + statistics
│   ├── dataset.py        whole-page dataset (letterbox 512x512, RGB+ELA)
│   ├── crops.py          native-resolution crop dataset + tiled inference
│   ├── baseline.py       ELA-statistics RandomForest / SVM
│   ├── train_unet.py     joint localisation + classification training
│   ├── infer.py          load_model() / predict()
│   ├── ocr_check.py      OCR rule checks
│   ├── fusion.py         fuse model + OCR, tune thresholds on val
│   └── evaluate.py       full test-split evaluation
├── run_all.sh            whole pipeline
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Developed on Python 3.14, Windows 11, an RTX 5050 laptop GPU. A GPU is strongly recommended: crop training takes about
  50 minutes on it. On Blackwell (RTX 50-series) GPUs install the CUDA 12.8 build of PyTorch:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`. Without a GPU the code falls back
  to CPU with a smaller batch and fewer epochs, and is very slow.
- EasyOCR downloads its English and Hindi models on first use (internet needed once).
- On Windows consoles set `PYTHONUTF8=1` (EasyOCR prints a progress bar that breaks on some code pages).

## Run order

```bash
python src/make_genuine.py                 # 1200 fictional documents
python src/make_tampered.py                # tampered copies + masks
python src/split.py                        # splits/*.csv
python src/verify.py                       # counts + preview.png
python src/baseline.py                     # ELA-stats baseline
python src/train_unet.py --crops           # train the U-Net on native-resolution crops -> models/unet_crop.pt
python src/fusion.py                       # score val/test, tune fusion thresholds on val -> models/threshold.json
python src/evaluate.py                     # test-split evaluation -> results/summary.md
streamlit run app/app.py                   # demo
```

`bash run_all.sh` runs all of it (`bash run_all.sh --app` also launches the demo).
`python src/train_unet.py` without `--crops` trains the earlier whole-page (letterbox) model instead.

## How it works

1. **Data.** Aadhaar (valid Verhoeff numbers), PAN (`AAAPS9999A` with the surname letter) and marksheets (6 subjects,
   total, percentage, result) are rendered with PIL. Each is tampered once by a random applicable method:
   `text_edit`, `photo_swap`, `copy_move`, `splice`, `marks_edit_naive` (marks changed, total left alone) and
   `marks_edit_full` (marks, total, percentage and result redrawn consistently). Every JPEG goes through the same
   quality distribution (60-95), so compression history does not reveal the class.
2. **Model.** U-Net input is RGB + ELA (4 channels). Training uses 512x512 crops at **native resolution** (crop origins
   are multiples of 8 so the JPEG block grid stays aligned); inference tiles the page with overlapping 512x512 tiles,
   averages the tile tamper maps and takes the maximum tile probability as the image score.
3. **OCR rules** raise flags; "hard" flags (Aadhaar checksum, PAN format / surname, marksheet total / percentage /
   result) force TAMPERED. A hard flag only stands if a second read at 2x scale reproduces it, which removes most OCR
   misreads on genuine documents.
4. **Fusion.** `final = 0.7 * model_prob + 0.3 * ocr_score`; TAMPERED if a hard flag fired or `final >= threshold`.
   Thresholds are tuned on the validation split only: one per document type, each capped at a 10% false-positive rate
   on genuine images.

## Results

Test split: 360 images (180 genuine / 180 tampered), never used for tuning. Full tables, plots and error analysis are in
[`results/summary.md`](results/summary.md).

| scope | n | accuracy | precision | recall | F1 | false-positive rate | ROC-AUC |
|---|---|---|---|---|---|---|---|
| overall | 360 | 0.761 | 0.935 | 0.561 | 0.701 | 0.039 | 0.819 |
| Aadhaar | 120 | 0.783 | 0.972 | 0.583 | 0.729 | 0.017 | 0.805 |
| PAN | 120 | 0.842 | 0.936 | 0.733 | 0.822 | 0.050 | 0.895 |
| marksheet | 120 | 0.658 | 0.880 | 0.367 | 0.518 | 0.050 | 0.702 |

**Ablation (F1)**

| variant | overall | Aadhaar | PAN | marksheet | genuine flagged |
|---|---|---|---|---|---|
| model only | 0.650 | 0.702 | 0.807 | 0.351 | 3.9% |
| OCR rules, hard fails only | 0.182 | 0.065 | 0.154 | 0.310 | 0.0% |
| OCR rules, any flag | 0.377 | 0.356 | 0.209 | 0.505 | 7.8% |
| **fused** | **0.701** | **0.729** | **0.822** | **0.518** | 3.9% |

**Recall per tamper type (fused, all document types pooled)**

| tamper type | n | model only | OCR hard fails | fused |
|---|---|---|---|---|
| photo_swap | 24 | 1.00 | 0.00 | 1.00 |
| copy_move | 53 | 0.77 | 0.11 | 0.79 |
| splice | 44 | 0.48 | 0.11 | 0.57 |
| text_edit | 35 | 0.11 | 0.03 | 0.11 |
| marks_edit_naive | 7 | 0.00 | 0.86 | 0.86 |
| marks_edit_full | 17 | 0.00 | 0.00 | 0.00 |

**Localisation** (tampered images, threshold 0.5): mean IoU 0.360, Dice 0.414. IoU is 0.99 for photo swaps, about 0.45 for
copy-move, 0.31 for splices and close to 0 for text edits.

**Robustness** (120-image subset; fused F1 on clean 0.729): JPEG 90/70/50 and noise sigma 5 leave it at 0.73-0.75; noise
sigma 10 gives 0.67. Blur, 50% downscaling and the phone-capture simulation raise the false-positive rate (to 32%, 50% and
52% respectively), so the detector is **not** reliable on blurred or phone-photographed documents.

What the numbers say, plainly:

- Large, structural tampering (photo swap, copy-move, many splices) is found well. Small text edits are mostly missed,
  and so are marksheet edits where the total and percentage were redrawn consistently.
- The OCR rules catch exactly what they were designed for (marks changed but the total left alone: 6 of 7) and almost
  nothing else. They are cheap and have essentially no false positives, which is why they are a hard override.
- The marksheet image model is weak on its own (ROC-AUC 0.65, against 0.79 for Aadhaar and 0.87 for PAN); marksheet detection leans on the OCR rules.
- Small sample sizes (7 and 17 for the marks edits, 10-22 per other cell) make per-type numbers noisy.
- An earlier whole-page (resized) model scored similarly (F1 0.718; `results/letterbox_baseline/`). Training on
  native-resolution crops improved marksheet localisation and reduced false positives but did not fix small text edits.

## Demo app

`streamlit run app/app.py` shows the detected document type, a GENUINE / TAMPERED badge, the model probability, OCR
score, final score and threshold, every raised flag with a plain-English explanation, the original image, a heatmap
overlay (JET, alpha 0.45, where the tamper map exceeds 0.3) and the extracted text with the first 8 digits of any
Aadhaar number hidden. The sidebar has a threshold control and a **Random test sample** button (draws from the held-out
test split and shows its ground truth). Uploaded files are processed in memory and never written to disk. Models are
cached with `st.cache_resource`.

## Limitations

- **Synthetic, fictional layouts.** Training and test documents come from the same generator, so the scores above are
  optimistic for real documents. The layouts are not official templates, and the model has never seen real scans,
  stamps, lamination glare or varied fonts.
- **No physical security features.** Holograms, microtext, UV / guilloche patterns, paper and ink properties are not
  modelled and cannot be checked from a flat image.
- **The Aadhaar QR is only checked for presence.** Its digital signature is not verified, so a forged card with a
  copied QR would pass that rule.
- **OCR dependence.** The rule checks assume the text is read correctly; blur or odd fonts raise false flags.
- **Weak spots:** small text edits, consistent marksheet edits, and phone-style captures (see Results).
- **Not for production use.** It must not be used to accept or reject real identity documents.

## Future work

- Real-layout data collected **with consent** (and privacy review), including phone captures and scans.
- Aadhaar **secure QR signature verification** (UIDAI-signed data) instead of presence-only checks.
- Verification against authoritative sources: **DigiLocker** and **NSDL** PAN verification APIs.
- Better small-edit detection: higher-resolution or multi-scale ELA, noise / camera-residual features, and training on
  consistent marksheet forgeries with tamper-aware augmentation.
- Distortion augmentation (blur, perspective, lighting) to fix the phone-capture false positives.
