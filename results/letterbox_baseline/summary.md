# doc-forgery: test-split evaluation

Test split: 360 images (180 genuine / 180 tampered). Thresholds were tuned on the validation split only (per-doc-type, genuine false-positive rate capped at 10%): aadhaar 0.250, pan 0.425, marksheet 0.405.

## Observations

- Fused detector on the test split: F1 0.718, recall 0.594, precision 0.907, false-positive rate 6.1%, ROC-AUC 0.821.
- `marks_edit_naive` (marksheet, n=7): OCR rules catch 86%, model alone 14%, fused 100%.
- `marks_edit_full` (marksheet, n=17): OCR rules catch 0%, model alone 0%, fused 12%.
- Overall F1 by variant: model only 0.660, OCR rules (hard-fail) 0.182, OCR rules (any flag) 0.377, fused 0.718.
- Weakest tamper types (fused recall, pooled): text_edit 9%, marks_edit_full 12%.
- Localisation (tampered only, threshold 0.5): mean IoU 0.358, Dice 0.407.
- Robustness (subset n=120): clean fused F1 0.740; largest drop under phone (0.667); largest FPR phone (50.0%).
- Error analysis: 8 false positives and 8 false negatives shown in `errors.png`.

## 1. Classification (fused verdict)

| scope | n | accuracy | precision | recall | f1 | fpr | roc_auc |
|---|---|---|---|---|---|---|---|
| overall | 360 | 0.767 | 0.907 | 0.594 | 0.718 | 0.061 | 0.821 |
| aadhaar | 120 | 0.800 | 0.909 | 0.667 | 0.769 | 0.067 | 0.827 |
| pan | 120 | 0.817 | 0.913 | 0.700 | 0.792 | 0.067 | 0.851 |
| marksheet | 120 | 0.683 | 0.893 | 0.417 | 0.568 | 0.050 | 0.702 |

Plots: `confusion_matrix.png`, `roc.png`.

## 2. Ablation (F1; last column = share of genuine images flagged)

Model-only uses its own val-tuned per-doc-type threshold (same FPR cap). OCR (hard-fail) = only the decisive rules; OCR (any flag) = any flag raised.

| variant | F1 overall | F1 aadhaar | F1 pan | F1 marksheet | genuine flagged (FPR) |
|---|---|---|---|---|---|
| model only | 0.660 | 0.752 | 0.757 | 0.429 | 0.072 |
| OCR rules (hard-fail) | 0.182 | 0.065 | 0.154 | 0.310 | 0.000 |
| OCR rules (any flag) | 0.377 | 0.356 | 0.209 | 0.505 | 0.078 |
| fused | 0.718 | 0.769 | 0.792 | 0.568 | 0.061 |

## 3. Recall per tamper type

| doc_type | tamper_type | n | model only | OCR rules (hard-fail) | OCR rules (any flag) | fused |
|---|---|---|---|---|---|---|
| aadhaar | text_edit | 14 | 0.000 | 0.000 | 0.000 | 0.071 |
| aadhaar | photo_swap | 14 | 1.000 | 0.000 | 0.000 | 1.000 |
| aadhaar | copy_move | 20 | 0.950 | 0.050 | 0.500 | 1.000 |
| aadhaar | splice | 12 | 0.417 | 0.083 | 0.250 | 0.417 |
| pan | text_edit | 10 | 0.000 | 0.100 | 0.100 | 0.100 |
| pan | photo_swap | 10 | 1.000 | 0.000 | 0.000 | 1.000 |
| pan | copy_move | 22 | 0.864 | 0.091 | 0.091 | 0.909 |
| pan | splice | 18 | 0.556 | 0.111 | 0.222 | 0.611 |
| marksheet | text_edit | 11 | 0.273 | 0.000 | 0.182 | 0.091 |
| marksheet | copy_move | 11 | 1.000 | 0.273 | 0.636 | 0.909 |
| marksheet | splice | 14 | 0.214 | 0.143 | 0.286 | 0.357 |
| marksheet | marks_edit_naive | 7 | 0.143 | 0.857 | 1.000 | 1.000 |
| marksheet | marks_edit_full | 17 | 0.000 | 0.000 | 0.294 | 0.118 |
| all | text_edit | 35 | 0.086 | 0.029 | 0.086 | 0.086 |
| all | photo_swap | 24 | 1.000 | 0.000 | 0.000 | 1.000 |
| all | copy_move | 53 | 0.925 | 0.113 | 0.358 | 0.943 |
| all | splice | 44 | 0.409 | 0.114 | 0.250 | 0.477 |
| all | marks_edit_naive | 7 | 0.143 | 0.857 | 1.000 | 1.000 |
| all | marks_edit_full | 17 | 0.000 | 0.000 | 0.294 | 0.118 |

## 4. Localisation (tampered images, threshold 0.5)

| scope | n | iou | dice | precision | recall |
|---|---|---|---|---|---|
| overall | 180 | 0.358 | 0.407 | 0.397 | 0.450 |
| aadhaar | 60 | 0.502 | 0.557 | 0.566 | 0.601 |
| marksheet | 60 | 0.070 | 0.094 | 0.080 | 0.129 |
| pan | 60 | 0.502 | 0.570 | 0.546 | 0.619 |

By tamper type:

| scope | n | iou | dice | precision | recall |
|---|---|---|---|---|---|
| text_edit | 35 | 0.011 | 0.018 | 0.072 | 0.019 |
| photo_swap | 24 | 0.996 | 0.998 | 0.996 | 1.000 |
| copy_move | 53 | 0.518 | 0.630 | 0.576 | 0.740 |
| splice | 44 | 0.288 | 0.347 | 0.330 | 0.389 |
| marks_edit_naive | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| marks_edit_full | 17 | 0.000 | 0.000 | 0.000 | 0.000 |

## 5. Robustness (subset of 120 test images)

| condition | n | fused_f1 | fused_recall | fused_fpr | model_only_f1 | genuine_hard_fail_rate | fused_f1_aadhaar | fused_f1_pan | fused_f1_marksheet |
|---|---|---|---|---|---|---|---|---|---|
| clean | 120 | 0.740 | 0.617 | 0.050 | 0.681 | 0.000 | 0.788 | 0.778 | 0.645 |
| jpeg90 | 120 | 0.750 | 0.650 | 0.083 | 0.688 | 0.000 | 0.765 | 0.833 | 0.647 |
| jpeg70 | 120 | 0.687 | 0.567 | 0.083 | 0.674 | 0.000 | 0.727 | 0.743 | 0.581 |
| jpeg50 | 120 | 0.736 | 0.650 | 0.117 | 0.667 | 0.000 | 0.757 | 0.811 | 0.625 |
| blur_k3 | 120 | 0.720 | 0.600 | 0.067 | 0.659 | 0.000 | 0.722 | 0.765 | 0.667 |
| blur_k5 | 120 | 0.718 | 0.700 | 0.250 | 0.688 | 0.033 | 0.750 | 0.800 | 0.588 |
| noise_s5 | 120 | 0.720 | 0.600 | 0.067 | 0.659 | 0.000 | 0.750 | 0.800 | 0.606 |
| noise_s10 | 120 | 0.729 | 0.650 | 0.133 | 0.667 | 0.000 | 0.765 | 0.778 | 0.649 |
| downscale50 | 120 | 0.691 | 0.783 | 0.483 | 0.688 | 0.117 | 0.667 | 0.800 | 0.634 |
| phone | 120 | 0.667 | 0.750 | 0.500 | 0.714 | 0.017 | 0.653 | 0.718 | 0.638 |

Plot: `robustness.png`.

## 6. Error analysis

`errors.png` (top: false positives, bottom: false negatives; image / GT mask / heatmap).

Worst false positives:

| path | doc_type | tamper_type | model_prob | ocr_flags | score |
|---|---|---|---|---|---|
| data/genuine/pan_0349.jpg | pan | none | 1.000 |  | 0.700 |
| data/genuine/aadhaar_0131.jpg | aadhaar | none | 0.871 |  | 0.610 |
| data/genuine/pan_0009.jpg | pan | none | 0.767 |  | 0.537 |
| data/genuine/pan_0070.jpg | pan | none | 0.762 |  | 0.534 |
| data/genuine/aadhaar_0233.jpg | aadhaar | none | 0.725 |  | 0.507 |
| data/genuine/marksheet_0126.jpg | marksheet | none | 0.693 |  | 0.485 |
| data/genuine/pan_0361.jpg | pan | none | 0.623 |  | 0.436 |
| data/genuine/marksheet_0093.jpg | marksheet | none | 0.444 | font_inconsistency | 0.411 |

Worst false negatives:

| path | doc_type | tamper_type | model_prob | ocr_flags | score |
|---|---|---|---|---|---|
| data/tampered/pan_0373.jpg | pan | splice | 0.206 |  | 0.144 |
| data/tampered/pan_0131.jpg | pan | text_edit | 0.209 |  | 0.146 |
| data/tampered/pan_0042.jpg | pan | text_edit | 0.212 |  | 0.148 |
| data/tampered/pan_0231.jpg | pan | text_edit | 0.214 |  | 0.150 |
| data/tampered/pan_0305.jpg | pan | splice | 0.218 |  | 0.153 |
| data/tampered/pan_0320.jpg | pan | text_edit | 0.221 |  | 0.155 |
| data/tampered/pan_0113.jpg | pan | copy_move | 0.225 |  | 0.157 |
| data/tampered/pan_0239.jpg | pan | copy_move | 0.226 |  | 0.158 |
