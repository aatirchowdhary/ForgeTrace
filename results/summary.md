# doc-forgery: test-split evaluation

Test split: 360 images (180 genuine / 180 tampered). Thresholds were tuned on the validation split only (per-doc-type, genuine false-positive rate capped at 10%): aadhaar 0.432, pan 0.164, marksheet 0.463.

## Observations

- Fused detector on the test split: F1 0.701, recall 0.561, precision 0.935, false-positive rate 3.9%, ROC-AUC 0.819.
- `marks_edit_naive` (marksheet, n=7): OCR rules catch 86%, model alone 0%, fused 86%.
- `marks_edit_full` (marksheet, n=17): OCR rules catch 0%, model alone 0%, fused 0%.
- Overall F1 by variant: model only 0.650, OCR rules (hard-fail) 0.182, OCR rules (any flag) 0.377, fused 0.701.
- Weakest tamper types (fused recall, pooled): marks_edit_full 0%, text_edit 11%.
- Localisation (tampered only, threshold 0.5): mean IoU 0.360, Dice 0.414.
- Robustness (subset n=120): clean fused F1 0.729; largest drop under phone (0.621); largest FPR phone (51.7%).
- Error analysis: 7 false positives and 8 false negatives shown in `errors.png`.

## 1. Classification (fused verdict)

| scope | n | accuracy | precision | recall | f1 | fpr | roc_auc |
|---|---|---|---|---|---|---|---|
| overall | 360 | 0.761 | 0.935 | 0.561 | 0.701 | 0.039 | 0.819 |
| aadhaar | 120 | 0.783 | 0.972 | 0.583 | 0.729 | 0.017 | 0.805 |
| pan | 120 | 0.842 | 0.936 | 0.733 | 0.822 | 0.050 | 0.895 |
| marksheet | 120 | 0.658 | 0.880 | 0.367 | 0.518 | 0.050 | 0.702 |

Plots: `confusion_matrix.png`, `roc.png`.

## 2. Ablation (F1; last column = share of genuine images flagged)

Model-only uses its own val-tuned per-doc-type threshold (same FPR cap). OCR (hard-fail) = only the decisive rules; OCR (any flag) = any flag raised.

| variant | F1 overall | F1 aadhaar | F1 pan | F1 marksheet | genuine flagged (FPR) |
|---|---|---|---|---|---|
| model only | 0.650 | 0.702 | 0.807 | 0.351 | 0.039 |
| OCR rules (hard-fail) | 0.182 | 0.065 | 0.154 | 0.310 | 0.000 |
| OCR rules (any flag) | 0.377 | 0.356 | 0.209 | 0.505 | 0.078 |
| fused | 0.701 | 0.729 | 0.822 | 0.518 | 0.039 |

## 3. Recall per tamper type

| doc_type | tamper_type | n | model only | OCR rules (hard-fail) | OCR rules (any flag) | fused |
|---|---|---|---|---|---|---|
| aadhaar | text_edit | 14 | 0.000 | 0.000 | 0.000 | 0.000 |
| aadhaar | photo_swap | 14 | 1.000 | 0.000 | 0.000 | 1.000 |
| aadhaar | copy_move | 20 | 0.700 | 0.050 | 0.500 | 0.750 |
| aadhaar | splice | 12 | 0.417 | 0.083 | 0.250 | 0.500 |
| pan | text_edit | 10 | 0.400 | 0.100 | 0.100 | 0.400 |
| pan | photo_swap | 10 | 1.000 | 0.000 | 0.000 | 1.000 |
| pan | copy_move | 22 | 0.864 | 0.091 | 0.091 | 0.818 |
| pan | splice | 18 | 0.611 | 0.111 | 0.222 | 0.667 |
| marksheet | text_edit | 11 | 0.000 | 0.000 | 0.182 | 0.000 |
| marksheet | copy_move | 11 | 0.727 | 0.273 | 0.636 | 0.818 |
| marksheet | splice | 14 | 0.357 | 0.143 | 0.286 | 0.500 |
| marksheet | marks_edit_naive | 7 | 0.000 | 0.857 | 1.000 | 0.857 |
| marksheet | marks_edit_full | 17 | 0.000 | 0.000 | 0.294 | 0.000 |
| all | text_edit | 35 | 0.114 | 0.029 | 0.086 | 0.114 |
| all | photo_swap | 24 | 1.000 | 0.000 | 0.000 | 1.000 |
| all | copy_move | 53 | 0.774 | 0.113 | 0.358 | 0.792 |
| all | splice | 44 | 0.477 | 0.114 | 0.250 | 0.568 |
| all | marks_edit_naive | 7 | 0.000 | 0.857 | 1.000 | 0.857 |
| all | marks_edit_full | 17 | 0.000 | 0.000 | 0.294 | 0.000 |

## 4. Localisation (tampered images, threshold 0.5)

| scope | n | iou | dice | precision | recall |
|---|---|---|---|---|---|
| overall | 180 | 0.360 | 0.414 | 0.475 | 0.442 |
| aadhaar | 60 | 0.446 | 0.488 | 0.489 | 0.527 |
| marksheet | 60 | 0.170 | 0.228 | 0.396 | 0.236 |
| pan | 60 | 0.464 | 0.525 | 0.540 | 0.563 |

By tamper type:

| scope | n | iou | dice | precision | recall |
|---|---|---|---|---|---|
| text_edit | 35 | 0.024 | 0.028 | 0.033 | 0.030 |
| photo_swap | 24 | 0.987 | 0.993 | 0.987 | 1.000 |
| copy_move | 53 | 0.449 | 0.548 | 0.545 | 0.635 |
| splice | 44 | 0.311 | 0.370 | 0.395 | 0.409 |
| marks_edit_naive | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| marks_edit_full | 17 | 0.167 | 0.252 | 0.847 | 0.170 |

## 5. Robustness (subset of 120 test images)

| condition | n | fused_f1 | fused_recall | fused_fpr | model_only_f1 | genuine_hard_fail_rate | fused_f1_aadhaar | fused_f1_pan | fused_f1_marksheet |
|---|---|---|---|---|---|---|---|---|---|
| clean | 120 | 0.729 | 0.583 | 0.017 | 0.696 | 0.000 | 0.750 | 0.824 | 0.600 |
| jpeg90 | 120 | 0.729 | 0.583 | 0.017 | 0.681 | 0.000 | 0.710 | 0.824 | 0.645 |
| jpeg70 | 120 | 0.747 | 0.617 | 0.033 | 0.729 | 0.000 | 0.750 | 0.833 | 0.645 |
| jpeg50 | 120 | 0.747 | 0.617 | 0.033 | 0.688 | 0.000 | 0.750 | 0.811 | 0.667 |
| blur_k3 | 120 | 0.692 | 0.617 | 0.167 | 0.667 | 0.000 | 0.710 | 0.711 | 0.645 |
| blur_k5 | 120 | 0.683 | 0.683 | 0.317 | 0.667 | 0.033 | 0.710 | 0.679 | 0.667 |
| noise_s5 | 120 | 0.750 | 0.600 | 0.000 | 0.681 | 0.000 | 0.710 | 0.857 | 0.667 |
| noise_s10 | 120 | 0.667 | 0.533 | 0.067 | 0.615 | 0.000 | 0.645 | 0.743 | 0.600 |
| downscale50 | 120 | 0.686 | 0.783 | 0.500 | 0.667 | 0.117 | 0.722 | 0.667 | 0.683 |
| phone | 120 | 0.621 | 0.683 | 0.517 | 0.619 | 0.017 | 0.578 | 0.702 | 0.533 |

Plot: `robustness.png`.

## 6. Error analysis

`errors.png` (top: false positives, bottom: false negatives; image / GT mask / heatmap).

Worst false positives:

| path | doc_type | tamper_type | model_prob | ocr_flags | score |
|---|---|---|---|---|---|
| data/genuine/aadhaar_0223.jpg | aadhaar | none | 0.745 |  | 0.521 |
| data/genuine/marksheet_0394.jpg | marksheet | none | 0.560 | font_inconsistency | 0.492 |
| data/genuine/marksheet_0278.jpg | marksheet | none | 0.681 |  | 0.477 |
| data/genuine/marksheet_0386.jpg | marksheet | none | 0.538 | font_inconsistency | 0.476 |
| data/genuine/pan_0131.jpg | pan | none | 0.302 |  | 0.212 |
| data/genuine/pan_0396.jpg | pan | none | 0.260 |  | 0.182 |
| data/genuine/pan_0381.jpg | pan | none | 0.241 |  | 0.169 |

Worst false negatives:

| path | doc_type | tamper_type | model_prob | ocr_flags | score |
|---|---|---|---|---|---|
| data/tampered/pan_0005.jpg | pan | splice | 0.099 |  | 0.069 |
| data/tampered/aadhaar_0000.jpg | aadhaar | text_edit | 0.109 |  | 0.076 |
| data/tampered/pan_0113.jpg | pan | copy_move | 0.109 |  | 0.077 |
| data/tampered/pan_0361.jpg | pan | splice | 0.111 |  | 0.078 |
| data/tampered/aadhaar_0113.jpg | aadhaar | splice | 0.114 |  | 0.080 |
| data/tampered/aadhaar_0278.jpg | aadhaar | text_edit | 0.117 |  | 0.082 |
| data/tampered/aadhaar_0116.jpg | aadhaar | text_edit | 0.117 |  | 0.082 |
| data/tampered/aadhaar_0239.jpg | aadhaar | splice | 0.117 |  | 0.082 |
