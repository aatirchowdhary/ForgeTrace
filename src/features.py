"""Error Level Analysis map and its summary statistics."""
from __future__ import annotations

import cv2
import numpy as np


def ela(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    """JPEG re-save at `quality`, absolute difference, mean over channels, scaled so the max is 255 (float32 HxW)."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = np.abs(img_bgr.astype(np.float32) - resaved.astype(np.float32)).mean(axis=2)
    return (diff * (255.0 / max(float(diff.max()), 1.0))).astype(np.float32)


def ela_stats(ela_map: np.ndarray) -> np.ndarray:
    """mean, std, max, p90/p95/p99 and a 16-bin normalised histogram over [0, 255] -> 22 floats."""
    p = np.percentile(ela_map, [90, 95, 99])
    hist, _ = np.histogram(ela_map, bins=16, range=(0, 255))
    hist = hist / max(hist.sum(), 1)
    return np.concatenate([[ela_map.mean(), ela_map.std(), ela_map.max()], p, hist]).astype(np.float32)
