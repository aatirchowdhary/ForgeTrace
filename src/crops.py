"""Native-resolution crop training data and tiled inference.

Images are never resized, and crop/tile origins are multiples of 8, so the JPEG 8x8 block grid that ELA relies on
stays aligned and a small edited field keeps its true size in pixels.
"""
from __future__ import annotations

import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import DOC_TYPES, ROOT, SEED
from dataset import make_input

CROP = 512
STRIDE = 256
MIN_TAMPER_PIXELS = 32  # a crop counts as "tampered" only if it holds at least this many tampered pixels


def border_color(img: np.ndarray) -> tuple:
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(c) for c in np.median(border, axis=0))


def pad_image(img: np.ndarray, mask: np.ndarray | None = None, size: int = CROP):
    """Pad bottom/right (so the block grid origin does not move) to >= size and a multiple of 8."""
    h, w = img.shape[:2]
    H, W = max(size, -(-h // 8) * 8), max(size, -(-w // 8) * 8)
    kw = dict(top=0, left=0, bottom=H - h, right=W - w, borderType=cv2.BORDER_CONSTANT)
    img = cv2.copyMakeBorder(img, value=border_color(img), **kw)
    if mask is not None:
        mask = cv2.copyMakeBorder(mask, value=0, **kw)
    return img, mask


def crop_augment() -> A.Compose:
    # no rotation / resize / flips: they would break the JPEG grid alignment ELA depends on
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
    ])


class CropDataset(Dataset):
    """512x512 native-resolution crops. Tampered images are cropped around a random tampered pixel with probability
    p_pos; otherwise the crop is uniformly random. The label is whether the crop itself contains tampering.
    Returns the same (x, mask, label, doc_type) tuple as ForgeryDataset."""

    def __init__(self, csv_path, train: bool = False, repeat: int = 1, p_pos: float | None = None):
        self.df = pd.read_csv(csv_path, keep_default_na=False)
        self.train, self.repeat = train, repeat if train else 1
        self.p_pos = p_pos if p_pos is not None else (0.8 if train else 0.5)
        self.aug = crop_augment() if train else None
        # Tamper locations per doc type. Genuine images are cropped around locations drawn from this same pool, so a
        # crop's position (and the edge-clamping it causes) carries no information about the label.
        self.pool: dict[str, list] = {dt: [] for dt in DOC_TYPES}
        for r in self.df[self.df.mask_path != ""].itertuples():
            ys, xs = np.nonzero(cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127)
            if len(ys):
                k = np.random.default_rng(SEED).choice(len(ys), min(10, len(ys)), replace=False)
                self.pool[r.doc_type] += list(zip(xs[k].tolist(), ys[k].tolist()))

    def __len__(self) -> int:
        return len(self.df) * self.repeat

    def __getitem__(self, i: int):
        i %= len(self.df)
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB)
        mask = ((cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                if r.mask_path else np.zeros(img.shape[:2], np.uint8))
        img, mask = pad_image(img, mask)
        H, W = img.shape[:2]
        rng = random if self.train else random.Random(SEED * 1000 + i)  # val crops are fixed
        ys, xs = np.nonzero(mask)
        pool = self.pool[r.doc_type]
        if len(ys) and rng.random() < self.p_pos:  # tampered image: centre on a tampered pixel
            k = rng.randrange(len(ys))
            x0, y0 = xs[k] - rng.randint(24, CROP - 24), ys[k] - rng.randint(24, CROP - 24)
        elif not len(ys) and pool and rng.random() < self.p_pos:  # genuine image: centre on a typical tamper location
            px, py = pool[rng.randrange(len(pool))]
            x0, y0 = px - rng.randint(24, CROP - 24), py - rng.randint(24, CROP - 24)
        else:
            x0, y0 = rng.randint(0, W - CROP), rng.randint(0, H - CROP)
        x0, y0 = int(np.clip(x0, 0, W - CROP)) // 8 * 8, int(np.clip(y0, 0, H - CROP)) // 8 * 8
        img, mask = img[y0:y0 + CROP, x0:x0 + CROP], mask[y0:y0 + CROP, x0:x0 + CROP]
        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        label = float(mask.sum() >= MIN_TAMPER_PIXELS)
        return (make_input(img), torch.from_numpy(mask[None].astype(np.float32)), torch.tensor(label),
                DOC_TYPES.index(r.doc_type))


@torch.no_grad()
def tile_predict(model, img_rgb: np.ndarray, device, tile: int = CROP, stride: int = STRIDE, batch: int = 8):
    """Run the model over overlapping native-resolution tiles.
    Returns (mask_prob HxW at the original size = average over overlapping tiles, image prob = max tile prob)."""
    h, w = img_rgb.shape[:2]
    pimg, _ = pad_image(img_rgb)
    H, W = pimg.shape[:2]

    def starts(n):
        s = list(range(0, n - tile + 1, stride))
        return s if s[-1] == n - tile else s + [n - tile]  # n - tile is a multiple of 8, so the grid stays aligned

    coords = [(y, x) for y in starts(H) for x in starts(W)]
    acc, cnt, probs = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), []
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = torch.stack([make_input(pimg[y:y + tile, x_:x_ + tile]) for y, x_ in chunk]).to(device)
        ml, cl = model(x)
        m, p = torch.sigmoid(ml)[:, 0].cpu().numpy(), torch.sigmoid(cl)[:, 0].cpu().numpy()
        for (y, x_), mm, pp in zip(chunk, m, p):
            acc[y:y + tile, x_:x_ + tile] += mm
            cnt[y:y + tile, x_:x_ + tile] += 1
            probs.append(float(pp))
    return (acc / cnt)[:h, :w], max(probs)
