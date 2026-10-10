"""ForgeryDataset: letterboxed 512x512 RGB+ELA tensors with aligned masks."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import DOC_TYPES, ROOT
from features import ela

SIZE = 512


def letterbox_geometry(h: int, w: int, size: int = SIZE):
    """(new_w, new_h, top, left, scale) of the aspect-preserving resize + centred padding."""
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return nw, nh, (size - nh) // 2, (size - nw) // 2, s


def letterbox(img: np.ndarray, mask: np.ndarray | None, size: int = SIZE):
    """Resize keeping aspect ratio, pad to size x size with the image's border colour (mask padded with 0)."""
    h, w = img.shape[:2]
    nw, nh, top, left, s = letterbox_geometry(h, w, size)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    color = tuple(int(c) for c in np.median(border, axis=0))
    pad = dict(top=top, bottom=size - nh - top, left=left, right=size - nw - left)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    img = cv2.copyMakeBorder(cv2.resize(img, (nw, nh), interpolation=interp), **pad,
                             borderType=cv2.BORDER_CONSTANT, value=color)
    if mask is not None:
        mask = cv2.copyMakeBorder(cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST), **pad,
                                  borderType=cv2.BORDER_CONSTANT, value=0)
    return img, mask


def make_input(img_rgb: np.ndarray) -> torch.Tensor:
    """512x512 RGB uint8 (already letterboxed/augmented) -> 4x512x512 float tensor: RGB in [0,1] + ELA in [0,1]."""
    elf = ela(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)) / 255.0
    x = np.concatenate([img_rgb.astype(np.float32) / 255.0, elf[..., None]], axis=2).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(x))


def train_augment() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=3, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.5),
    ])  # no flips


class ForgeryDataset(Dataset):
    def __init__(self, csv_path, train: bool = False):
        self.df = pd.read_csv(csv_path, keep_default_na=False)
        self.train = train
        self.aug = train_augment() if train else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB)
        if r.mask_path:
            mask = (cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[:2], np.uint8)
        img, mask = letterbox(img, mask)
        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = make_input(img)  # ELA computed after augmentation
        return (x, torch.from_numpy(mask[None].astype(np.float32)),
                torch.tensor(float(r.label)), DOC_TYPES.index(r.doc_type))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from common import SPLITS
    for train in (True, False):
        x, m, y, d = next(iter(DataLoader(ForgeryDataset(SPLITS / "train.csv", train=train), batch_size=8, shuffle=True)))
        print(f"train={train}: x {tuple(x.shape)} {x.dtype} [{x.min():.2f},{x.max():.2f}]  mask {tuple(m.shape)} "
              f"vals={sorted(m.unique().tolist())}  label {tuple(y.shape)}  doc_type {d.tolist()}")
        print("  mask>0 for tampered only:", all((m[i].sum() > 0) == bool(y[i]) for i in range(8)))
