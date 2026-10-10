"""Load a trained checkpoint and run it on a BGR image at its original size."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from common import ROOT
from crops import tile_predict
from dataset import letterbox, letterbox_geometry, make_input
from train_unet import pick_device

CROP_MODEL, LETTERBOX_MODEL = ROOT / "models" / "unet_crop.pt", ROOT / "models" / "unet_best.pt"


def load_model(path=None, device: torch.device | None = None):
    """Loads the native-resolution crop model (tiled inference) when it exists, else the letterbox model."""
    path = Path(path or (CROP_MODEL if CROP_MODEL.exists() else LETTERBOX_MODEL))
    device = device or pick_device()
    ck = torch.load(path, map_location=device, weights_only=True)
    model = smp.Unet(encoder_name=ck["encoder_name"], encoder_weights=None, in_channels=4, classes=1,
                     aux_params=dict(classes=1, dropout=0.2))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    model.mode = ck.get("mode", "letterbox")
    model.file = path.name
    return model


@torch.no_grad()
def predict(img_bgr: np.ndarray, model=None) -> dict:
    """{'prob': tamper probability (aux head), 'mask_prob': HxW float32 map at the original image size}."""
    model = model or load_model()
    device = next(model.parameters()).device
    h, w = img_bgr.shape[:2]
    if getattr(model, "mode", "letterbox") == "tiled":
        mask, prob = tile_predict(model, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), device)
        return {"prob": prob, "mask_prob": mask.astype(np.float32)}
    rgb, _ = letterbox(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), None)  # same preprocessing as ForgeryDataset
    ml, cl = model(make_input(rgb)[None].to(device))
    full = torch.sigmoid(ml)[0, 0].cpu().numpy()
    nw, nh, top, left, _ = letterbox_geometry(h, w)
    mask = cv2.resize(full[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR)
    return {"prob": float(torch.sigmoid(cl)[0, 0]), "mask_prob": mask.astype(np.float32)}
