"""Joint tampering localisation (U-Net mask) + classification (aux head) on RGB+ELA input."""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from common import ROOT, SEED, SPLITS
from dataset import ForgeryDataset

MODELS, RESULTS = ROOT / "models", ROOT / "results"


def build_model(encoder_weights: str | None = "imagenet") -> smp.Unet:
    kw = dict(encoder_name="resnet34", in_channels=4, classes=1, aux_params=dict(classes=1, dropout=0.2))
    try:
        return smp.Unet(encoder_weights=encoder_weights, **kw)
    except Exception as e:  # offline / download failure
        print(f"WARNING: could not load {encoder_weights} weights ({type(e).__name__}: {e}); training from scratch")
        return smp.Unet(encoder_weights=None, **kw)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


class JointLoss:
    """Pos-weighted BCE + Tversky (beta > alpha favours recall on tiny tampered regions) + 0.5 * aux BCE."""

    def __init__(self, pos_weight: float = 10.0, device="cpu"):
        self.dice = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.3, beta=0.7)
        self.pw = torch.tensor(pos_weight, device=device)

    def __call__(self, mask_logits, cls_logits, mask, label):
        return (F.binary_cross_entropy_with_logits(mask_logits, mask, pos_weight=self.pw) + self.dice(mask_logits, mask)
                + 0.5 * F.binary_cross_entropy_with_logits(cls_logits.squeeze(1), label))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot, n, ious, ys, ps = 0.0, 0, {t: [] for t in THRESHOLDS}, [], []
    for x, m, y, _ in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        ml, cl = model(x)
        tot += loss_fn(ml, cl, m, y).item() * len(x)
        n += len(x)
        prob = torch.sigmoid(ml)
        gt = m > 0.5
        for t in THRESHOLDS:
            pred = prob > t
            for i in torch.nonzero(y > 0.5).flatten().tolist():  # tampered images only
                inter, union = (pred[i] & gt[i]).sum().item(), (pred[i] | gt[i]).sum().item()
                ious[t].append(inter / union if union else 1.0)
        ys += y.cpu().tolist()
        ps += (torch.sigmoid(cl.squeeze(1)) > 0.5).int().cpu().tolist()
    iou = {t: float(np.mean(v)) for t, v in ious.items()}
    return tot / n, iou, f1_score(ys, ps, zero_division=0)


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(h["train_loss"]) + 1)
    ax[0].plot(ep, h["train_loss"], label="train")
    ax[0].plot(ep, h["val_loss"], label="val")
    ax[0].set_title("loss")
    ax[1].plot(ep, h["val_iou"], label="thr 0.5")
    ax[1].plot(ep, h["val_iou_tuned"], label="best thr")
    ax[1].legend()
    ax[1].set_title("val pixel IoU (tampered only)")
    ax[2].plot(ep, h["val_f1"])
    ax[2].set_title("val classification F1")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 64 train / 32 val samples; nothing saved to models/")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crops", action="store_true", help="train on native-resolution 512x512 crops (tiled inference)")
    ap.add_argument("--repeat", type=int, default=2, help="crops drawn per training image per epoch (--crops)")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    epochs, bs = 15, 16
    if device.type == "cpu":
        epochs, bs = 5, 8
        print("WARNING: no CUDA/MPS device found - training on CPU with batch_size=8, 5 epochs. This will be slow.")
    if a.crops and device.type != "cpu":
        epochs = 30
    epochs, bs = a.epochs or epochs, a.batch_size or bs
    if a.smoke:
        epochs = 1
    print(f"device={device} epochs={epochs} batch_size={bs}")

    if a.crops:
        from crops import CropDataset
        tr = CropDataset(SPLITS / "train.csv", train=True, repeat=a.repeat)
        va = CropDataset(SPLITS / "val.csv", train=False)
    else:
        tr, va = ForgeryDataset(SPLITS / "train.csv", train=True), ForgeryDataset(SPLITS / "val.csv", train=False)
    if a.smoke:
        g = np.random.default_rng(SEED)
        tr, va = Subset(tr, g.permutation(len(tr))[:64].tolist()), Subset(va, g.permutation(len(va))[:32].tolist())
    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va, batch_size=bs, shuffle=False, **kw)

    model = build_model().to(device)
    loss_fn = JointLoss(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))  # stepped per batch
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    hist = {k: [] for k in ("train_loss", "val_loss", "val_iou", "val_iou_tuned", "val_f1")}
    best = -1.0
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    suffix = ("_crop" if a.crops else "") + ("_smoke" if a.smoke else "")
    best_path = MODELS / ("unet_crop.pt" if a.crops else "unet_best.pt")
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, m, y, _) in enumerate(tl):
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                ml, cl = model(x)
            loss = loss_fn(ml.float(), cl.float(), m, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            if i % 20 == 0:
                print(f"  ep{ep} step {i}/{len(tl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        vloss, ious, vf1 = evaluate(model, vl, loss_fn, device)
        thr = max(ious, key=ious.get)
        viou, vtuned = ious[0.5], ious[thr]
        for k, v in zip(hist, (tot / n, vloss, viou, vtuned, vf1)):
            hist[k].append(v)
        print(f"epoch {ep}/{epochs}: train_loss {tot / n:.4f} val_loss {vloss:.4f} val_IoU@0.5 {viou:.4f} "
              f"best-thr IoU {vtuned:.4f} (thr {thr}) val_F1 {vf1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        ckpt = {"state_dict": model.state_dict(), "encoder_name": "resnet34", "epoch": ep, "val_iou": viou,
                "val_iou_tuned": vtuned, "mask_thr": thr, "val_f1": vf1,
                "mode": "tiled" if a.crops else "letterbox"}
        if a.smoke:
            torch.save(ckpt, RESULTS / f"unet{suffix}.pt")
        elif vtuned > best:  # select on IoU at the best threshold (IoU@0.5 is ~0 while probabilities are still low)
            best = vtuned
            torch.save(ckpt, best_path)
            print(f"  saved {best_path.relative_to(ROOT)} (val IoU {vtuned:.4f} @ thr {thr})", flush=True)
        plot_curves(hist, RESULTS / f"curves{suffix}.png")
        (RESULTS / f"history{suffix}.json").write_text(json.dumps(hist))
    print(f"FINAL: best val IoU (tuned thr) {max(hist['val_iou_tuned']):.4f} | last-epoch IoU@0.5 {hist['val_iou'][-1]:.4f} "
          f"val F1 {hist['val_f1'][-1]:.4f}")


if __name__ == "__main__":
    main()
