"""Supervised CNN baseline: ImageNet ResNet50 fine-tuned end to end.

Fixed recipe for every run (pre-registered, not tuned on test folds): 224 px input, class-balanced sampling,
AdamW (backbone 1e-4, new head 1e-3, wd 1e-4), cosine schedule, label smoothing 0.1, AMP, at most 30 epochs of
max(n/32, 20) steps, early stopping (patience 6) on inner-validation macro-F1, best checkpoint restored.
Images are preloaded once as uint8 tensors; augmentation runs on the GPU.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import torch
from PIL import Image

CFG = dict(arch="resnet50", img=224, batch=32, epochs=30, min_steps=20, lr_backbone=1e-4, lr_head=1e-3,
           weight_decay=1e-4, label_smoothing=0.1, patience=6, seed=42)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LOAD = 256                                   # stored resolution of the preloaded tensors


def load_images(project: str, rel_paths: list, size: int = LOAD) -> torch.Tensor:
    arr = np.stack([np.asarray(Image.open(os.path.join(project, p)).convert("RGB").resize((size, size), Image.BICUBIC))
                    for p in rel_paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()        # uint8 N,3,H,W


def build_model(n_classes: int, arch: str = CFG["arch"], pretrained: bool = True) -> torch.nn.Module:
    import torchvision
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    elif arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise KeyError(arch)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    return m


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / 255.0 - MEAN.to(x.device)) / STD.to(x.device)


def augment(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Random resized crop (scale 0.6-1), flips, 90-degree rotations, brightness/contrast/saturation jitter. x uint8."""
    import torchvision.transforms.v2.functional as F
    n, _, H, W = x.shape
    out = torch.empty(n, 3, size, size, device=x.device)
    r = torch.rand(n, 8, generator=gen).tolist()
    for i in range(n):
        s = math.sqrt(0.6 + 0.4 * r[i][0])
        h = w = int(H * s)
        top, left = int((H - h) * r[i][1]), int((W - w) * r[i][2])
        im = F.resized_crop(x[i], top, left, h, w, [size, size], antialias=True)
        if r[i][3] < 0.5:
            im = F.horizontal_flip(im)
        if r[i][4] < 0.5:
            im = F.vertical_flip(im)
        im = torch.rot90(im, int(r[i][5] * 4), dims=(1, 2))
        im = im.float() / 255.0
        im = F.adjust_brightness(im, 0.85 + 0.3 * r[i][6])
        im = F.adjust_contrast(im, 0.85 + 0.3 * r[i][7])
        out[i] = im
    return (out - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, x: torch.Tensor, size: int = CFG["img"], batch: int = 64, tta: bool = True) -> np.ndarray:
    """Softmax probabilities; TTA = mean over identity and horizontal flip."""
    import torch.nn.functional as Fn
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for b in range(0, len(x), batch):
        xb = _normalize(Fn.interpolate(x[b:b + batch].to(dev).float(), size=(size, size), mode="bilinear",
                                       antialias=True, align_corners=False) )
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            p = model(xb).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(xb, dims=[3])).float().softmax(1)) / 2
        outs.append(p.cpu())
    return torch.cat(outs).numpy()


def train(x_fit: torch.Tensor, y_fit: np.ndarray, x_val: torch.Tensor, y_val: np.ndarray, n_classes: int,
          device: str = "cuda", log=print, model: torch.nn.Module | None = None, **kw) -> tuple:
    """Train one model; returns (model with best weights, history list)."""
    from sklearn.metrics import f1_score
    cfg = {**CFG, **kw}
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model = (model or build_model(n_classes, cfg["arch"])).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": cfg["lr_backbone"]}, {"params": head, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    steps = max(math.ceil(len(y_fit) / cfg["batch"]), cfg["min_steps"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lr_backbone"], cfg["lr_head"]],
                                                total_steps=steps * cfg["epochs"], pct_start=0.1)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    counts = np.bincount(y_fit, minlength=n_classes)
    w = np.where(counts[y_fit] > 0, 1.0 / counts[y_fit], 0); w = torch.tensor(w / w.sum())
    y_t = torch.tensor(y_fit)
    best, best_state, wait, hist, t0 = -1.0, None, 0, [], time.time()
    for ep in range(cfg["epochs"]):
        model.train(); tl = 0.0
        for _ in range(steps):
            idx = torch.multinomial(w, cfg["batch"], replacement=True, generator=gen)
            xb = augment(x_fit[idx].to(device), cfg["img"], gen)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(xb), y_t[idx].to(device),
                                                         label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        pv = predict(model, x_val, cfg["img"], tta=False)
        f1 = f1_score(y_val, pv.argmax(1), labels=np.arange(n_classes), average="macro", zero_division=0)
        hist.append({"epoch": ep, "train_loss": tl / steps, "val_macro_f1": f1, "seconds": round(time.time() - t0, 1)})
        log(f"    ep {ep:2d} loss {tl / steps:.3f} val macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)")
        if f1 > best + 1e-4:
            best, wait = f1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist
