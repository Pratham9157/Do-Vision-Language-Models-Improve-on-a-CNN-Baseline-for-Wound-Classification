"""Image index, duplicate groups, cross-validation folds, low-label subsets and wound crops.

Raw layout (data/raw/):
    wseg/wseg_dataset/sample/<id>.jpg, wseg/wseg_dataset/mask/<id>.jpg   WoundSeg images + masks (HF subbareddyoota/wseg_dataset)
    wsnet_meta/woundids_types.csv                                       wound type per image id (github subbareddy248/WSNET)
    azh/{Train,Test}/{BG,N,D,P,S,V}/*.jpg + wound_locations_Labels_AZH_{Train,Test}.csv   (github uwm-bigdata)
The Mendeley 'Lower Limb and Feet Wound' release contains the same 2,686 wound images (renamed, no labels) plus a healthy
-feet set from another site; we use the WoundSeg originals because the labels are keyed by their ids.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from PIL import Image

from . import CLASSES

WSEG_LABELS = {"Venous Ulcer": "venous", "Pressure Ulcer": "pressure", "Trauma Wound": "trauma",
               "Diabetic Ulcer": "diabetic", "Surgical Wound": "surgical", "Other": "other"}
AZH_LABELS = {"D": "diabetic", "P": "pressure", "S": "surgical", "V": "venous", "BG": "background", "N": "normal_skin"}
DUP_THRESHOLD = 12            # dHash (16x16 = 256 bits) Hamming distance; 0-5 = re-encoded copies, gap until ~20
SEED = 42
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
CROP_SIZE = 384


# ---------------------------------------------------------------- hashing / duplicate groups
def dhash(path: str, n: int = 16) -> np.ndarray:
    im = Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def duplicate_groups(paths: list, threshold: int = DUP_THRESHOLD) -> np.ndarray:
    """Connected components of 'near-identical image' (dHash distance <= threshold). Returns a component id per path."""
    from scipy.sparse.csgraph import connected_components
    h = np.array([dhash(p) for p in paths])
    adj = np.zeros((len(h), len(h)), dtype=bool)
    for i in range(0, len(h), 512):                              # blockwise to bound memory
        adj[i:i + 512] = (h[i:i + 512, None, :] != h[None, :, :]).sum(-1) <= threshold
    return connected_components(adj, directed=False)[1]


# ---------------------------------------------------------------- index
def _wseg_index(raw: str) -> pd.DataFrame:
    d = os.path.join(raw, "wseg", "wseg_dataset")
    ids = sorted(f for f in os.listdir(os.path.join(d, "sample")) if f.lower().endswith(".jpg"))
    lab = pd.read_csv(os.path.join(raw, "wsnet_meta", "woundids_types.csv")).fillna("")
    lab = dict(zip(lab.imageid, lab.woundtype.map(lambda s: WSEG_LABELS.get(s, ""))))
    df = pd.DataFrame({"image_id": ["wseg_" + i[:-4] for i in ids],
                       "path": [os.path.join(d, "sample", i) for i in ids],
                       "mask_path": [os.path.join(d, "mask", i) for i in ids],
                       "label": [lab.get(i, "") for i in ids]})
    df["source"] = "wseg"
    comp = duplicate_groups(df.path.tolist())
    df["group"] = ["wseg_g%04d" % c for c in comp]
    # group status: conflicting labels -> excluded; labelled -> all labelled members kept, unlabelled copies excluded
    status = []
    for g, lbl in zip(df.group, df.label):
        labels = set(df.label[df.group == g]) - {""}
        if len(labels) > 1:
            status.append("excluded_conflicting_labels")
        elif lbl:
            status.append("labelled")
        elif labels:
            status.append("excluded_copy_of_labelled")
        else:
            status.append("unlabelled_pool")
    df["status"] = status
    return df


def _azh_index(raw: str) -> pd.DataFrame:
    rows = []
    for split in ("Train", "Test"):
        meta = pd.read_csv(os.path.join(raw, "azh", split, f"wound_locations_Labels_AZH_{split}.csv"))
        for idx, loc in zip(meta["index"], meta["Locations"]):
            cls, stem = re.split(r"[\\/]", idx)
            patient = stem.split("_")[0]
            rows.append({"image_id": f"azh_{split.lower()}_{cls}_{stem}",
                         "path": os.path.join(raw, "azh", split, cls, stem + ".jpg"), "mask_path": "",
                         "label": AZH_LABELS[cls], "source": "azh", "azh_split": split.lower(), "azh_location": int(loc),
                         # same patient/wound prefix within a class = same group; BG/N crops are unlinked
                         "group": f"azh_{cls}{patient}" if cls in "DPSV" else f"azh_{split}{cls}{stem}",
                         "status": "external"})
    return pd.DataFrame(rows)


def build_index(raw: str) -> pd.DataFrame:
    df = pd.concat([_wseg_index(raw), _azh_index(raw)], ignore_index=True)
    missing = [p for p in df.path if not os.path.isfile(p)]
    assert not missing, f"{len(missing)} image files missing, e.g. {missing[:3]}"
    return df


# ---------------------------------------------------------------- folds and low-label subsets
def make_folds(lab: pd.DataFrame, k: int = 5, seed: int = SEED) -> pd.Series:
    """Stratified, group-aware k folds for the labelled WoundSeg images (duplicates never split across folds)."""
    from sklearn.model_selection import StratifiedGroupKFold
    fold = pd.Series(-1, index=lab.index)
    for f, (_, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(lab, lab.label, lab.group)):
        fold.iloc[te] = f
    return fold


def inner_split(train: pd.DataFrame, seed: int = SEED, n_splits: int = 6) -> tuple:
    """(fit, val) index arrays: about 1/6 of the training fold's groups for early stopping / hyper-parameter choice."""
    from sklearn.model_selection import StratifiedGroupKFold
    fit, val = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(train, train.label, train.group))
    return train.index[fit], train.index[val]


def low_label_subset(train: pd.DataFrame, frac: float, seed: int = SEED, draw: int = 0) -> pd.Index:
    """Nested, class-stratified, group-whole subset holding about `frac` of each class (at least one group per class).
    Nested: the subset for a smaller fraction is contained in the subset for a larger one (same seed/draw)."""
    if frac >= 1.0:
        return train.index
    rng = np.random.default_rng([seed, draw])
    keep = []
    for c, part in train.groupby("label"):
        groups = part.group.unique()
        order = groups[rng.permutation(len(groups))]
        need, n = int(np.ceil(frac * len(part))), 0
        for g in order:
            members = part.index[part.group == g]
            keep.extend(members); n += len(members)
            if n >= need:
                break
    return pd.Index(sorted(keep))


def few_shot_subset(train: pd.DataFrame, k: int, seed: int = SEED, draw: int = 0) -> pd.Index:
    """k images per class (one per group), nested in k."""
    rng = np.random.default_rng([seed, 1000 + draw])
    keep = []
    for c, part in train.groupby("label"):
        firsts = part.drop_duplicates("group")
        keep.extend(firsts.index[rng.permutation(len(firsts))][:k])
    return pd.Index(sorted(keep))


# ---------------------------------------------------------------- crops
def context_box(mask: np.ndarray, scale: float = 1.3, min_frac: float = 0.35) -> tuple:
    """Square box around the wound mask: side = max(scale * longer bbox side, min_frac * shorter image side),
    centred on the wound, shifted to stay inside the image. Keeps the wound and its anatomical context while dropping
    most of the periphery (rulers, handwritten labels)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, W, H
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(round(max(scale * max(x1 - x0, y1 - y0), min_frac * min(H, W))))
    side = min(side, H, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), W - side)))
    top = int(round(min(max(cy - side / 2, 0), H - side)))
    return left, top, left + side, top + side


def prepare_images(df: pd.DataFrame, project: str, size: int = CROP_SIZE) -> pd.DataFrame:
    """Write data/processed/{crop,full}/<image_id>.jpg (RGB, size x size) and return df with relative paths.
    WoundSeg: crop = mask-guided context crop, full = whole photo. AZH images are already wound ROIs: crop = full."""
    out = df.copy()
    for view in ("crop", "full"):
        os.makedirs(os.path.join(project, "data", "processed", view), exist_ok=True)
    crop_rel, full_rel, boxes = [], [], []
    for r in out.itertuples():
        im = Image.open(r.path).convert("RGB")
        if r.source == "wseg":
            m = np.asarray(Image.open(r.mask_path).convert("L").resize(im.size, Image.NEAREST)) > 127
            box = context_box(m)
        else:
            box = (0, 0) + im.size
        boxes.append(box)
        for view, img, rel_list in (("crop", im.crop(box), crop_rel), ("full", im, full_rel)):
            rel = f"data/processed/{view}/{r.image_id}.jpg"
            t = os.path.join(project, rel)
            if not os.path.exists(t):
                img.resize((size, size), Image.BICUBIC).save(t, quality=95)
            rel_list.append(rel)
    out["crop_path"], out["full_path"] = crop_rel, full_rel
    out["crop_box"] = [",".join(map(str, b)) for b in boxes]
    return out


def load_index(project: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(project, "data", "processed", "index.csv"), keep_default_na=False)
    df["fold"] = df.fold.astype(int)
    return df


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df.source == "wseg") & (df.status == "labelled")].copy()
    out["y"] = out.label.map(CLASSES.index)
    return out


def pool(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.source == "wseg") & (df.status == "unlabelled_pool")].copy()


def azh_wounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df.source == "azh") & df.label.isin(CLASSES)].copy()
    out["y"] = out.label.map(CLASSES.index)
    return out
