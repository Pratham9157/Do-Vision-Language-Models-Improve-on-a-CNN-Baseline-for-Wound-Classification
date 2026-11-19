"""Multiclass metrics, group-bootstrap confidence intervals and the paired comparison rule.

Prediction tables have one row per image: image_id, group, y (class index into CLASSES), p_<class> for every class.
Resampling unit = duplicate group (WoundSeg) or patient/wound prefix (AZH), so near-identical images move together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CLASSES

N_BOOT = 2000
SEED = 42
PCOLS = [f"p_{c}" for c in CLASSES]


def probs(df: pd.DataFrame, classes=None) -> np.ndarray:
    """Probability matrix; with `classes`, keep only those columns and renormalise (e.g. the 4 classes AZH shares)."""
    cols = [f"p_{c}" for c in (classes or CLASSES)]
    p = df[cols].to_numpy(float)
    return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)


def labels_for(df: pd.DataFrame, classes=None) -> np.ndarray:
    """y re-indexed into `classes` (rows whose class is not in `classes` must be removed beforehand)."""
    if not classes:
        return df.y.to_numpy(int)
    m = {CLASSES.index(c): i for i, c in enumerate(classes)}
    return df.y.map(m).to_numpy(int)


def _f1_parts(y: np.ndarray, pred: np.ndarray, k: int) -> tuple:
    c = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp, support, predicted = np.diag(c).astype(float), c.sum(1).astype(float), c.sum(0).astype(float)
    den = support + predicted
    f1 = np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)
    return f1, tp, support


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Same definitions as sklearn (macro-F1 over all classes, zero_division=0; balanced accuracy = mean recall of
    classes present), computed from one confusion matrix because the bootstrap calls this thousands of times."""
    y = np.asarray(y, int); pred = p.argmax(1); k = p.shape[1]
    f1, tp, support = _f1_parts(y, pred, k)
    present = support > 0
    return {"n": len(y), "macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / max(len(y), 1)),
            "balanced_accuracy": float((tp[present] / support[present]).mean()),
            "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1))}


def per_class_f1(y: np.ndarray, p: np.ndarray, classes=None) -> dict:
    f = _f1_parts(np.asarray(y, int), p.argmax(1), p.shape[1])[0]
    return {f"f1_{c}": v for c, v in zip(classes or CLASSES, f)}


def _group_index(groups: np.ndarray) -> list:
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def summarize(df: pd.DataFrame, classes=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    y, p = labels_for(df, classes), probs(df, classes)
    s = scores(y, p)
    s.update(per_class_f1(y, p, classes))
    rng = np.random.default_rng(seed)
    idx = _group_index(df.group.to_numpy())
    vals = {"macro_f1": [], "accuracy": []}
    for _ in range(n_boot):
        r = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        sc = scores(y[r], p[r])
        for k in vals:
            vals[k].append(sc[k])
    for k, v in vals.items():
        s[f"{k}_ci_lo"], s[f"{k}_ci_hi"] = np.percentile(v, [2.5, 97.5])
    return s


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, metric: str = "macro_f1", classes=None,
                      n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """metric(A) - metric(B) on the same images; 95% CI and two-sided bootstrap p-value by resampling groups.
    Decision rule: A better iff the CI excludes 0 on the positive side (after Holm for primary tests)."""
    cols = ["image_id", "group", "y"] + PCOLS
    m = a[cols].merge(b[cols], on=["image_id", "group", "y"], suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "A and B must contain the same images"
    y = labels_for(m, classes)
    sel = [f"p_{c}" for c in (classes or CLASSES)]
    pa = np.array(m[[s + "_a" for s in sel]], dtype=float); pb = np.array(m[[s + "_b" for s in sel]], dtype=float)
    pa /= pa.sum(1, keepdims=True); pb /= pb.sum(1, keepdims=True)
    diff = scores(y, pa)[metric] - scores(y, pb)[metric]
    rng = np.random.default_rng(seed)
    idx = _group_index(m.group.to_numpy())
    vals = np.empty(n_boot)
    for i in range(n_boot):
        r = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        vals[i] = scores(y[r], pa[r])[metric] - scores(y[r], pb[r])[metric]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p_two = float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())))
    return {"metric": metric, "diff": float(diff), "ci_lo": float(lo), "ci_hi": float(hi), "p_boot": p_two,
            "a_better": bool(lo > 0), "b_better": bool(hi < 0), "n": len(m)}


def holm(pvals: list, alpha: float = 0.05) -> list:
    """Holm step-down: returns reject flags in the input order."""
    order = np.argsort(pvals)
    reject = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            reject[i] = True
        else:
            break
    return reject


def prediction_frame(meta: pd.DataFrame, p: np.ndarray, **extra) -> pd.DataFrame:
    """Standard prediction table from index rows (image_id, group, y) and an (n, len(CLASSES)) probability matrix."""
    out = meta[["image_id", "group", "y"]].reset_index(drop=True).copy()
    out[PCOLS] = np.asarray(p, float)
    for k, v in extra.items():
        out[k] = v
    return out
