"""Classifier heads on frozen vision-language embeddings (all CPU, seconds per fit).

One estimator covers every trained head: multinomial logistic regression whose weights are pulled toward a
language-derived prior P instead of toward zero,

    loss = class-balanced cross-entropy(X W^T + b) + lam * ||W - P||^2

  * lam -> infinity : the prior alone (zero-shot; no labels used)
  * P = 0           : ordinary ridge logistic regression (language-free linear probe)
  * P from text     : labels refine what language already says; few labels -> stays near the prior

lam is picked by group-aware inner cross-validation on the training subset only (macro-F1); with too few images per
class for that, a pre-registered default is used.

Heads (name -> features, prior):
  zs_names      image emb      zero-shot with class-name prompts (no training)
  zs_desc       image emb      zero-shot with concept-description prompts ("classification by description")
  cbm_zero      concept sims   zero-shot through the concept bank: mean similarity of each class's concepts
  lp            image emb      linear probe, P = 0                    <- language-free control (same encoder)
  lp_text       image emb      linear probe, P = zero-shot text weights (language-initialised probe)
  cbm           concept sims   knowledge-prior concept bottleneck, P = class-concept association   <- main VLM head
  cbm_noprior   concept sims   same bottleneck, P = 0 (does the class-concept knowledge matter?)
  cbm_random    random sims    same bottleneck on random-word "concepts" with a random association (control)
"""
from __future__ import annotations

import numpy as np
import torch

from . import CLASSES
from .concepts import association, concept_list, random_concept_list

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_LAM = 1e-2
TRAINED = ("lp", "lp_text", "cbm", "cbm_noprior", "cbm_random")
ZERO_SHOT = ("zs_names", "zs_desc", "cbm_zero")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class PriorLogReg:
    def __init__(self, lam: float = DEFAULT_LAM, prior: np.ndarray | None = None, max_iter: int = 200):
        self.lam, self.prior, self.max_iter = lam, prior, max_iter

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int):
        X_t = torch.tensor(X, dtype=torch.float64); y_t = torch.tensor(y, dtype=torch.long)
        P = torch.zeros(n_classes, X.shape[1], dtype=torch.float64) if self.prior is None else \
            torch.tensor(self.prior, dtype=torch.float64)
        counts = np.bincount(y, minlength=n_classes).astype(float)
        w = torch.tensor(np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0))
        W = P.clone().requires_grad_(True)
        b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-5, tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(X_t @ W.T + b, y_t, weight=w) + self.lam * ((W - P) ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach().numpy(), b.detach().numpy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W.T + self.b)


def choose_lambda(X, y, groups, n_classes, prior=None, lambdas=LAMBDAS, seed: int = 42) -> float:
    """Inner group-aware stratified CV (up to 3 folds) on the training subset; default if a class has < 2 groups."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    per_class_groups = min(len(set(groups[y == k])) for k in np.unique(y))
    n_splits = min(3, per_class_groups)
    if n_splits < 2 or len(np.unique(y)) < n_classes:
        return DEFAULT_LAM
    best, best_lam = -1.0, DEFAULT_LAM
    splits = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    for lam in lambdas:
        pred = np.empty(len(y), int)
        for tr, va in splits:
            pred[va] = PriorLogReg(lam, prior).fit(X[tr], y[tr], n_classes).predict_proba(X[va]).argmax(1)
        f = f1_score(y, pred, average="macro")
        if f > best + 1e-9:
            best, best_lam = f, lam
    return best_lam


# ---------------------------------------------------------------- features and priors from one encoder's embeddings
class Space:
    """Everything the heads need from one encoder + one image view, for a chosen class list (default: all 6)."""

    def __init__(self, emb: dict, view: str = "crop", classes=None):
        self.classes = list(classes or CLASSES)
        self.cidx = [CLASSES.index(c) for c in self.classes]
        self.ids = {i: k for k, i in enumerate(emb["image_id"])}
        self.img = emb[f"img_{view}"].astype(np.float64)
        self.scale = float(emb["logit_scale"])
        self.T_names = self._class_means(emb["txt_names"], emb["owner_names"])
        self.T_desc = self._class_means(emb["txt_desc"], emb["owner_desc"])
        _, owner = concept_list(); _, rowner = random_concept_list()
        self.A = association(owner)[self.cidx]; self.A_rand = association(rowner)[self.cidx]
        # concept features: scaled cosine similarity, centred by the mean over ALL images (no labels involved)
        self.C = self.scale * self.img @ emb["txt_concepts"].T.astype(np.float64)
        self.C -= self.C.mean(0)
        self.R = self.scale * self.img @ emb["txt_random"].T.astype(np.float64)
        self.R -= self.R.mean(0)

    def _class_means(self, T, owner):
        M = np.stack([T[owner == CLASSES.index(c)].mean(0) for c in self.classes])
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    def rows(self, image_ids) -> np.ndarray:
        return np.array([self.ids[i] for i in image_ids])

    def features(self, head: str, r: np.ndarray) -> tuple:
        """(X, prior) for a head and row indices."""
        if head in ("lp", "lp_text", "zs_names", "zs_desc"):
            X = self.img[r]
            prior = {"lp": None, "lp_text": self.scale * self.T_desc, "zs_names": self.scale * self.T_names,
                     "zs_desc": self.scale * self.T_desc}[head]
        elif head in ("cbm", "cbm_noprior", "cbm_zero"):
            X, prior = self.C[r], (None if head == "cbm_noprior" else self.A)
        elif head == "cbm_random":
            X, prior = self.R[r], self.A_rand
        else:
            raise KeyError(head)
        return X, prior


def fit_predict(space: Space, head: str, train_ids, y_train, groups_train, predict_ids: dict, seed: int = 42) -> dict:
    """Fit one head on train ids (y indexed into space.classes) and return {name: probs over space.classes}.
    Zero-shot heads ignore the training data."""
    k = len(space.classes)
    out = {}
    if head in ZERO_SHOT:
        for name, ids in predict_ids.items():
            X, prior = space.features(head, space.rows(ids))
            out[name] = softmax(X @ prior.T)
        return out
    Xtr, prior = space.features(head, space.rows(train_ids))
    lam = choose_lambda(Xtr, np.asarray(y_train), np.asarray(groups_train), k, prior, seed=seed)
    model = PriorLogReg(lam, prior).fit(Xtr, np.asarray(y_train), k)
    for name, ids in predict_ids.items():
        out[name] = model.predict_proba(space.features(head, space.rows(ids))[0])
    out["_lambda"] = lam
    return out


def to_all_classes(p: np.ndarray, classes) -> np.ndarray:
    """Embed probabilities over a class subset into the full CLASSES order (absent classes get 0)."""
    full = np.zeros((len(p), len(CLASSES)))
    full[:, [CLASSES.index(c) for c in classes]] = p
    return full


def fuse(*prob_tables: np.ndarray) -> np.ndarray:
    """Pre-registered late fusion: equal-weight geometric mean of class probabilities (= mean log-probability)."""
    logp = np.mean([np.log(np.clip(p, 1e-8, 1)) for p in prob_tables], axis=0)
    return softmax(logp)


def select_pseudo_labels(p_pool: np.ndarray, keep_frac: float = 0.5) -> tuple:
    """Pre-registered pseudo-label rule: hard label = teacher argmax; keep the most confident `keep_frac` of the
    images predicted as each class. Returns (selected row indices, labels, confidences)."""
    lab, conf = p_pool.argmax(1), p_pool.max(1)
    keep = []
    for k in np.unique(lab):
        r = np.flatnonzero(lab == k)
        n = int(np.ceil(keep_frac * len(r)))
        keep.extend(r[np.argsort(-conf[r])[:n]])
    keep = np.array(sorted(keep), dtype=int)
    return keep, lab[keep], conf[keep]
