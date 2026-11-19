"""Final analysis (CPU): pooled out-of-fold summaries, fusion, generative-VLM heads, the pre-registered primary tests
(with Holm correction), the published-anchor table, the visible-text audit and concept-weight explanations."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import CLASSES, SHARED_AZH, data, heads, metrics
from .concepts import ATTRIBUTES, concept_list
from .embed import PRIMARY
from .metrics import PCOLS

KEY = ["model", "encoder", "view", "setting", "frac", "k", "draw", "eval_set"]
PUBLISHED_AZH4 = [   # AZH official test split, 4 wound classes (D, P, S, V), accuracy (%)
    ("Anisuzzaman et al. 2022 (Sci Rep), best image-only (VGG16)", "image", 71.73),
    ("Anisuzzaman et al. 2022 (Sci Rep), best image + location (VGG16+LSTM)", "image+location", 79.35),
]


# ---------------------------------------------------------------- loading
def load_all(project: str) -> pd.DataFrame:
    o = os.path.join(project, "outputs")
    parts = [pd.read_csv(f) for f in (os.path.join(o, "heads", "preds_lowlabel.csv.gz"),
                                      os.path.join(o, "heads", "preds_fewshot.csv.gz"),
                                      os.path.join(o, "heads", "preds_anchor.csv.gz")) if os.path.exists(f)]
    parts += [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(o, "cnn", "*.csv.gz")))]
    ann = os.path.join(o, "genvlm", "annotations.csv")
    if os.path.exists(ann):
        parts += genvlm_preds(project, pd.read_csv(ann))
    df = pd.concat(parts, ignore_index=True)
    for c in ("frac", "k", "draw", "fold"):
        df[c] = df[c].astype(float)
    if "seed" not in df:
        df["seed"] = np.nan
    return df


def genvlm_preds(project: str, ann: pd.DataFrame) -> list:
    """Generative-VLM predictions: zero-shot multiple choice, and an attribute head (16 VLM answers -> logistic
    regression) trained per fold and label fraction exactly like the embedding heads."""
    idx = data.load_index(project)
    lab, azh = data.labelled(idx), data.azh_wounds(idx)
    a = ann.set_index("image_id")
    out = []
    mc = [f"mc_p_{c}" for c in CLASSES]
    tags = dict(model="genvlm_mc", encoder="qwen2.5-vl-3b", view="crop", setting="lowlabel", k=-1, draw=0, frac=0.0)
    for f in range(5):
        te = lab[lab.fold == f]
        out.append(metrics.prediction_frame(te, a.loc[te.image_id, mc].to_numpy(), fold=f, eval_set="wseg_test", **tags))
        out.append(metrics.prediction_frame(azh, a.loc[azh.image_id, mc].to_numpy(), fold=f, eval_set="azh", **tags))
    cols = [f"attr_{k}" for k, _ in ATTRIBUTES]
    X = np.log(np.clip(a[cols].to_numpy(float), 1e-4, 1 - 1e-4) / np.clip(1 - a[cols].to_numpy(float), 1e-4, 1))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pos = {i: r for r, i in enumerate(a.index)}
    for f in range(5):
        train, te = lab[lab.fold != f], lab[lab.fold == f]
        for frac in data.FRACTIONS:
            sub = train.loc[data.low_label_subset(train, frac)]
            Xs = X[[pos[i] for i in sub.image_id]]
            lam = heads.choose_lambda(Xs, sub.y.to_numpy(), sub.group.to_numpy(), len(CLASSES))
            m = heads.PriorLogReg(lam).fit(Xs, sub.y.to_numpy(), len(CLASSES))
            t = dict(model="genvlm_attr", encoder="qwen2.5-vl-3b", view="crop", setting="lowlabel", fold=f, frac=frac,
                     k=-1, draw=0)
            out.append(metrics.prediction_frame(te, m.predict_proba(X[[pos[i] for i in te.image_id]]),
                                                eval_set="wseg_test", **t))
            out.append(metrics.prediction_frame(azh, m.predict_proba(X[[pos[i] for i in azh.image_id]]),
                                                eval_set="azh", **t))
    return out


def select(df: pd.DataFrame, model: str, frac=None, encoder=None, view="crop", eval_set="wseg_test",
           setting="lowlabel", draw=0, k=None) -> pd.DataFrame:
    m = (df.model == model) & (df.view == view) & (df.eval_set == eval_set) & (df.setting == setting) & (df.draw == draw)
    if frac is not None:
        m &= np.isclose(df.frac, frac)
    if encoder is not None:
        m &= df.encoder == encoder
    if k is not None:
        m &= df.k == k
    return df[m]


def fold_ensemble(sel: pd.DataFrame) -> pd.DataFrame:
    """For external evaluation: average the 5 fold models' probabilities per image."""
    return sel.groupby(["image_id", "group", "y"], as_index=False)[PCOLS].mean()


def fuse_frames(a: pd.DataFrame, b: pd.DataFrame, on=("image_id", "group", "y", "fold")) -> pd.DataFrame:
    on = [c for c in on if c in a and c in b]
    m = a[on + PCOLS].merge(b[on + PCOLS], on=on, suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b)
    p = heads.fuse(m[[c + "_a" for c in PCOLS]].to_numpy(), m[[c + "_b" for c in PCOLS]].to_numpy())
    out = m[on].copy(); out[PCOLS] = p
    return out


# ---------------------------------------------------------------- summaries
def summary_lowlabel(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Pooled out-of-fold scores on WoundSeg for every model x encoder x view x frac (draw 0)."""
    rows = []
    s = df[(df.setting == "lowlabel") & (df.eval_set == "wseg_test") & (df.draw == 0)]
    for key, g in s.groupby(["model", "encoder", "view", "frac"]):
        if g.image_id.duplicated().any():
            raise ValueError(f"duplicate images for {key}")
        rows.append({**dict(zip(["model", "encoder", "view", "frac"], key)), **metrics.summarize(g, n_boot=n_boot)})
    return pd.DataFrame(rows)


def summary_draws(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of pooled macro-F1 over the low-label draws (primary encoder heads only)."""
    s = df[(df.setting == "lowlabel") & (df.eval_set == "wseg_test") & (df.encoder == PRIMARY) & (df.view == "crop")]
    rows = []
    for (m, fr, d), g in s.groupby(["model", "frac", "draw"]):
        rows.append({"model": m, "frac": fr, "draw": d,
                     "macro_f1": metrics.scores(metrics.labels_for(g), metrics.probs(g))["macro_f1"]})
    t = pd.DataFrame(rows)
    return t.groupby(["model", "frac"]).macro_f1.agg(["mean", "std", "count"]).reset_index()


def summary_fewshot(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df.setting == "fewshot"]
    rows = []
    for (m, k, d), g in s.groupby(["model", "k", "draw"]):
        rows.append({"model": m, "k": k, "draw": d,
                     "macro_f1": metrics.scores(metrics.labels_for(g), metrics.probs(g))["macro_f1"]})
    return pd.DataFrame(rows).groupby(["model", "k"]).macro_f1.agg(["mean", "std"]).reset_index()


def summary_ood(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Trained on WoundSeg (fold ensemble), tested on all AZH wound images, scored over the 4 shared classes."""
    rows = []
    s = df[(df.setting == "lowlabel") & (df.eval_set == "azh") & (df.draw == 0)]
    for key, g in s.groupby(["model", "encoder", "view", "frac"]):
        e = fold_ensemble(g)
        rows.append({**dict(zip(["model", "encoder", "view", "frac"], key)),
                     **metrics.summarize(e, SHARED_AZH, n_boot=n_boot)})
    return pd.DataFrame(rows)


def summary_anchor(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df.setting == "anchor_azh4"]
    rows = []
    for key, g in s.groupby(["model", "encoder", "view", "seed"], dropna=False):
        rows.append({**dict(zip(["model", "encoder", "view", "seed"], key)), **metrics.summarize(g, SHARED_AZH)})
    t = pd.DataFrame(rows)
    pub = pd.DataFrame([{"model": n, "encoder": "published", "input": i, "accuracy": a / 100} for n, i, a in PUBLISHED_AZH4])
    return pd.concat([t, pub], ignore_index=True)


def trivial_baselines(project: str, seed: int = 42) -> pd.DataFrame:
    """Floors: always the majority training class; random guessing with training class frequencies (mean of 200)."""
    from sklearn.metrics import f1_score
    lab, azh = data.labelled(data.load_index(project)), data.azh_wounds(data.load_index(project))
    rng = np.random.default_rng(seed)
    prior = np.bincount(lab.y, minlength=len(CLASSES)) / len(lab)
    rows = []
    for name, df, classes in (("wseg 6-class", lab, CLASSES), ("azh 4-class", azh, SHARED_AZH)):
        ids = [CLASSES.index(c) for c in classes]
        y = df.y.to_numpy(); pr = prior[ids] / prior[ids].sum()
        maj = np.full(len(y), ids[int(np.argmax(pr))])
        rnd = [f1_score(y, rng.choice(ids, len(y), p=pr), labels=ids, average="macro") for _ in range(200)]
        rows += [{"set": name, "model": "majority class", "macro_f1": f1_score(y, maj, labels=ids, average="macro"),
                  "accuracy": (maj == y).mean()},
                 {"set": name, "model": "stratified random", "macro_f1": float(np.mean(rnd)), "accuracy": np.nan}]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pre-registered primary tests
def primary_tests(df: pd.DataFrame) -> pd.DataFrame:
    P = dict(encoder=PRIMARY, view="crop")
    cnn = lambda fr, es="wseg_test": select(df, "cnn", fr, view="crop", eval_set=es)
    tests = [
        ("H1a", "low labels (10%): concept bottleneck vs language-free linear probe (same encoder)",
         select(df, "cbm", 0.10, **P), select(df, "lp", 0.10, **P), None),
        ("H1b", "low labels (10%): concept bottleneck vs CNN",
         select(df, "cbm", 0.10, **P), cnn(0.10), None),
        ("H2", "external AZH test (trained on 100% WoundSeg): concept bottleneck vs CNN",
         fold_ensemble(select(df, "cbm", 1.0, eval_set="azh", **P)), fold_ensemble(cnn(1.0, "azh")), SHARED_AZH),
        ("H3", "100% labels: CNN + concept bottleneck fusion vs CNN",
         fuse_frames(cnn(1.0), select(df, "cbm", 1.0, **P)), cnn(1.0), None),
        ("H4", "low labels (10%): CNN trained with VLM pseudo-labels vs CNN",
         select(df, "cnn_pl", 0.10, view="crop"), cnn(0.10), None),
    ]
    rows = []
    for hid, desc, a, b, classes in tests:
        if len(a) == 0 or len(b) == 0:
            rows.append({"id": hid, "test": desc, "status": "missing outputs"}); continue
        r = metrics.paired_difference(a, b, classes=classes)
        sa = metrics.scores(metrics.labels_for(a, classes), metrics.probs(a, classes))["macro_f1"]
        sb = metrics.scores(metrics.labels_for(b, classes), metrics.probs(b, classes))["macro_f1"]
        rows.append({"id": hid, "test": desc, "macro_f1_a": sa, "macro_f1_b": sb, **r})
    t = pd.DataFrame(rows)
    ok = t.p_boot.notna() if "p_boot" in t else pd.Series(False, index=t.index)
    t["holm_reject"] = False
    if ok.any():
        t.loc[ok, "holm_reject"] = metrics.holm(t.loc[ok, "p_boot"].tolist())
    t["verdict"] = np.where(t.holm_reject & (t.get("diff", 0) > 0), "A better (confirmed)",
                            np.where(t.holm_reject & (t.get("diff", 0) < 0), "B better", "no significant difference"))
    return t


def secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-listed secondary comparisons (no multiplicity correction; reported as exploratory evidence)."""
    P = dict(encoder=PRIMARY, view="crop")
    rows = []

    def add(name, a, b, classes=None):
        if len(a) and len(b) and len(a) == len(b):
            rows.append({"comparison": name, **metrics.paired_difference(a, b, classes=classes)})
    for fr in data.FRACTIONS:
        add(f"cbm vs cbm_random @ {fr}", select(df, "cbm", fr, **P), select(df, "cbm_random", fr, **P))
        add(f"cbm vs cbm_noprior @ {fr}", select(df, "cbm", fr, **P), select(df, "cbm_noprior", fr, **P))
        add(f"lp_text vs lp @ {fr}", select(df, "lp_text", fr, **P), select(df, "lp", fr, **P))
        add(f"cbm vs cnn @ {fr}", select(df, "cbm", fr, **P), select(df, "cnn", fr, view="crop"))
        add(f"lp_text vs cnn @ {fr}", select(df, "lp_text", fr, **P), select(df, "cnn", fr, view="crop"))
        cnn = select(df, "cnn", fr, view="crop")
        for other in ("cbm", "lp_text"):
            o = select(df, other, fr, **P)
            if len(o) and len(cnn):
                add(f"fusion cnn+{other} vs cnn @ {fr}", fuse_frames(cnn, o), cnn)
        g = select(df, "genvlm_attr", fr, encoder="qwen2.5-vl-3b")
        if len(g) and len(cnn):
            add(f"fusion cnn+genvlm_attr vs cnn @ {fr}", fuse_frames(cnn, g), cnn)
        if fr in (0.05, 0.25):
            add(f"cnn_pl vs cnn @ {fr}", select(df, "cnn_pl", fr, view="crop"), cnn)
    a, b = fold_ensemble(select(df, "lp_text", 1.0, eval_set="azh", **P)), fold_ensemble(select(df, "cnn", 1.0, view="crop", eval_set="azh"))
    add("AZH: lp_text vs cnn", a, b, SHARED_AZH)
    c = fold_ensemble(select(df, "cbm", 1.0, eval_set="azh", **P))
    if len(c) and len(b):
        add("AZH: fusion cnn+cbm vs cnn", fuse_frames(b, c, on=("image_id", "group", "y")), b, SHARED_AZH)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- audits and explanations
def text_audit(project: str, df: pd.DataFrame) -> pd.DataFrame:
    """How often the generative VLM sees a ruler/label/handwriting in the crop vs the full photo, by class, and CNN
    accuracy on images with vs without visible text in the crop (a shortcut check)."""
    ann = pd.read_csv(os.path.join(project, "outputs", "genvlm", "annotations.csv"))
    idx = data.labelled(data.load_index(project)).merge(ann, on="image_id")
    t = idx.groupby("label").agg(n=("image_id", "size"), text_in_crop=("text_crop", lambda s: (s > 0.5).mean()),
                                 text_in_full=("text_full", lambda s: (s > 0.5).mean())).reset_index()
    cnn = select(df, "cnn", 1.0, view="crop").merge(idx[["image_id", "text_crop"]], on="image_id")
    for flag, g in cnn.groupby(cnn.text_crop > 0.5):
        sc = metrics.scores(metrics.labels_for(g), metrics.probs(g))
        t = pd.concat([t, pd.DataFrame([{"label": f"cnn@100% on crops {'with' if flag else 'without'} visible text",
                                         "n": len(g), "text_in_crop": sc["accuracy"]}])], ignore_index=True)
    return t


def concept_weights(project: str, top: int = 5) -> pd.DataFrame:
    """Fit the concept bottleneck on all labelled WoundSeg images (primary encoder, crop) and list, per class, the
    concepts with the largest positive weights (how the model explains each class)."""
    from .embed import load_embeddings
    lab = data.labelled(data.load_index(project))
    space = heads.Space(load_embeddings(os.path.join(project, "outputs", "emb", f"{PRIMARY}.npz")), "crop")
    X, prior = space.features("cbm", space.rows(lab.image_id))
    lam = heads.choose_lambda(X, lab.y.to_numpy(), lab.group.to_numpy(), len(CLASSES), prior)
    W = heads.PriorLogReg(lam, prior).fit(X, lab.y.to_numpy(), len(CLASSES)).W
    texts, owner = concept_list()
    rows = []
    for k, c in enumerate(CLASSES):
        for j in np.argsort(-W[k])[:top]:
            rows.append({"class": c, "concept": texts[j], "concept_written_for": CLASSES[owner[j]],
                         "weight": W[k, j], "lambda": lam})
    return pd.DataFrame(rows)
