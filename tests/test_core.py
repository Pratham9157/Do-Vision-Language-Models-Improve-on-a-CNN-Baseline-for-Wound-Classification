"""CPU tests for data handling, heads, metrics and the CNN loop. Run from the project root: pytest -q tests"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from woundvlm import CLASSES, SHARED_AZH, data, heads, metrics  # noqa: E402
from woundvlm.concepts import association, concept_list, mc_orders, random_concept_list  # noqa: E402

INDEX = os.path.join(ROOT, "data", "processed", "index.csv")
needs_index = pytest.mark.skipif(not os.path.exists(INDEX), reason="run notebook 01 first")


# ---------------------------------------------------------------- data
def test_context_box_square_inside_and_contains_wound():
    m = np.zeros((300, 400), bool); m[20:60, 350:390] = True                   # wound near the top-right corner
    l, t, r, b = data.context_box(m)
    assert r - l == b - t and 0 <= l and 0 <= t and r <= 400 and b <= 300
    assert l <= 350 and r >= 390 and t <= 20 and b >= 60
    assert data.context_box(np.zeros((50, 50), bool)) == (0, 0, 50, 50)


def _toy(n_per=(30, 20, 12, 8, 6, 4)):
    rows = []
    for k, n in enumerate(n_per):
        for i in range(n):
            rows.append({"image_id": f"{k}_{i}", "label": CLASSES[k], "y": k, "group": f"g{k}_{i // 2}"})
    return pd.DataFrame(rows)


def test_low_label_subsets_nested_stratified_groupwhole():
    df = _toy()
    prev = set()
    for f in data.FRACTIONS:
        s = set(data.low_label_subset(df, f))
        assert prev <= s
        sub = df.loc[sorted(s)]
        assert set(sub.label) == set(CLASSES)                                     # every class present
        assert all(set(df.index[df.group == g]) <= s for g in sub.group)           # whole groups only
        prev = s
    assert len(prev) == len(df)
    a, b = data.low_label_subset(df, 0.25, draw=0), data.low_label_subset(df, 0.25, draw=1)
    assert list(a) != list(b)
    fs = df.loc[data.few_shot_subset(df, 2)]
    assert (fs.label.value_counts() == 2).all() and fs.group.is_unique


@needs_index
def test_real_index_folds_and_status():
    idx = data.load_index(ROOT)
    lab = data.labelled(idx)
    assert (lab.groupby("group").fold.nunique() == 1).all()                         # duplicates never split
    assert set(lab.fold) == set(range(5)) and len(lab) > 1500
    pool = data.pool(idx)
    assert not set(pool.group) & set(lab.group)                                     # pool has no copies of labelled images
    azh = data.azh_wounds(idx)
    assert set(azh.label) == set(SHARED_AZH) and set(azh.azh_split) == {"train", "test"}
    assert not set(azh[azh.azh_split == "train"].group) & set(azh[azh.azh_split == "test"].group)
    for p in idx.crop_path.sample(20, random_state=0):
        assert os.path.exists(os.path.join(ROOT, p))
    tr = lab[lab.fold != 0]
    fit, val = data.inner_split(tr)
    assert not set(tr.loc[fit].group) & set(tr.loc[val].group)


# ---------------------------------------------------------------- concepts
def test_concept_banks():
    t, o = concept_list(); rt, ro = random_concept_list()
    assert len(t) == len(rt) == len(o) == len(ro) and len(set(t)) == len(t)
    assert sorted(np.bincount(o)) == sorted(np.bincount(ro))
    A = association(o)
    assert A.shape == (len(CLASSES), len(t)) and np.allclose(A.sum(1), 1)
    orders = mc_orders()
    assert all(sorted(o) == list(range(6)) for o in orders)
    assert all(sorted(col) == list(range(6)) for col in zip(*orders))                # each class once per position


# ---------------------------------------------------------------- heads
def _blobs(n=240, d=16, k=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    y = np.arange(n) % k
    return centers[y] + rng.normal(size=(n, d)), y


def test_prior_logreg_learns_and_respects_prior():
    X, y = _blobs()
    m = heads.PriorLogReg(1e-3).fit(X[:180], y[:180], 6)
    assert (m.predict_proba(X[180:]).argmax(1) == y[180:]).mean() > 0.9
    P = np.random.default_rng(1).normal(size=(6, 16))
    strong = heads.PriorLogReg(1e4, P).fit(X, y, 6)
    assert np.allclose(strong.W, P, atol=1e-2)                                       # huge lambda -> stays at prior
    lam = heads.choose_lambda(X, y, np.arange(len(y)) // 2, 6)
    assert lam in heads.LAMBDAS


def fake_embeddings(ids, d=32, seed=0):
    from woundvlm.concepts import class_prompts
    rng = np.random.default_rng(seed)
    t, _ = concept_list()
    pr = class_prompts()
    nrm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    owner = lambda key: np.concatenate([[k] * len(pr[key][c]) for k, c in enumerate(CLASSES)])
    on, od = owner("names"), owner("descriptions")
    return {"image_id": np.array(list(ids), dtype=str), "img_crop": nrm(rng.normal(size=(len(ids), d))),
            "img_full": nrm(rng.normal(size=(len(ids), d))), "txt_concepts": nrm(rng.normal(size=(len(t), d))),
            "txt_random": nrm(rng.normal(size=(len(t), d))), "txt_names": nrm(rng.normal(size=(len(on), d))),
            "owner_names": on, "txt_desc": nrm(rng.normal(size=(len(od), d))), "owner_desc": od,
            "logit_scale": np.array(100.0)}


def test_all_heads_fit_predict():
    df = _toy((12, 12, 12, 12, 12, 12))
    emb = fake_embeddings(df.image_id)
    space = heads.Space(emb, "crop")
    for h in heads.ZERO_SHOT + heads.TRAINED:
        out = heads.fit_predict(space, h, df.image_id, df.y, df.group, {"test": df.image_id[:10]})
        assert out["test"].shape == (10, 6) and np.allclose(out["test"].sum(1), 1)
    sub = heads.Space(emb, "crop", SHARED_AZH)
    y4 = df[df.label.isin(SHARED_AZH)]
    out = heads.fit_predict(sub, "cbm", y4.image_id, y4.label.map(SHARED_AZH.index), y4.group, {"t": y4.image_id})
    full = heads.to_all_classes(out["t"], SHARED_AZH)
    assert full.shape[1] == 6 and np.allclose(full[:, [CLASSES.index("trauma"), CLASSES.index("other")]], 0)


def test_fuse_and_pseudo_labels():
    a = np.array([[0.9, 0.1], [0.2, 0.8]]); b = np.array([[0.6, 0.4], [0.5, 0.5]])
    f = heads.fuse(a, b)
    assert np.allclose(f.sum(1), 1) and (f.argmax(1) == [0, 1]).all()
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.45, 0.55]])
    keep, lab, conf = heads.select_pseudo_labels(p)
    assert list(keep) == [0, 2] and list(lab) == [0, 1]


# ---------------------------------------------------------------- metrics
def _preds(y, p, groups=None):
    df = pd.DataFrame({"image_id": [f"i{i}" for i in range(len(y))], "group": groups or [f"g{i}" for i in range(len(y))],
                       "y": y})
    df[metrics.PCOLS] = p
    return df


def test_fast_scores_match_sklearn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    rng = np.random.default_rng(3)
    for k, n in ((6, 200), (4, 37), (6, 9)):                    # incl. classes absent from y / predictions
        y = rng.integers(0, k - 1, n); p = rng.dirichlet(np.ones(k), n); pred = p.argmax(1)
        s = metrics.scores(y, p)
        labs = np.arange(k)
        assert np.isclose(s["macro_f1"], f1_score(y, pred, labels=labs, average="macro", zero_division=0))
        assert np.isclose(s["weighted_f1"], f1_score(y, pred, labels=labs, average="weighted", zero_division=0))
        assert np.isclose(s["accuracy"], accuracy_score(y, pred))
        assert np.isclose(s["balanced_accuracy"], balanced_accuracy_score(y, pred))


def test_metrics_and_paired_difference():
    rng = np.random.default_rng(0)
    y = np.arange(120) % 6
    good = np.eye(6)[y] * 0.9 + 0.1 / 6
    bad = rng.dirichlet(np.ones(6), 120)
    s = metrics.summarize(_preds(y, good), n_boot=50)
    assert s["macro_f1"] == 1.0 and s["macro_f1_ci_lo"] == 1.0
    r = metrics.paired_difference(_preds(y, good), _preds(y, bad), n_boot=200)
    assert r["a_better"] and r["p_boot"] < 0.05
    same = metrics.paired_difference(_preds(y, good), _preds(y, good), n_boot=50)
    assert same["diff"] == 0 and not same["a_better"]
    # 4-class restriction (AZH): trauma/other columns ignored
    y4 = np.array([CLASSES.index(c) for c in SHARED_AZH] * 5)
    p4 = np.eye(6)[y4] * 0.5 + 0.5 * np.eye(6)[[CLASSES.index("trauma")] * 20]
    assert metrics.summarize(_preds(y4, p4), SHARED_AZH, n_boot=20)["accuracy"] == 1.0
    assert metrics.holm([0.01, 0.04, 0.03]) == [True, False, False]
    assert metrics.holm([0.001, 0.01, 0.02]) == [True, True, True]


# ---------------------------------------------------------------- CNN
@needs_index
def test_cnn_train_predict_tiny():
    import torch
    from woundvlm import cnn
    idx = data.labelled(data.load_index(ROOT)).groupby("label").head(3)
    x = cnn.load_images(ROOT, idx.crop_path.tolist(), size=64)
    assert x.dtype == torch.uint8 and x.shape == (len(idx), 3, 64, 64)
    model = cnn.build_model(6, "resnet18", pretrained=False)
    model, hist = cnn.train(x, idx.y.to_numpy(), x, idx.y.to_numpy(), 6, device="cpu", model=model,
                            img=48, epochs=2, min_steps=2, batch=8)
    assert len(hist) >= 1 and "val_macro_f1" in hist[0]
    p = cnn.predict(model, x, size=48)
    assert p.shape == (len(idx), 6) and np.allclose(p.sum(1), 1, atol=1e-4)
