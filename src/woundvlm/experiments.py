"""Orchestration of every pre-registered run. Notebooks call these functions and nothing else.

Prediction files are long tables: one row per (run, image) with model, encoder, view, setting, fold, frac, k, draw,
eval_set, image_id, group, y, p_<class>. Nothing is ever overwritten: an existing output means "already done".
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import CLASSES, SHARED_AZH, data, heads
from .embed import ENCODERS, PRIMARY
from .metrics import PCOLS, prediction_frame

PL_FRACS = (0.05, 0.10, 0.25)
FEW_SHOT_K = (1, 2, 4, 8, 16)
DRAWS = 3                          # extra low-label draws for the primary encoder's heads (draw 0 = primary)
HEADS = heads.ZERO_SHOT + heads.TRAINED


# ---------------------------------------------------------------- 01 data (local, CPU)
def prepare(project: str, raw: str | None = None) -> pd.DataFrame:
    raw = raw or os.path.join(project, "data", "raw")
    df = data.prepare_images(data.build_index(raw), project)
    lab = data.labelled(df)
    df["fold"] = -1
    df.loc[lab.index, "fold"] = data.make_folds(lab)
    for c in ("path", "mask_path"):                    # keep paths relative to data/raw (no local absolute paths)
        df[c] = df[c].map(lambda p: os.path.relpath(p, raw).replace("\\", "/") if p else "")
    df.to_csv(os.path.join(project, "data", "processed", "index.csv"), index=False)
    return df


# ---------------------------------------------------------------- 02 embeddings (GPU)
def run_embeddings(project: str, encoders=tuple(ENCODERS), device: str = "cuda") -> None:
    from .embed import Encoder, embed_all
    idx = data.load_index(project)
    for name in encoders:
        out = os.path.join(project, "outputs", "emb", f"{name}.npz")
        if os.path.exists(out):
            print(f"{name}: exists, skipped"); continue
        print(f"{name}: {ENCODERS[name][1]}")
        try:
            enc = Encoder(name, device)
            embed_all(enc, idx, project, out)
            del enc
        except Exception as e:                       # secondary encoders are optional; the primary one is not
            if name == PRIMARY:
                raise
            print(f"{name}: FAILED ({type(e).__name__}: {e}) - continuing without it")
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 03 heads (CPU)
def _single_thread():
    """One torch thread per worker process: parallel workers otherwise fight over cores (10x slower)."""
    import torch
    torch.set_num_threads(1)


def _spaces(project: str) -> list:
    """(encoder, view) combinations available: every encoder on the crop view, the primary encoder also on full."""
    out = []
    for name in ENCODERS:
        if os.path.exists(os.path.join(project, "outputs", "emb", f"{name}.npz")):
            out.append((name, "crop"))
            if name == PRIMARY:
                out.append((name, "full"))
    return out


def _frame(meta, p, classes=None, **tags) -> pd.DataFrame:
    if classes:
        p = heads.to_all_classes(p, classes)
    return prediction_frame(meta, p, **tags)


def _lowlabel_task(project, enc, view, fold, frac, draw):
    _single_thread()
    from .embed import load_embeddings
    idx = data.load_index(project)
    lab, azh, pl = data.labelled(idx), data.azh_wounds(idx), data.pool(idx)
    space = heads.Space(load_embeddings(os.path.join(project, "outputs", "emb", f"{enc}.npz")), view)
    train, test = lab[lab.fold != fold], lab[lab.fold == fold]
    sub = train.loc[data.low_label_subset(train, frac, draw=draw)]
    tags = dict(encoder=enc, view=view, setting="lowlabel", fold=fold, frac=frac, k=-1, draw=draw)
    frames, pseudo, lams = [], None, []
    for h in HEADS:
        targets = {"test": test.image_id}
        if draw == 0:
            targets["azh"] = azh.image_id
        teacher = enc == PRIMARY and view == "crop" and h == "cbm" and draw == 0 and frac in PL_FRACS
        if teacher:
            targets["pool"] = pl.image_id
        res = heads.fit_predict(space, h, sub.image_id, sub.y, sub.group, targets)
        lams.append({**tags, "model": h, "lambda": res.get("_lambda", np.nan), "n_train": len(sub)})
        frames.append(_frame(test, res["test"], model=h, eval_set="wseg_test", **tags))
        if "azh" in res:
            frames.append(_frame(azh, res["azh"], model=h, eval_set="azh", **tags))
        if teacher:
            keep, lbl, conf = heads.select_pseudo_labels(res["pool"])
            pseudo = pd.DataFrame({"image_id": pl.image_id.to_numpy()[keep], "y": lbl, "confidence": conf,
                                   "fold": fold, "frac": frac})
    return frames, pseudo, lams


def _fewshot_task(project, fold, k, draw):
    _single_thread()
    from .embed import load_embeddings
    idx = data.load_index(project)
    lab = data.labelled(idx)
    space = heads.Space(load_embeddings(os.path.join(project, "outputs", "emb", f"{PRIMARY}.npz")), "crop")
    train, test = lab[lab.fold != fold], lab[lab.fold == fold]
    sub = train.loc[data.few_shot_subset(train, k, draw=draw)]
    tags = dict(encoder=PRIMARY, view="crop", setting="fewshot", fold=fold, frac=np.nan, k=k, draw=draw)
    frames = []
    for h in heads.TRAINED:
        res = heads.fit_predict(space, h, sub.image_id, sub.y, sub.group, {"test": test.image_id})
        frames.append(_frame(test, res["test"], model=h, eval_set="wseg_test", **tags))
    return frames


def _anchor_task(project, enc, view):
    """Published-comparison setting: AZH official train -> AZH official test, 4 wound classes (D, P, S, V)."""
    _single_thread()
    from .embed import load_embeddings
    idx = data.load_index(project)
    azh = data.azh_wounds(idx)
    tr, te = azh[azh.azh_split == "train"], azh[azh.azh_split == "test"]
    space = heads.Space(load_embeddings(os.path.join(project, "outputs", "emb", f"{enc}.npz")), view, SHARED_AZH)
    ytr = tr.label.map(SHARED_AZH.index).to_numpy()
    frames = []
    for h in HEADS:
        res = heads.fit_predict(space, h, tr.image_id, ytr, tr.group, {"test": te.image_id})
        frames.append(_frame(te, res["test"], SHARED_AZH, model=h, encoder=enc, view=view, setting="anchor_azh4",
                             fold=-1, frac=1.0, k=-1, draw=0, eval_set="azh_test"))
    return frames


def run_heads(project: str, n_jobs: int = -1) -> None:
    from joblib import Parallel, delayed
    out = os.path.join(project, "outputs", "heads")
    os.makedirs(os.path.join(out, "pseudo"), exist_ok=True)
    done = os.path.join(out, "preds_lowlabel.csv.gz")
    if os.path.exists(done):
        print("heads: outputs exist, skipped"); return
    t0 = time.time()
    tasks = []
    for enc, view in _spaces(project):
        primary = enc == PRIMARY and view == "crop"
        for fold in range(5):
            for frac in data.FRACTIONS:
                for draw in range(DRAWS if primary and frac < 1 else 1):
                    tasks.append((enc, view, fold, frac, draw))
    print(f"heads: {len(tasks)} low-label tasks on {_spaces(project)}")
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_lowlabel_task)(project, *t) for t in tasks)
    preds = pd.concat([f for r in res for f in r[0]], ignore_index=True)
    pseudo = [r[1] for r in res if r[1] is not None]
    pd.DataFrame([l for r in res for l in r[2]]).to_csv(os.path.join(out, "lambdas.csv"), index=False)
    for p in pseudo:
        f, fr = int(p.fold.iloc[0]), float(p.frac.iloc[0])
        p.to_csv(os.path.join(out, "pseudo", f"pseudo_f{f}_r{fr:.2f}.csv"), index=False)
    fs = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_fewshot_task)(project, f, k, d)
                                            for f in range(5) for k in FEW_SHOT_K for d in range(DRAWS))
    an = Parallel(n_jobs=n_jobs)(delayed(_anchor_task)(project, e, v) for e, v in _spaces(project) if v == "crop")
    pd.concat([f for r in fs for f in r], ignore_index=True).to_csv(os.path.join(out, "preds_fewshot.csv.gz"), index=False)
    pd.concat([f for r in an for f in r], ignore_index=True).to_csv(os.path.join(out, "preds_anchor.csv.gz"), index=False)
    preds.to_csv(done, index=False)                 # written last = completion marker
    print(f"heads: done in {time.time() - t0:.0f}s, {len(preds)} prediction rows, {len(pseudo)} pseudo-label sets")


# ---------------------------------------------------------------- 04/05 CNN (GPU)
def cnn_plan(kind: str) -> list:
    """Run specs: dict(tag, view, fold, frac, pseudo, anchor_seed)."""
    runs = []
    if kind == "main":
        runs += [dict(tag=f"cnn_crop_f{f}_r{r:.2f}", view="crop", fold=f, frac=r, pseudo=False)
                 for r in data.FRACTIONS for f in range(5)]
        runs += [dict(tag=f"cnn_full_f{f}_r1.00", view="full", fold=f, frac=1.0, pseudo=False) for f in range(5)]
        runs += [dict(tag=f"cnn_anchor_azh4_s{s}", view="crop", fold=-1, frac=1.0, pseudo=False, anchor_seed=s)
                 for s in (42, 43, 44)]
    elif kind == "pseudo":
        runs += [dict(tag=f"cnnpl_crop_f{f}_r{r:.2f}", view="crop", fold=f, frac=r, pseudo=True)
                 for r in PL_FRACS for f in range(5)]
    else:
        raise KeyError(kind)
    return runs


def run_cnn(project: str, kind: str = "main", device: str = "cuda", only: list | None = None, model_fn=None,
            **cfg) -> None:
    """model_fn(n_classes) -> fresh model (tests use a small random network); default ImageNet ResNet50."""
    import torch
    from . import cnn
    out = os.path.join(project, "outputs", "cnn")
    os.makedirs(out, exist_ok=True)
    idx = data.load_index(project)
    lab, azh, pl = data.labelled(idx), data.azh_wounds(idx), data.pool(idx)
    runs = [r for r in cnn_plan(kind) if (only is None or r["tag"] in only)
            and not os.path.exists(os.path.join(out, r["tag"] + ".json"))]
    print(f"cnn {kind}: {len(runs)} runs to do")
    if not runs:
        return
    cache = {}

    def imgs(df, view):
        cache.setdefault(view, {})
        missing = [i for i in df.image_id if i not in cache[view]]
        if missing:
            rows = idx.set_index("image_id").loc[missing, f"{view}_path"].tolist()
            x = cnn.load_images(project, rows)
            cache[view].update(dict(zip(missing, x)))
        return torch.stack([cache[view][i] for i in df.image_id])

    for r in runs:
        t0 = time.time()
        print(f"[{r['tag']}]", flush=True)
        if "anchor_seed" in r:
            tr, te = azh[azh.azh_split == "train"], azh[azh.azh_split == "test"]
            tr = tr.assign(y=tr.label.map(SHARED_AZH.index)); n_cls = len(SHARED_AZH)
            fit_i, val_i = data.inner_split(tr, seed=r["anchor_seed"])
            fit, val = tr.loc[fit_i], tr.loc[val_i]
            evals = {"azh_test": te}
            seed = r["anchor_seed"]
        else:
            train, test = lab[lab.fold != r["fold"]], lab[lab.fold == r["fold"]]
            sub = train.loc[data.low_label_subset(train, r["frac"])]
            fit_i, val_i = data.inner_split(sub)
            fit, val = sub.loc[fit_i], sub.loc[val_i]
            evals = {"wseg_test": test, "azh": azh}
            n_cls, seed = len(CLASSES), cfg.get("seed", 42)
        x_fit, y_fit = imgs(fit, r["view"]), fit.y.to_numpy()
        n_pseudo = 0
        if r["pseudo"]:
            ps = pd.read_csv(os.path.join(project, "outputs", "heads", "pseudo", f"pseudo_f{r['fold']}_r{r['frac']:.2f}.csv"))
            x_fit = torch.cat([x_fit, imgs(pl.set_index("image_id").loc[ps.image_id].reset_index(), "crop")])
            y_fit = np.concatenate([y_fit, ps.y.to_numpy()]); n_pseudo = len(ps)
        model, hist = cnn.train(x_fit, y_fit, imgs(val, r["view"]), val.y.to_numpy(), n_cls, device=device,
                                model=model_fn(n_cls) if model_fn else None, **{**cfg, "seed": seed})
        for name, df in evals.items():
            p = cnn.predict(model, imgs(df, r["view"]), size=cfg.get("img", cnn.CFG["img"]))
            if n_cls != len(CLASSES):
                p = heads.to_all_classes(p, SHARED_AZH)
            model_name = "cnn_pl" if r["pseudo"] else "cnn"
            prediction_frame(df, p, model=model_name, encoder="resnet50", view=r["view"],
                             setting="anchor_azh4" if "anchor_seed" in r else "lowlabel", fold=r["fold"],
                             frac=r["frac"], k=-1, draw=0, eval_set=name, seed=seed).to_csv(
                os.path.join(out, f"{r['tag']}_{name}.csv.gz"), index=False)
        with open(os.path.join(out, r["tag"] + ".json"), "w") as fh:     # written last = completion marker
            json.dump({**r, "n_fit": len(fit), "n_val": len(val), "n_pseudo": n_pseudo, "config": {**cnn.CFG, **cfg},
                       "history": hist, "seconds": round(time.time() - t0, 1)}, fh, indent=1)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- 06 generative VLM (GPU)
def run_genvlm(project: str, device: str = "cuda", model=None, proc=None, limit: int | None = None) -> pd.DataFrame:
    import torch
    from . import genvlm
    idx = data.load_index(project)
    df = pd.concat([data.labelled(idx), data.azh_wounds(idx)])
    if limit:
        df = df.groupby("source").head(limit)
    if model is None:
        model, proc = genvlm.load(device=device)
        try:          # fp16 sanity check on one image; Qwen2.5-VL can overflow in fp16 on GPUs without bf16
            r = df.iloc[0]
            from PIL import Image
            genvlm.mc_and_attributes(model, proc, Image.open(os.path.join(project, r.crop_path)).convert("RGB").resize((genvlm.IMG, genvlm.IMG)))
        except FloatingPointError:
            print("fp16 produced non-finite scores -> float32")
            del model; torch.cuda.empty_cache()
            model, proc = genvlm.load(device=device, dtype=torch.float32)
    out = os.path.join(project, "outputs", "genvlm")
    os.makedirs(out, exist_ok=True)
    return genvlm.run(model, proc, df, project, os.path.join(out, "annotations.csv"))
