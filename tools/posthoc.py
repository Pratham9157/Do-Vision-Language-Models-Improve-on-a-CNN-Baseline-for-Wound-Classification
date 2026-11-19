"""Post-hoc analyses (not among the five pre-registered tests) -> results/08_posthoc/.

    python tools/posthoc.py            # run from the project root after notebook 07

Tables:
  per_class_f1.csv                    per-class F1 of the main models at 10 % and 100 % labels
  text_shortcut_by_model.csv          accuracy on images with vs without visible text (ruler, label, handwriting)
  qwen_mc_confusion.csv               true class vs Qwen2.5-VL zero-shot answer
  qwen_attribute_means_by_class.csv   mean P(Yes) of each Qwen2.5-VL attribute question per class
  anchor_with_ci.csv                  AZH official split: accuracy / macro-F1 with 95 % CIs
  anchor_paired.csv                   AZH official split: SigLIP heads vs the 3-seed CNN ensemble (paired bootstrap)
If results/08_posthoc/ already exists, a new folder results/08_posthoc_run<N>/ is used (nothing is overwritten).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from woundvlm import CLASSES, SHARED_AZH, data  # noqa: E402
from woundvlm import analysis as A  # noqa: E402
from woundvlm import metrics as M  # noqa: E402


def out_dir(project: str, out: str | None) -> str:
    if out:
        os.makedirs(out, exist_ok=True)
        return out
    d, n = os.path.join(project, "results", "08_posthoc"), 1
    while os.path.exists(d):
        n += 1
        d = os.path.join(project, "results", f"08_posthoc_run{n}")
    os.makedirs(d)
    return d


def scores(g: pd.DataFrame) -> dict:
    return M.scores(M.labels_for(g), M.probs(g))


def main(project: str, out: str | None) -> str:
    d = out_dir(project, out)
    df = A.load_all(project)
    idx = data.labelled(data.load_index(project))
    ann = pd.read_csv(os.path.join(project, "outputs", "genvlm", "annotations.csv"))
    fusion = lambda f: A.fuse_frames(A.select(df, "cnn", f, view="crop"), A.select(df, "cbm", f, "siglip"))  # noqa: E731

    # per-class F1 at 10 % and 100 % labels
    low = pd.read_csv(os.path.join(project, "results", "07_lowlabel_summary.csv"))
    keep = (low.view == "crop") & low.frac.isin([0.1, 1.0]) & (
        ((low.encoder == "siglip") & low.model.isin(["lp", "lp_text", "cbm", "cbm_random"])) | low.model.isin(["cnn", "cnn_pl"]))
    cols = ["model", "frac", "macro_f1"] + [f"f1_{c}" for c in CLASSES]
    fused = []
    for f in (0.1, 1.0):
        g = fusion(f)
        fused.append({"model": "fusion cnn+cbm", "frac": f, "macro_f1": scores(g)["macro_f1"],
                      **M.per_class_f1(M.labels_for(g), M.probs(g))})
    pd.concat([low.loc[keep, cols], pd.DataFrame(fused)]).to_csv(os.path.join(d, "per_class_f1.csv"), index=False)

    # visible-text shortcut audit (text flags from the generative VLM, threshold 0.5)
    tx = idx.merge(ann[["image_id", "text_crop", "text_full"]], on="image_id")
    rows = []
    for name, sel in [("cnn crop", A.select(df, "cnn", 1.0, view="crop")), ("cnn full", A.select(df, "cnn", 1.0, view="full")),
                      ("lp siglip crop", A.select(df, "lp", 1.0, "siglip")), ("cbm siglip crop", A.select(df, "cbm", 1.0, "siglip")),
                      ("lp siglip full", A.select(df, "lp", 1.0, "siglip", view="full")), ("fusion cnn+cbm crop", fusion(1.0))]:
        g = sel.merge(tx[["image_id", "text_crop", "text_full"]], on="image_id")
        for col in ("text_crop", "text_full"):
            for visible in (False, True):
                h = g[(g[col] > 0.5) == visible]
                s = scores(h)
                rows.append(dict(model=name, text_flag=col, visible=visible, n=len(h), accuracy=s["accuracy"], macro_f1=s["macro_f1"]))
    pd.DataFrame(rows).to_csv(os.path.join(d, "text_shortcut_by_model.csv"), index=False)

    # generative VLM: zero-shot confusion and attribute means
    m = idx.merge(ann, on="image_id")
    pred = np.array(CLASSES)[m[[f"mc_p_{c}" for c in CLASSES]].to_numpy().argmax(1)]
    pd.crosstab(m.label, pd.Series(pred, name="qwen_pred", index=m.index)).to_csv(os.path.join(d, "qwen_mc_confusion.csv"))
    m.groupby("label")[[c for c in m if c.startswith("attr_")]].mean().T.round(2).to_csv(
        os.path.join(d, "qwen_attribute_means_by_class.csv"))

    # AZH official split (published anchor): CIs and paired comparisons with the CNN 3-seed ensemble
    s = df[df.setting == "anchor_azh4"]
    cnn = s[s.model == "cnn"]
    ens = cnn.groupby(["image_id", "group", "y"], as_index=False)[M.PCOLS].mean().sort_values("image_id")
    head = lambda name: s[(s.model == name) & (s.encoder == "siglip")].sort_values("image_id")  # noqa: E731
    rows = []
    for name, g in [("cnn 3-seed ensemble", ens), ("lp siglip", head("lp")), ("cbm siglip", head("cbm")),
                    ("cbm_zero siglip", head("cbm_zero"))] + [(f"cnn seed {int(k)}", g) for k, g in cnn.groupby("seed")]:
        r = M.summarize(g, SHARED_AZH)
        rows.append({"model": name, **{k: r[k] for k in r if k.startswith(("accuracy", "macro_f1"))}})
    pd.DataFrame(rows).to_csv(os.path.join(d, "anchor_with_ci.csv"), index=False)
    rows = []
    for name in ("lp", "cbm", "cbm_zero"):
        for metric in ("accuracy", "macro_f1"):
            r = M.paired_difference(head(name), ens, metric, SHARED_AZH)
            rows.append({"comparison": f"AZH anchor: {name} siglip vs cnn 3-seed ensemble", "metric": metric, **r})
    pd.DataFrame(rows).to_csv(os.path.join(d, "anchor_paired.csv"), index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=ROOT)
    ap.add_argument("--out", default=None, help="output folder (default: a new results/08_posthoc* folder)")
    a = ap.parse_args()
    print("wrote", main(a.project, a.out))
