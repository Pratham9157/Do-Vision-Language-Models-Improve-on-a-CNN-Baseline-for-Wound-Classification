"""Notebook 07: write every final table, the learning-curve figures and a markdown summary into results/.
Never overwrites: if results/07_summary.md exists, outputs go to a new numbered folder results/07_run<N>/."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import analysis

LABELS = {"cnn": "CNN (ResNet50)", "cnn_pl": "CNN + VLM pseudo-labels", "lp": "Linear probe (no language)",
          "lp_text": "Language-initialised probe", "cbm": "Concept bottleneck (knowledge prior)",
          "cbm_noprior": "Concept bottleneck (no prior)", "cbm_random": "Random-word bottleneck (control)",
          "zs_names": "Zero-shot (class names)", "zs_desc": "Zero-shot (descriptions)", "cbm_zero": "Zero-shot (concepts)",
          "genvlm_mc": "Qwen2.5-VL zero-shot", "genvlm_attr": "Qwen2.5-VL attributes head"}


def _out_dir(results: str) -> str:
    os.makedirs(results, exist_ok=True)
    if not os.path.exists(os.path.join(results, "07_summary.md")):
        return results
    n = 2
    while os.path.exists(os.path.join(results, f"07_run{n}")):
        n += 1
    d = os.path.join(results, f"07_run{n}")
    os.makedirs(d)
    return d


def learning_curve(low: pd.DataFrame, path: str, encoder: str = "siglip") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    show = ["cnn", "cnn_pl", "lp", "lp_text", "cbm", "cbm_random", "genvlm_attr"]
    for m in show:
        enc = None if m.startswith(("cnn", "genvlm")) else encoder
        s = low[(low.model == m) & (low.view == "crop") & ((low.encoder == enc) if enc else True)].sort_values("frac")
        if len(s):
            ax.plot(s.frac * 100, s.macro_f1, marker="o", label=LABELS.get(m, m),
                    linestyle="--" if m in ("cbm_random", "lp") else "-")
            ax.fill_between(s.frac * 100, s.macro_f1_ci_lo, s.macro_f1_ci_hi, alpha=0.12)
    for m in ("zs_desc", "genvlm_mc"):
        s = low[(low.model == m) & (low.view == "crop")]
        if len(s):
            ax.axhline(s.macro_f1.iloc[0], linestyle=":", linewidth=1, color="grey")
            ax.text(100, s.macro_f1.iloc[0], " " + LABELS[m], va="center", fontsize=7, color="grey")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("labelled training images used (per fold)"); ax.set_ylabel("macro-F1 (pooled out-of-fold)")
    ax.set_title("WoundSeg 6-class wound type: label-efficiency"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _md(df: pd.DataFrame, cols, digits: int = 3) -> str:
    t = df[cols].copy()
    for c in t.select_dtypes("number"):
        t[c] = t[c].round(digits)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False))


def write_all(project: str) -> str:
    res = _out_dir(os.path.join(project, "results"))
    df = analysis.load_all(project)
    low = analysis.summary_lowlabel(df); low.to_csv(os.path.join(res, "07_lowlabel_summary.csv"), index=False)
    draws = analysis.summary_draws(df); draws.to_csv(os.path.join(res, "07_lowlabel_draws.csv"), index=False)
    few = analysis.summary_fewshot(df); few.to_csv(os.path.join(res, "07_fewshot.csv"), index=False)
    ood = analysis.summary_ood(df); ood.to_csv(os.path.join(res, "07_external_azh.csv"), index=False)
    anchor = analysis.summary_anchor(df); anchor.to_csv(os.path.join(res, "07_published_anchor_azh.csv"), index=False)
    prim = analysis.primary_tests(df); prim.to_csv(os.path.join(res, "07_primary_tests.csv"), index=False)
    sec = analysis.secondary_comparisons(df); sec.to_csv(os.path.join(res, "07_secondary_comparisons.csv"), index=False)
    parts = []
    try:
        audit = analysis.text_audit(project, df); audit.to_csv(os.path.join(res, "07_text_audit.csv"), index=False)
        parts.append("## Visible-text audit (generative VLM)\n" + _md(audit, list(audit.columns)))
    except FileNotFoundError:
        pass
    triv = analysis.trivial_baselines(project); triv.to_csv(os.path.join(res, "07_trivial_baselines.csv"), index=False)
    parts.append("## Trivial floors\n" + _md(triv, ["set", "model", "macro_f1", "accuracy"]))
    cw = analysis.concept_weights(project); cw.to_csv(os.path.join(res, "07_concept_weights.csv"), index=False)
    learning_curve(low, os.path.join(res, "07_learning_curve.png"))

    key = low[(low.view == "crop") & low.model.isin(LABELS) &
              ((low.encoder == "siglip") | low.model.str.startswith(("cnn", "genvlm")))]
    piv = key.pivot_table(index="model", columns="frac", values="macro_f1").round(3)
    piv.index = [LABELS.get(i, i) for i in piv.index]
    ood_key = ood[(ood.view == "crop") & np.isclose(ood.frac, 1.0) &
                  ((ood.encoder == "siglip") | ood.model.str.startswith(("cnn", "genvlm")))]
    lines = ["# Results summary (generated by notebook 07)", "",
             "## Pre-registered primary tests (Holm-corrected)",
             _md(prim, [c for c in ["id", "test", "macro_f1_a", "macro_f1_b", "diff", "ci_lo", "ci_hi", "p_boot", "verdict"]
                        if c in prim]), "",
             "## WoundSeg macro-F1 by label fraction (SigLIP heads, crop view)",
             _md(piv.rename(columns=lambda c: f"{c:.0%}").rename_axis("model").reset_index(),
                 ["model"] + [f"{c:.0%}" for c in piv.columns]), "",
             "## External test: AZH (trained on 100% WoundSeg, 4 shared classes)",
             _md(ood_key, ["model", "encoder", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi", "accuracy"]), "",
             "## Published anchor: AZH official split, 4 classes",
             _md(anchor, [c for c in ["model", "encoder", "seed", "accuracy", "macro_f1"] if c in anchor]), "",
             "## Concept bottleneck: top concepts per class",
             _md(cw.groupby("class").head(3), ["class", "concept", "concept_written_for", "weight"]), ""] + parts
    text = "\n".join(lines)
    with open(os.path.join(res, "07_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
