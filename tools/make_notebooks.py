"""Generate the thin notebooks in notebooks/ (they only call src/woundvlm). Re-run after editing this file:
    python tools/make_notebooks.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIP = "transformers==4.51.3 open_clip_torch==2.32.0 sentencepiece"

SETUP = """import glob, os, sys
if os.path.isdir("/kaggle/input"):
    hits = glob.glob("/kaggle/input/**/src/woundvlm/__init__.py", recursive=True)
    assert hits, "attach the project dataset (lowerlimbwoundvlm) as an input"
    SRC = hits[0].rsplit("/woundvlm/", 1)[0]
else:
    SRC = os.path.abspath("../src")
sys.path.insert(0, SRC)
from woundvlm import env
E = env.setup()
P = E["project"]
print(E)"""

GPU_CHECK = """import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
assert DEVICE == "cuda" or not E["in_kaggle"], "select a GPU accelerator (T4) for this notebook" """


STAGES = """import gc, json, subprocess, time, traceback
STATUS = os.path.join(E["results"], "00_stage_status.json")
status = json.load(open(STATUS)) if os.path.exists(STATUS) else {}

def stage(name, fn):
    t = time.time()
    print(f"===== {name} =====", flush=True)
    try:
        fn(); status[name] = {"ok": True}
    except Exception as e:
        traceback.print_exc(); status[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    status[name]["minutes"] = round((time.time() - t) / 60, 1)
    json.dump(status, open(STATUS, "w"), indent=1)
    gc.collect(); torch.cuda.empty_cache()
    print(name, status[name], flush=True)

from woundvlm import experiments, report"""

HEADS_BG = """# CPU heads grid in a background process while the GPU trains (needs the SigLIP embeddings)
heads_log = open(os.path.join(E["results"], "03_heads.log"), "w")
code = f"import sys; sys.path.insert(0, {SRC!r}); from woundvlm import experiments; experiments.run_heads({P!r}, n_jobs=2)"
heads_proc = subprocess.Popen([sys.executable, "-c", code], stdout=heads_log, stderr=subprocess.STDOUT)
t_heads = time.time()
print("heads started, pid", heads_proc.pid)"""

HEADS_WAIT = """rc = heads_proc.wait(); heads_log.close()
status["heads"] = {"ok": rc == 0, "returncode": rc, "minutes": round((time.time() - t_heads) / 60, 1)}
json.dump(status, open(STATUS, "w"), indent=1)
print(status["heads"]); print(open(os.path.join(E["results"], "03_heads.log")).read()[-3000:])"""

NOTEBOOKS = {
    "00_all.ipynb": [
        ("md", "# 00 Everything in one GPU session\nEmbeddings -> (heads on CPU in the background) -> generative VLM -> "
               "CNN baselines -> CNN + VLM pseudo-labels -> final comparisons. Each stage is resumable and records its "
               "status in results/00_stage_status.json; a failing stage does not stop the others."),
        ("code", f"!pip -q install {PIP}"),
        ("code", SETUP), ("code", GPU_CHECK), ("code", STAGES),
        ("code", 'stage("embeddings", lambda: experiments.run_embeddings(P, device=DEVICE))'),
        ("code", HEADS_BG),
        ("code", 'stage("generative_vlm", lambda: experiments.run_genvlm(P, device=DEVICE))'),
        ("code", 'stage("cnn_main", lambda: experiments.run_cnn(P, "main", device=DEVICE))'),
        ("code", HEADS_WAIT),
        ("code", 'stage("cnn_pseudo", lambda: experiments.run_cnn(P, "pseudo", device=DEVICE))'),
        ("code", 'stage("report", lambda: print(report.write_all(P)))'),
        ("code", 'print(json.dumps(status, indent=1))'),
    ],

    "01_data.ipynb": [
        ("md", "# 01 Data: index, duplicate groups, folds, wound crops (local, CPU)\n"
               "Builds `data/processed/index.csv` and the crop/full images from `data/raw/`."),
        ("code", SETUP.replace("E = env.setup()", "E = {'project': os.path.abspath('..'), 'in_kaggle': False}")),
        ("code", "from woundvlm import experiments, data\n"
                 "df = experiments.prepare(P)\n"
                 "print(df.groupby(['source', 'status']).size())"),
        ("code", "import pandas as pd\n"
                 "lab, azh = data.labelled(df), data.azh_wounds(df)\n"
                 "audit = pd.concat([\n"
                 "    df.groupby(['source', 'status', 'label']).size().rename('n').reset_index().assign(table='status'),\n"
                 "    lab.groupby(['fold', 'label']).size().rename('n').reset_index().assign(table='folds'),\n"
                 "    azh.groupby(['azh_split', 'label']).size().rename('n').reset_index().assign(table='azh')])\n"
                 "os.makedirs(os.path.join(P, 'results'), exist_ok=True)\n"
                 "audit.to_csv(os.path.join(P, 'results', '01_data_audit.csv'), index=False)\n"
                 "assert (lab.groupby('group').fold.nunique() == 1).all()\n"
                 "lab.groupby(['fold', 'label']).size().unstack()"),
    ],
    "02_embeddings.ipynb": [
        ("md", "# 02 Contrastive VLM embeddings (GPU)\nSigLIP so400m (primary), CLIP ViT-L/14, BiomedCLIP: image embeddings "
               "of both views + all text banks -> `outputs/emb/<encoder>.npz`."),
        ("code", f"!pip -q install {PIP}"),
        ("code", SETUP), ("code", GPU_CHECK),
        ("code", "from woundvlm import experiments\nexperiments.run_embeddings(P, device=DEVICE)"),
        ("code", "import numpy as np\nfor f in sorted(glob.glob(os.path.join(P, 'outputs', 'emb', '*.npz'))):\n"
                 "    z = np.load(f); print(os.path.basename(f), {k: z[k].shape for k in z.files})"),
    ],
    "03_vlm_heads.ipynb": [
        ("md", "# 03 VLM heads (CPU)\nZero-shot, linear probe, knowledge-prior concept bottleneck and controls over all folds, "
               "label fractions, few-shot k and the AZH anchor; VLM pseudo-labels for notebook 05."),
        ("code", SETUP),
        ("code", "from woundvlm import experiments\nexperiments.run_heads(P, n_jobs=-1)"),
        ("code", "from woundvlm import analysis\nimport pandas as pd\n"
                 "df = analysis.load_all(P)\n"
                 "s = analysis.summary_lowlabel(df, n_boot=200)\n"
                 "s.to_csv(os.path.join(E['results'], '03_heads_lowlabel_quick.csv'), index=False)\n"
                 "s[s.encoder == 'siglip'].pivot_table(index='model', columns='frac', values='macro_f1').round(3)"),
    ],
    "04_cnn.ipynb": [
        ("md", "# 04 CNN baseline (GPU)\nResNet50 on 5 folds x 5 label fractions (crop view), full-photo view at 100%, "
               "and the AZH published-anchor runs (3 seeds). Resumable: finished runs are skipped."),
        ("code", SETUP), ("code", GPU_CHECK),
        ("code", "from woundvlm import experiments\nexperiments.run_cnn(P, 'main', device=DEVICE)"),
        ("code", "import json\nrows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(P, 'outputs', 'cnn', '*.json')))]\n"
                 "import pandas as pd\n"
                 "t = pd.DataFrame([{k: r[k] for k in ('tag', 'n_fit', 'n_val', 'seconds')} | "
                 "{'best_val_f1': max(h['val_macro_f1'] for h in r['history']), 'epochs': len(r['history'])} for r in rows])\n"
                 "t.to_csv(os.path.join(E['results'], '04_cnn_runs.csv'), index=False); t"),
    ],
    "05_cnn_pseudo.ipynb": [
        ("md", "# 05 CNN + VLM pseudo-labels (GPU)\nNeeds notebook 03 (pseudo-label files). 5 folds x {5, 10, 25}% labels."),
        ("code", SETUP), ("code", GPU_CHECK),
        ("code", "assert glob.glob(os.path.join(P, 'outputs', 'heads', 'pseudo', '*.csv')), 'attach notebook 03 output'\n"
                 "from woundvlm import experiments\nexperiments.run_cnn(P, 'pseudo', device=DEVICE)"),
        ("code", "import json, pandas as pd\nrows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(P, 'outputs', 'cnn', 'cnnpl_*.json')))]\n"
                 "t = pd.DataFrame([{k: r[k] for k in ('tag', 'n_fit', 'n_pseudo', 'seconds')} for r in rows])\n"
                 "t.to_csv(os.path.join(E['results'], '05_cnn_pseudo_runs.csv'), index=False); t"),
    ],
    "06_generative_vlm.ipynb": [
        ("md", "# 06 Generative VLM (GPU)\nQwen2.5-VL-3B: position-debiased zero-shot multiple choice, 16 yes/no visual "
               "attributes and the visible-text audit, all from answer-token logits. Resumable."),
        ("code", f"!pip -q install {PIP}"),
        ("code", SETUP), ("code", GPU_CHECK),
        ("code", "from woundvlm import experiments\nann = experiments.run_genvlm(P, device=DEVICE)\nann.describe().T.head(12)"),
        ("code", "import pandas as pd\n"
                 "lt = ann[[c for c in ann if c.startswith('letter_')]].mean()\n"
                 "lt.to_frame('mean_prob').to_csv(os.path.join(E['results'], '06_letter_bias.csv')); lt"),
    ],
    "07_comparisons.ipynb": [
        ("md", "# 07 Comparisons (CPU)\nAll pre-registered tests, secondary comparisons, external test, "
               "published anchor, text audit, concept explanations, learning curves."),
        ("code", SETUP),
        ("code", "from woundvlm import report\nsummary = report.write_all(P)\nprint(summary)"),
    ],
}


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main():
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": [cell(k, s) for k, s in cells], "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}}}
        with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1)
    print("wrote", ", ".join(NOTEBOOKS))


if __name__ == "__main__":
    main()
