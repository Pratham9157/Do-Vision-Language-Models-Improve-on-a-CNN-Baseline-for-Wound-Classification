"""Run the notebooks on Kaggle from the command line (Kaggle CLI >= 2.0, token in ~/.kaggle/access_token).

    python tools/kaggle_pipeline.py dataset            # upload src/, notebooks/, data/processed/ (new version if it exists)
    python tools/kaggle_pipeline.py push 02            # push + run one notebook
    python tools/kaggle_pipeline.py status             # status of every notebook
    python tools/kaggle_pipeline.py fetch 02           # copy that notebook's outputs/ and results/ into the project
    python tools/kaggle_pipeline.py run 02 03 04       # push each in order, wait, fetch, stop at the first failure

Everything is private. Fetch never overwrites a local file (outputs are append-only).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build", "kaggle")
DATASET_SLUG = "lowerlimbwoundvlm"
WORK = "LowerLimbWoundVLM"
DATASET_FILES = ["src", "notebooks", "data/processed", "README.md"]
SKIP = ("__pycache__", ".pyc", ".pt", ".pth")
NOTEBOOKS = {   # id: (file, kernel slug, GPU?, earlier notebooks whose outputs it needs)
    "00": ("00_all.ipynb", "llw-00-all", True, []),
    "02": ("02_embeddings.ipynb", "llw-02-embeddings", True, []),
    "03": ("03_vlm_heads.ipynb", "llw-03-vlm-heads", False, ["02"]),
    "04": ("04_cnn.ipynb", "llw-04-cnn", True, []),
    "05": ("05_cnn_pseudo.ipynb", "llw-05-cnn-pseudo", True, ["03"]),
    "06": ("06_generative_vlm.ipynb", "llw-06-generative-vlm", True, []),
    "07": ("07_comparisons.ipynb", "llw-07-comparisons", False, ["03", "04", "05", "06"]),
}
GPU = "NvidiaTeslaT4"


def kaggle(*args: str, check: bool = True, capture: bool = False) -> str:
    exe = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
    r = subprocess.run([exe if os.path.exists(exe) else "kaggle", *args], text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{(r.stdout or '') + (r.stderr or '')}")
    return (r.stdout or "") + (r.stderr or "") if capture else ""


def username() -> str:
    for line in kaggle("config", "view", capture=True).splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no Kaggle username - put your API token in ~/.kaggle/access_token")


def dataset(message: str = "update") -> None:
    user = username()
    d = os.path.join(BUILD, "dataset")
    shutil.rmtree(d, ignore_errors=True)
    n = 0
    for item in DATASET_FILES:
        src = os.path.join(ROOT, item)
        files = [src] if os.path.isfile(src) else [os.path.join(p, f) for p, _, fs in os.walk(src) for f in fs]
        for f in files:
            if any(s in f for s in SKIP):
                continue
            t = os.path.join(d, WORK, os.path.relpath(f, ROOT))
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(f, t); n += 1
    json.dump({"title": "LowerLimbWoundVLM project", "id": f"{user}/{DATASET_SLUG}",
               "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, open(os.path.join(d, "dataset-metadata.json"), "w"), indent=1)
    print(f"staged {n} files")
    exists = DATASET_SLUG in kaggle("datasets", "list", "--mine", "-s", DATASET_SLUG, capture=True)
    kaggle("datasets", "version" if exists else "create", "-p", d, "--dir-mode", "zip", *(["-m", message] if exists else []))
    print(f"dataset {user}/{DATASET_SLUG} {'new version' if exists else 'created'} (private)")


def push(nb: str) -> None:
    user = username()
    fname, slug, gpu, needs = NOTEBOOKS[nb]
    d = os.path.join(BUILD, f"nb{nb}")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    shutil.copy2(os.path.join(ROOT, "notebooks", fname), d)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": fname, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [f"{user}/{DATASET_SLUG}"],
            "kernel_sources": [f"{user}/{NOTEBOOKS[k][1]}" for k in needs]}
    if gpu:
        meta["machine_shape"] = GPU
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    kaggle("kernels", "push", "-p", d)
    print(f"pushed {user}/{slug} ({'GPU T4' if gpu else 'CPU'}; inputs: dataset + {needs or 'nothing else'})")


def status(nb: str) -> str:
    out = kaggle("kernels", "status", f"{username()}/{NOTEBOOKS[nb][1]}", check=False, capture=True)
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out.upper():
            return s
    return out.strip()


def fetch(nb: str) -> None:
    """Download the kernel output and copy outputs/** and results/** into the project, never overwriting."""
    user, slug = username(), NOTEBOOKS[nb][1]
    d = os.path.join(BUILD, f"out{nb}")
    shutil.rmtree(d, ignore_errors=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", d, "-o", "-q")
    new, kept = 0, 0
    for dirpath, _, files in os.walk(d):
        for f in files:
            src = os.path.join(dirpath, f)
            parts = os.path.relpath(src, d).replace("\\", "/").split("/")
            if WORK in parts:
                parts = parts[parts.index(WORK) + 1:]
                if not parts or parts[0] not in ("outputs", "results"):
                    continue
                t = os.path.join(ROOT, *parts)
            elif f.endswith(".log"):                                  # the kernel's own run log
                t = os.path.join(ROOT, "results", f"{nb}_kaggle.log")
            else:
                continue
            if os.path.exists(t) and not f.endswith(".log"):
                kept += 1; continue
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(src, t); new += 1
    print(f"fetched {new} new files ({kept} already present, left untouched)")


def run(nbs: list, poll: int = 120) -> None:
    for nb in nbs:
        push(nb)
        time.sleep(30)
        while (s := status(nb)) in ("RUNNING", "QUEUED"):
            print(time.strftime("%H:%M:%S"), nb, s, flush=True)
            time.sleep(poll)
        print(nb, "finished:", s, flush=True)
        fetch(nb)
        if s != "COMPLETE":
            raise SystemExit(f"notebook {nb} ended with {s} - see results/{nb}_kaggle.log")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dataset", "push", "status", "fetch", "run"])
    ap.add_argument("nbs", nargs="*")
    ap.add_argument("-m", "--message", default="update")
    a = ap.parse_args()
    if a.cmd == "dataset":
        dataset(a.message)
    elif a.cmd == "status":
        for nb in a.nbs or NOTEBOOKS:
            print(nb, NOTEBOOKS[nb][1], status(nb))
    elif a.cmd == "run":
        run(a.nbs or list(NOTEBOOKS))
    else:
        for nb in a.nbs:
            {"push": push, "fetch": fetch}[a.cmd](nb)
