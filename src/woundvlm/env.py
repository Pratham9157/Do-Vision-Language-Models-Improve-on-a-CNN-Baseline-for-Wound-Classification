"""Paths for a local checkout and for Kaggle. Every notebook starts with setup().

Kaggle layout (the project is a private dataset; earlier notebooks are attached as kernel inputs):
    /kaggle/tmp/LowerLimbWoundVLM/            project root seen by the code
        data     -> <dataset>/data            read-only images + index (symlink, never copied into the output)
        outputs  -> /kaggle/working/LowerLimbWoundVLM/outputs   (saved with the notebook version)
        results  -> /kaggle/working/LowerLimbWoundVLM/results
Outputs/results of every attached input are merged in first (never overwriting), so a notebook sees what earlier
notebooks produced.
"""
from __future__ import annotations

import glob
import os
import shutil

KAGGLE_ROOT = "/kaggle"
WORK_NAME = "LowerLimbWoundVLM"
SKIP_ON_IMPORT = (".pt", ".pth")


def find_kaggle_project(kaggle_root: str = KAGGLE_ROOT) -> str | None:
    hits = sorted(glob.glob(os.path.join(kaggle_root, "input", "**", "src", "woundvlm", "__init__.py"), recursive=True))
    return os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))) if hits else None


def _merge_tree(src: str, dst: str) -> int:
    """Copy src into dst without overwriting files already in dst."""
    n = 0
    for dirpath, _, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        for f in files:
            if f.endswith(SKIP_ON_IMPORT):
                continue
            t = os.path.join(dst, rel, f)
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True)
                shutil.copy2(os.path.join(dirpath, f), t)
                n += 1
    return n


def _kaggle_project(kaggle_root: str) -> str:
    source = find_kaggle_project(kaggle_root)
    if not source:
        raise FileNotFoundError("Kaggle: attach the project dataset (it must contain src/woundvlm/).")
    project = os.path.join(kaggle_root, "tmp", WORK_NAME)
    work = os.path.join(kaggle_root, "working", WORK_NAME)
    os.makedirs(project, exist_ok=True)
    if not os.path.exists(os.path.join(project, "data")):
        os.symlink(os.path.join(source, "data"), os.path.join(project, "data"))
    for kind in ("outputs", "results"):
        os.makedirs(os.path.join(work, kind), exist_ok=True)
        if not os.path.exists(os.path.join(project, kind)):
            os.symlink(os.path.join(work, kind), os.path.join(project, kind))
        n = 0
        for d in sorted(glob.glob(os.path.join(kaggle_root, "input", "**", kind), recursive=True)):
            if os.path.isdir(d):
                n += _merge_tree(d, os.path.join(work, kind))
        print(f"kaggle: {n} files imported into {kind}/")
    return project


def setup(local_project: str | None = None, kaggle_root: str = KAGGLE_ROOT) -> dict:
    in_kaggle = local_project is None and os.path.isdir(os.path.join(kaggle_root, "input"))
    project = _kaggle_project(kaggle_root) if in_kaggle else \
        os.path.abspath(local_project or os.path.join(os.getcwd(), ".."))
    out, results = os.path.join(project, "outputs"), os.path.join(project, "results")
    for d in (out, results):
        os.makedirs(d, exist_ok=True)
    if not os.path.isfile(os.path.join(project, "data", "processed", "index.csv")):
        raise FileNotFoundError("data/processed/index.csv missing - run notebook 01 (locally) first.")
    return {"in_kaggle": in_kaggle, "project": project, "out": out, "results": results}
