"""env.setup on a fake /kaggle tree: dataset symlinked (not copied), outputs written under working/, earlier
notebooks' outputs merged in without overwriting."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _can_symlink(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe")
        return True
    except OSError:
        return False


def test_kaggle_layout(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not permitted here (Windows without developer mode)")
    from woundvlm import env
    k = tmp_path / "kaggle"
    ds = k / "input" / "lowerlimbwoundvlm" / "LowerLimbWoundVLM"
    (ds / "src" / "woundvlm").mkdir(parents=True)
    (ds / "src" / "woundvlm" / "__init__.py").write_text("")
    (ds / "data" / "processed").mkdir(parents=True)
    (ds / "data" / "processed" / "index.csv").write_text("image_id\n")
    prev = k / "input" / "llw-02-embeddings" / "LowerLimbWoundVLM" / "outputs" / "emb"
    prev.mkdir(parents=True)
    (prev / "siglip.npz").write_text("x")
    (k / "working").mkdir()
    (k / "tmp").mkdir()
    E = env.setup(kaggle_root=str(k))
    assert E["in_kaggle"] and os.path.islink(os.path.join(E["project"], "data"))
    assert os.path.exists(os.path.join(E["project"], "outputs", "emb", "siglip.npz"))
    with open(os.path.join(E["out"], "new.txt"), "w") as fh:
        fh.write("y")
    assert (k / "working" / "LowerLimbWoundVLM" / "outputs" / "new.txt").exists()
    assert not (k / "working" / "LowerLimbWoundVLM" / "data").exists()                # images never copied to output
    env.setup(kaggle_root=str(k))                                                    # idempotent


def test_local_layout():
    from woundvlm import env
    if not os.path.exists(os.path.join(ROOT, "data", "processed", "index.csv")):
        pytest.skip("run notebook 01 first")
    E = env.setup(local_project=ROOT)
    assert not E["in_kaggle"] and E["project"] == ROOT
