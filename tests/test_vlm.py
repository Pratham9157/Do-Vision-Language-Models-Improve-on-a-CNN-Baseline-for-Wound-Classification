"""VLM code paths on tiny randomly initialised models with the real processors/tokenizers (CPU, internet for the
processor files). Checks plumbing only - outputs are meaningless."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
INDEX = os.path.join(ROOT, "data", "processed", "index.csv")
pytestmark = [pytest.mark.skipif(not importlib.util.find_spec("transformers"), reason="transformers missing"),
              pytest.mark.skipif(not os.path.exists(INDEX), reason="run notebook 01 first")]


def _small_index():
    from woundvlm import data
    idx = data.load_index(ROOT)
    return pd.concat([data.labelled(idx).groupby("label").head(1), data.azh_wounds(idx).head(2)])


def tiny_clip():
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor
    torch.manual_seed(0)
    cfg = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        image_size=224, patch_size=32), projection_dim=16)
    return CLIPModel(cfg), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def tiny_siglip():
    import torch
    from transformers import AutoProcessor, SiglipConfig, SiglipModel
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    cfg = SiglipConfig(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        vocab_size=proc.tokenizer.vocab_size, max_position_embeddings=64),
                       vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                          num_attention_heads=2, image_size=384, patch_size=48))
    return SiglipModel(cfg), proc


@pytest.mark.parametrize("name,factory", [("clip", tiny_clip), ("siglip", tiny_siglip)])
def test_embed_all(name, factory, tmp_path):
    from woundvlm import heads
    from woundvlm.concepts import concept_list
    from woundvlm.embed import Encoder, embed_all, load_embeddings
    model, proc = factory()
    enc = Encoder(name, "cpu", model=model, processor=proc)
    df = _small_index()
    out = embed_all(enc, df, ROOT, str(tmp_path / f"{name}.npz"), batch=4, log_every=0)
    e = load_embeddings(out)
    assert e["img_crop"].shape[0] == len(df) and np.allclose(np.linalg.norm(e["img_full"], axis=1), 1, atol=1e-4)
    assert e["txt_concepts"].shape[0] == len(concept_list()[0]) == e["txt_random"].shape[0]
    assert e["txt_names"].shape[0] == len(e["owner_names"]) and float(e["logit_scale"]) > 0
    space = heads.Space(e, "crop")
    p = heads.fit_predict(space, "zs_desc", [], [], [], {"t": df.image_id})["t"]
    assert p.shape == (len(df), 6)


def tiny_qwen25vl():
    import torch
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {**cfg.rope_scaling, "mrope_section": [2, 3, 3]}
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 2, 32, 64, 2, 64
    v.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(cfg).float()


def test_genvlm_scores_and_resume(tmp_path):
    from PIL import Image
    from woundvlm import CLASSES, genvlm
    from woundvlm.concepts import ATTRIBUTES
    model, proc = genvlm.load(device="cpu", model=tiny_qwen25vl())
    df = _small_index().head(3)
    img = Image.open(os.path.join(ROOT, df.crop_path.iloc[0])).convert("RGB").resize((genvlm.IMG, genvlm.IMG))
    enc = proc(text=[genvlm._prompt(proc, "q")], images=[img], return_tensors="pt")
    n_img = int((enc["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    assert n_img == (genvlm.IMG // 28) ** 2
    out = genvlm.mc_and_attributes(model, proc, img, img)
    assert np.isclose(sum(out[f"mc_p_{c}"] for c in CLASSES), 1, atol=1e-5)
    assert all(0 < out[f"attr_{k}"] < 1 for k, _ in ATTRIBUTES) and 0 < out["text_full"] < 1
    # batched scoring == one-at-a-time scoring (left padding must not change the answer logits)
    texts = ["Is the wound on the foot or toes? Answer with one word: Yes or No.",
             "Are there sutures or staples visible? Answer with one word: Yes or No."]
    both = genvlm.last_logits(model, proc, img, texts)
    single = genvlm.last_logits(model, proc, img, texts[1:])
    assert np.allclose(both[1].numpy(), single[0].numpy(), atol=1e-3)
    csv = str(tmp_path / "ann.csv")
    genvlm.run(model, proc, df.head(2), ROOT, csv, log_every=1)
    res = genvlm.run(model, proc, df, ROOT, csv, log_every=1)                      # resumes: only 1 new row
    assert len(res) == 3 and res.image_id.is_unique
