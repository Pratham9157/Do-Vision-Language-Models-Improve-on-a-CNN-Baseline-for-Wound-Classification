"""Contrastive vision-language encoders: L2-normalised image and text embeddings, computed once on the GPU and saved,
so every head (zero-shot, linear probe, concept bottleneck) runs on CPU afterwards.

outputs/emb/<encoder>.npz holds:
    image_id                 (n,)  str
    img_crop, img_full       (n, d) image embeddings of the two views
    txt_concepts, txt_random (K, d) concept bank / random-word control bank
    txt_names, txt_desc      (m, d) zero-shot prompts; owner_names, owner_desc (m,) class index of each prompt
    logit_scale              ()    the encoder's own temperature (exp already applied)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from PIL import Image

from . import CLASSES
from .concepts import class_prompts, concept_list, random_concept_list

ENCODERS = {   # name: (backend, model id)
    "siglip": ("hf_siglip", "google/siglip-so400m-patch14-384"),
    "clip": ("hf_clip", "openai/clip-vit-large-patch14"),
    "biomedclip": ("open_clip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
}
PRIMARY = "siglip"          # pre-registered encoder for the primary hypotheses


class Encoder:
    def __init__(self, name: str, device: str = "cuda", model=None, processor=None, model_id: str | None = None):
        self.name, self.device = name, device
        self.backend, mid = ENCODERS[name]
        self.model_id = model_id or mid
        if self.backend == "open_clip":
            import open_clip
            if model is None:
                model, self.preprocess = open_clip.create_model_from_pretrained(self.model_id)
            else:
                self.preprocess = processor
            self.tokenizer = open_clip.get_tokenizer(self.model_id)
        else:
            from transformers import AutoModel, AutoProcessor
            self.processor = processor or AutoProcessor.from_pretrained(self.model_id)
            if model is None:
                model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.model = model.to(device).eval()
        self.dtype = next(self.model.parameters()).dtype

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.detach().float().exp().cpu())

    @torch.no_grad()
    def images(self, imgs: list) -> np.ndarray:
        if self.backend == "open_clip":
            x = torch.stack([self.preprocess(im) for im in imgs]).to(self.device, self.dtype)
            f = self.model.encode_image(x)
        else:
            x = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            f = self.model.get_image_features(pixel_values=x)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def texts(self, texts: list) -> np.ndarray:
        if self.backend == "open_clip":
            f = self.model.encode_text(self.tokenizer(texts).to(self.device))
        else:
            if self.backend == "hf_siglip":      # SigLIP was trained on max_length padding without an attention mask
                enc = self.processor(text=texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
                f = self.model.get_text_features(input_ids=enc["input_ids"].to(self.device))
            else:
                enc = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                f = self.model.get_text_features(input_ids=enc["input_ids"].to(self.device),
                                                 attention_mask=enc["attention_mask"].to(self.device))
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def _flatten(prompts: dict) -> tuple:
    texts, owner = [], []
    for c in CLASSES:
        texts += prompts[c]; owner += [CLASSES.index(c)] * len(prompts[c])
    return texts, np.array(owner)


def embed_all(enc: Encoder, df, project: str, out_path: str, batch: int = 32, log_every: int = 20) -> str:
    """Embed both views of every image in df (needs image_id, crop_path, full_path) plus all text banks."""
    res = {"image_id": df.image_id.to_numpy(str)}
    t0 = time.time()
    for view in ("crop", "full"):
        rows = []
        paths = df[f"{view}_path"].tolist()
        for b in range(0, len(paths), batch):
            rows.append(enc.images([Image.open(os.path.join(project, p)).convert("RGB") for p in paths[b:b + batch]]))
            if log_every and (b // batch) % log_every == 0:
                print(f"  {enc.name} {view}: {b + len(rows[-1])}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
        res[f"img_{view}"] = np.concatenate(rows).astype(np.float32)
    ct, _ = concept_list(); rt, _ = random_concept_list()
    res["txt_concepts"], res["txt_random"] = enc.texts(ct), enc.texts(rt)
    pr = class_prompts()
    for key, short in (("names", "names"), ("descriptions", "desc")):
        t, o = _flatten(pr[key])
        res[f"txt_{short}"], res[f"owner_{short}"] = enc.texts(t), o
    res["logit_scale"] = np.array(enc.logit_scale, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **res)
    print(f"  saved {out_path} ({time.time() - t0:.0f}s)")
    return out_path


def load_embeddings(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
