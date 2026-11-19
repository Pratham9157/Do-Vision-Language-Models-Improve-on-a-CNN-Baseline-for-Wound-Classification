"""Generative VLM (Qwen2.5-VL-3B-Instruct) used three ways, never by parsing free text:

  1. zero-shot multiple choice: 6 wound types as lettered options; letter log-probabilities from the next-token logits;
     averaged over all 6 cyclic option orders so every class sits at every letter once (cancels the
     letter-position bias of small generative VLMs);
  2. visual attributes: 16 yes/no questions about location, shape, tissue and surrounding skin -> P(Yes) per question,
     an interpretable 16-dim description of the image that downstream heads use as features;
  3. visible-text audit: P(Yes) for "is there a ruler, label or handwriting visible" on the crop and the full photo,
     to measure how much text (a possible shortcut) the models could see.

All prompts for one image go through the model as one left-padded batch (same image repeated).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import CLASSES
from .concepts import ATTRIBUTES, LETTERS, MC_PROMPT, TEXT_AUDIT, YES_NO_SUFFIX, mc_options, mc_orders

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMG = 336                                   # 336 px -> 12 x 12 merged patches = 144 image tokens
PIXELS = IMG * IMG


def load(model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.float16, model=None):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=PIXELS, max_pixels=PIXELS, use_fast=False)
    proc.tokenizer.padding_side = "left"
    if model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval(), proc


def _prompt(proc, text: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def token_ids(proc, words) -> list:
    ids = [proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in words]
    assert len(set(ids)) == len(ids), f"options share a first token: {words}"
    return ids


@torch.no_grad()
def last_logits(model, proc, img: Image.Image, texts: list) -> torch.Tensor:
    dev = next(model.parameters()).device
    enc = proc(text=[_prompt(proc, t) for t in texts], images=[img] * len(texts), padding=True,
               return_tensors="pt").to(dev)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        out = model(**enc).logits[:, -1].float()
    return out.cpu()


def mc_and_attributes(model, proc, crop: Image.Image, full: Image.Image | None = None) -> dict:
    """One image -> dict of class probabilities (mc_p_<class>), mean letter probabilities (letter_<X>, for the
    position-bias check), attribute P(Yes) (attr_<key>) and the text audit (text_crop, text_full)."""
    orders = mc_orders()
    mc_texts = [MC_PROMPT.format(options=mc_options(o)) for o in orders]
    yn_texts = [q + YES_NO_SUFFIX for _, q in ATTRIBUTES] + [TEXT_AUDIT[1] + YES_NO_SUFFIX]
    logits = last_logits(model, proc, crop, mc_texts + yn_texts)
    letters = token_ids(proc, list(LETTERS[:len(CLASSES)]))
    yes_no = token_ids(proc, ["Yes", "No"])
    out = {}
    lp = torch.log_softmax(logits[:len(orders)][:, letters], dim=1).numpy()        # orders x letters
    class_lp = np.zeros((len(orders), len(CLASSES)))
    for i, o in enumerate(orders):
        for pos, k in enumerate(o):
            class_lp[i, k] = lp[i, pos]
    mean_lp = class_lp.mean(0)
    p = np.exp(mean_lp - mean_lp.max()); p /= p.sum()
    out.update({f"mc_p_{c}": float(v) for c, v in zip(CLASSES, p)})
    out.update({f"letter_{L}": float(v) for L, v in zip(LETTERS, np.exp(lp).mean(0))})
    yn = torch.softmax(logits[len(orders):][:, yes_no], dim=1)[:, 0].numpy()
    out.update({f"attr_{k}": float(v) for (k, _), v in zip(ATTRIBUTES, yn[:len(ATTRIBUTES)])})
    out["text_crop"] = float(yn[-1])
    if full is not None:
        out["text_full"] = float(torch.softmax(last_logits(model, proc, full, [yn_texts[-1]])[:, yes_no], 1)[0, 0])
    if not np.isfinite(list(out.values())).all():
        raise FloatingPointError("non-finite VLM scores (fp16 overflow?) - reload the model in float32")
    return out


def run(model, proc, df: pd.DataFrame, project: str, out_csv: str, log_every: int = 50) -> pd.DataFrame:
    """Annotate every row of df (image_id, crop_path, full_path). Resumable: rows already in out_csv are skipped."""
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["image_id"])
    todo = df[~df.image_id.isin(set(done.image_id))]
    rows, t0 = [], time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        crop = Image.open(os.path.join(project, r.crop_path)).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        full = Image.open(os.path.join(project, r.full_path)).convert("RGB").resize((IMG, IMG), Image.BICUBIC) \
            if r.full_path != r.crop_path else None
        rows.append({"image_id": r.image_id, **mc_and_attributes(model, proc, crop, full)})
        if i % log_every == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
            done.to_csv(out_csv, index=False)
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return done
