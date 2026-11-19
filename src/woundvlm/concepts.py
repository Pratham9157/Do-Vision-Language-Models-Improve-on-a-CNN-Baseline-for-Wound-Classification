"""Language side of the study: class prompts, the clinical concept bank, a random-word control bank and the closed
questions for the generative VLM. Frozen before any model was trained.

The concept bank is LLM-authored (as in LaBo, Yang et al. CVPR 2023) from standard wound-care descriptions of each
wound aetiology (location, shape, edge, wound bed, surrounding skin, typical context). Each concept is assigned to the
one class it is most characteristic of; that assignment is the knowledge prior of the concept-bottleneck head.
"""
from __future__ import annotations

import numpy as np

from . import CLASSES

CLASS_NAMES = {"venous": "venous leg ulcer", "pressure": "pressure ulcer", "trauma": "traumatic wound",
               "diabetic": "diabetic foot ulcer", "surgical": "surgical wound", "other": "chronic skin wound"}
TEMPLATES = ["a clinical photo of a {}.", "a close-up photograph of a {}.", "a medical image showing a {}.",
             "a wound care photo of a {}."]

CONCEPTS = {
    "venous": [
        "a shallow ulcer with irregular borders on the lower leg",
        "a wound just above the ankle on the inner side of the leg",
        "brown hemosiderin staining of the skin around the wound",
        "a large weeping wound with moist yellow slough",
        "swollen lower leg with shiny tight skin",
        "dry scaly eczema-like skin around a leg ulcer",
        "several ulcers merging on the calf",
        "thickened hardened skin of the lower leg",
    ],
    "pressure": [
        "a wound over the sacrum or buttocks",
        "a wound over the heel of a bedridden patient",
        "a deep crater-like wound over a bony prominence",
        "black hard eschar covering the wound bed",
        "a round wound with undermined edges",
        "a purple or maroon area of intact skin from deep tissue injury",
        "a wound on the hip or over the tailbone",
        "a patient lying on a hospital bed sheet",
    ],
    "trauma": [
        "a skin tear with a flap of skin",
        "an abrasion with scraped superficial skin",
        "a laceration with jagged irregular edges",
        "bruising around a wound",
        "a fresh scab on the shin",
        "a wound from a fall or injury on the arm or leg",
        "thin fragile elderly skin with a tear",
        "a puncture wound",
    ],
    "diabetic": [
        "an ulcer on the sole of the foot",
        "a punched-out ulcer surrounded by thick callus",
        "an ulcer on a toe or under the ball of the foot",
        "a wound at the site of an amputated toe",
        "dry cracked skin of the foot",
        "a deep ulcer on the plantar foot",
        "a deformed foot with a wound on a pressure point",
        "a dark gangrenous toe",
    ],
    "surgical": [
        "a straight linear incision",
        "sutures or staples along a wound",
        "a surgical wound that has opened along its incision line",
        "a surgical scar with a wound",
        "a wound after an operation on the abdomen",
        "a wound after amputation of a limb",
        "a long narrow wound with straight edges",
        "a drain or dressing next to a surgical incision",
    ],
    "other": [
        "a blister on the skin",
        "a burn wound",
        "a wound with spreading redness of cellulitis",
        "a wound with dry black necrotic tissue on the leg",
        "a cluster of small sores",
        "a fungating wound with a raised mass",
        "a skin rash with open areas",
        "an atypical wound of unclear cause",
    ],
}
assert set(CONCEPTS) == set(CLASSES)

# Neutral vocabulary for the control bank: same number of "concepts" and same text encoder, but no wound knowledge.
_NEUTRAL = ("table window river paper garden pencil mountain cloud chair music orange bottle ticket ladder candle "
            "violin carpet basket rocket anchor lantern jacket meadow puzzle saddle teapot harbor kettle blanket "
            "marble pepper island trumpet feather compass lemon pillow tunnel wagon").split()


def concept_list() -> tuple:
    """(texts, class_index_per_concept) in a fixed order."""
    texts, owner = [], []
    for c in CLASSES:
        for t in CONCEPTS[c]:
            texts.append(t); owner.append(CLASSES.index(c))
    return texts, np.array(owner)


def random_concept_list(seed: int = 0) -> tuple:
    """Control bank: as many 3-word phrases of neutral words as real concepts, assigned to classes in the same proportions."""
    texts, owner = concept_list()
    rng = np.random.default_rng(seed)
    rand = [" ".join(rng.choice(_NEUTRAL, 3, replace=False)) for _ in texts]
    return rand, rng.permutation(owner)


def association(owner: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Class x concept prior: 1/|concepts of class| for concepts owned by the class, 0 otherwise."""
    A = np.zeros((n_classes, len(owner)))
    for k in range(n_classes):
        A[k, owner == k] = 1.0 / max(1, (owner == k).sum())
    return A


def class_prompts() -> dict:
    """Prompt sets for zero-shot: name templates and description prompts (classification by description)."""
    names = {c: [t.format(CLASS_NAMES[c]) for t in TEMPLATES] for c in CLASSES}
    desc = {c: [f"a photo of a {CLASS_NAMES[c]}: {d}." for d in CONCEPTS[c]] for c in CLASSES}
    return {"names": names, "descriptions": desc}


# ---------------------------------------------------------------- generative VLM (closed questions, answer-token logits)
MC_PROMPT = ("This is a clinical photograph of a wound. Which type of wound is it?\n{options}\n"
             "Answer with the letter only.")
LETTERS = "ABCDEF"

ATTRIBUTES = [   # (key, yes/no question). Answer scored as P(Yes) from the next-token logits.
    ("on_foot", "Is the wound on the foot or toes?"),
    ("on_lower_leg", "Is the wound on the lower leg, shin or ankle?"),
    ("on_heel", "Is the wound on the heel?"),
    ("on_sacrum_buttock", "Is the wound on the buttocks, sacrum or hip?"),
    ("linear_incision", "Does the wound follow a straight line, like a surgical incision?"),
    ("sutures_staples", "Are there sutures or staples visible?"),
    ("black_eschar", "Is there black or dark brown dead tissue in the wound?"),
    ("yellow_slough", "Is there yellow tissue covering part of the wound bed?"),
    ("red_granulation", "Is the wound bed mostly bright red?"),
    ("brown_staining", "Is the skin around the wound stained brown?"),
    ("callus_rim", "Is the wound surrounded by thick white or yellow callus?"),
    ("shallow_irregular", "Is the wound shallow with irregular edges?"),
    ("deep_crater", "Is the wound deep, like a crater?"),
    ("leg_swelling", "Does the leg look swollen?"),
    ("skin_flap", "Is there a torn flap of skin?"),
    ("spreading_redness", "Is there redness spreading into the skin around the wound?"),
]
TEXT_AUDIT = ("visible_text", "Is there a ruler, a label or any handwriting visible in this image?")
YES_NO_SUFFIX = " Answer with one word: Yes or No."


def mc_options(order: list) -> str:
    return "\n".join(f"{LETTERS[i]}. {CLASS_NAMES[CLASSES[k]]}" for i, k in enumerate(order))


def mc_orders(n: int = len(CLASSES)) -> list:
    """All n cyclic shifts: every class appears once at every letter position (cancels position bias)."""
    return [[(i + s) % n for i in range(n)] for s in range(n)]
