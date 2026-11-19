"""Wound-type classification on lower-limb / foot wound photographs (WoundSeg, Oota et al. WACV 2023, redistributed in
the Lower Limb and Feet Wound Image Dataset, Data in Brief 2026) with an external test on AZH (Anisuzzaman et al. 2022).

Question: does vision-language knowledge (contrastive VLM concept bottlenecks, zero-shot text classifiers, generative
VLM attributes, VLM pseudo-labels) add to a supervised CNN when labels are few and when the data come from another
clinic? The pre-registered protocol is summarised in the README.
"""
CLASSES = ["venous", "pressure", "trauma", "diabetic", "surgical", "other"]
SHARED_AZH = ["venous", "pressure", "diabetic", "surgical"]          # classes present in both datasets
__version__ = "0.1.0"
