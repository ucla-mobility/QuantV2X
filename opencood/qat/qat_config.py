"""
qat_config.py — parsing & validation of the ``qat:`` section in hypes_yaml.
============================================================================

WHY A SEPARATE PARSER (instead of editing yaml_utils.py):
    ``opencood/hypes_yaml/yaml_utils.py`` is shared by every training / eval /
    PTQ entry point in the repo. The safest way to add QAT configuration
    without any chance of perturbing the PTQ baseline is to leave the loader
    completely untouched — YAML keys it does not know about (like ``qat:``)
    already pass through ``load_yaml`` unmodified as plain dict entries.
    This module owns defaulting + validation of that dict, so every QAT
    entry point gets one canonical, fully-populated config object.

Usage:
    hypes = yaml_utils.load_yaml(path, opt)      # unchanged upstream loader
    qat_cfg = get_qat_config(hypes)              # dict with all defaults filled
"""

import copy

# ---------------------------------------------------------------------------
# The single source of truth for QAT defaults. Every key is documented here
# and NOWHERE ELSE — downstream code must not invent hidden defaults.
# ---------------------------------------------------------------------------
DEFAULT_QAT_CONFIG = {
    "weight": {
        # Number of codebook entries K.
        #   3  -> ternary, log2(3) ~= 1.58 bits/weight (the headline target)
        #   4  -> 2-bit
        #   Any K >= 2 is supported by the quantizer (a uniform grid is just
        #   the special case of equally spaced levels), which is how the
        #   "W8 island" fallback for sensitive layers is implemented
        #   (K = 255, learn_levels=False).
        "n_levels": 3,
        # TTQ-style learnable level magnitudes ({-w_n, 0, +w_p} instead of a
        # fixed {-1, 0, +1}). The asymmetry lets each layer place its 3-4
        # representable values where its weight distribution needs them.
        "learn_levels": True,
        # Per-output-channel scale (BitNet b1.58 "absmean" init). Group-wise
        # quantization (the LLM convention) is meaningless for 3x3 convs;
        # per-Cout is the correct granularity for conv nets.
        "channel_wise": True,
        # LSQ/PACT-style gradient clipping: zero the STE gradient for shadow
        # weights that sit far outside the representable range, so weights
        # that have drifted "off the grid" stop accumulating updates that
        # can never be expressed.
        "clip_grad": True,
    },
    "act": {
        # Activation fake-quant is OFF for Epics 0-2 (weights-only QAT);
        # Epic 3 flips this on progressively (A8 -> A6 -> A4).
        "enabled": False,
        "n_bits": 8,
        # 'ema'  : running min/max asymmetric range (bring-up mode, matches
        #          UniformAffineQuantizer(leaf_param=True) semantics)
        # 'lsq'  : learned step size with the 1/sqrt(N*Qp) gradient scaling
        #          from the LSQ paper (final-run mode)
        "mode": "ema",
        # QDrop: with probability (1 - qdrop_prob) an element bypasses
        # quantization during training. Randomly *dropping* the quantization
        # perturbation flattens the loss surface w.r.t. it — the same reason
        # dropout flattens w.r.t. co-adaptation. 1.0 = always quantize.
        "qdrop_prob": 1.0,
    },
    # Modules that are NEVER wrapped (stay FP32). Matched with the same
    # semantics as QuantModel._should_skip_quantization: exact local name,
    # exact full name, or full-name prefix.
    # Defaults mirror the PTQ skip-list instinct: heterogeneity aligners are
    # tiny and sensitive; the UMGM codebook has its own quantization story
    # (it IS a quantizer) and is co-trained in Epic 4, not weight-quantized.
    "skip_names": ["aligner_m1", "aligner_m2", "aligner_m3", "aligner_m4",
                   "codebook"],
    # Modules kept at high precision (K=255 uniform, frozen levels) instead
    # of ternary — the standard "first/last layer" rule for extreme low-bit.
    "high_precision_names": ["shrink_conv", "cls_head", "reg_head",
                             "dir_head"],
    "high_precision_n_levels": 255,
    # Keep the FIRST eligible conv under each `encoder_m*` prefix at high
    # precision (first-layer rule). Resolved during graph surgery in
    # traversal (definition) order — see QATQuantModel for the caveat.
    "first_conv_high_precision": True,
    # Optimization split (consumed by QATQuantModel.qat_param_groups):
    "lr_weight": 2e-5,     # FP32 shadow weights: small lr, EfficientQAT W2 lesson
    "lr_quant": 1e-4,      # scales & levels: they see exact (low-variance) grads
                           # and are few in number, so they tolerate ~5x more lr
    # Telemetry cadence (iterations) for flip-rate / collapse tracking.
    "telemetry_every": 100,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Unknown keys in ``override`` are REJECTED (typo protection): a silently
    ignored ``learn_lvls: false`` would cost a week of confused experiments.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in base:
            raise KeyError(
                f"Unknown qat config key '{k}'. Valid keys at this depth: "
                f"{sorted(base.keys())}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"qat config key '{k}' must be a mapping, "
                                f"got {type(v).__name__}")
            out[k] = _deep_merge(base[k], v)
        else:
            out[k] = v
    return out


def get_qat_config(hypes_or_qat_section) -> dict:
    """Return a fully-defaulted, validated QAT config dict.

    Accepts either the full hypes dict (uses its ``qat`` key, which may be
    absent -> pure defaults) or the ``qat:`` sub-dict directly.
    """
    if hypes_or_qat_section is None:
        section = {}
    elif "qat" in hypes_or_qat_section and isinstance(
            hypes_or_qat_section.get("qat"), dict):
        section = hypes_or_qat_section["qat"]
    elif set(hypes_or_qat_section.keys()) & set(DEFAULT_QAT_CONFIG.keys()):
        section = hypes_or_qat_section
    else:
        # A full hypes dict without a qat section -> defaults.
        section = {}

    cfg = _deep_merge(DEFAULT_QAT_CONFIG, section)

    # ---- validation: fail loudly at config time, not mid-training ---------
    K = cfg["weight"]["n_levels"]
    if not (isinstance(K, int) and K >= 2):
        raise ValueError(f"qat.weight.n_levels must be an int >= 2, got {K}")
    if not (isinstance(cfg["act"]["n_bits"], int)
            and 2 <= cfg["act"]["n_bits"] <= 8):
        raise ValueError("qat.act.n_bits must be an int in [2, 8]")
    if cfg["act"]["mode"] not in ("ema", "lsq"):
        raise ValueError("qat.act.mode must be 'ema' or 'lsq'")
    if not (0.0 < cfg["act"]["qdrop_prob"] <= 1.0):
        raise ValueError("qat.act.qdrop_prob must be in (0, 1]")
    for key in ("skip_names", "high_precision_names"):
        if not isinstance(cfg[key], (list, tuple)):
            raise TypeError(f"qat.{key} must be a list of name prefixes")
    return cfg
