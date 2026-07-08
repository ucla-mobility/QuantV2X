# PLAN 2 — Research & Alignment Brief
## Ultra-Low-Bit QAT for Cooperative Perception: Twin-Codebook Co-Training on QuantV2X

*Prepared for structural alignment with the project lead. One page per section; each maps to one slide.*

---

## 1. Core Problem & Solution Space

**Claim:** no existing QAT framework can train QuantV2X at 1.58/2 bits. This is structural, not an engineering gap.

- **FX-tracing frameworks (MQBench-style) cannot see the model.** QuantV2X's forward is dynamically dispatched per modality (`eval(f"self.encoder_{modality}")`), takes dictionary inputs, and branches on `record_len` — none of which is symbolically traceable. Graph-capture-based quantization fails at import time.
- **LLM QAT (EfficientQAT, OmniQuant) assumes the wrong topology.** They iterate homogeneous, identically-shaped transformer blocks over a single hidden state, quantizing `nn.Linear` group-wise. QuantV2X is ~95% convolution — 2D, transposed, and **sparse 3D** — organized as heterogeneous subsystems (voxel encoder → BEV backbone → multi-input pyramid fusion). Group-wise linear quantization has no meaning for a 3×3 conv; per-output-channel is the correct granularity.
- **PTQ has a hard floor below INT4 — this is principled, not empirical bad luck.** AdaRound's soft-rounding objective rests on a second-order Taylor expansion valid for small perturbations; at 3 levels, the "perturbation" *is* the weight. Per the Precision Scaling Law (Kumar et al., arXiv 2411.04330), PTQ degradation *grows* with how well-trained the FP checkpoint is — our checkpoints are exactly such sharp minima. QAT finds a different minimum, flat along the quantization direction.

**Our solution space:** keep QuantV2X's in-house module-swap surgery (the only mechanism that already understands spconv, BEV backbones, and PyramidFusion), replace the quantizer with a custom autograd Function (STE to shadow weights; **exact** gradients to codebook levels and scales), and adopt EfficientQAT's two-phase *recipe* — not its code.

---

## 2. The Twin-Codebook Novelty

The system contains **two discrete vocabularies**, and no prior work trains them jointly:

| | On-device weight codebook | On-air transmission codebook (UMGM) |
|---|---|---|
| Domain | every conv/linear weight tensor | BEV features crossing the V2X channel |
| Grid | learnable ternary `{−w_n, 0, +w_p}` per layer (TTQ-style, absmean-scaled) | k learned codewords × m segments × 3 residual levels |
| Gradient path | custom STE Function (exact grads to levels/scales) | Gumbel-softmax (already differentiable) |
| Budget it buys | model storage & compute | channel bandwidth (m·log₂k bits per BEV cell) |

**Why joint training is necessary, not optional:** the UMGM codebook was fit to *full-precision* aligner outputs. Ternarizing the encoders shifts the feature distribution it tessellates — centroids land in the wrong places, code usage collapses, and channel reconstruction error stacks on top of weight quantization error. PTQ has no gradient path to re-fit codewords; our QAT loop does.

**How they harmonize:** in the global fine-tuning phase, detection loss + codebook reconstruction loss + usage-entropy regularization backpropagate *simultaneously* through ternary encoders and transmission codewords. The token vocabulary re-tessellates around the quantized feature distribution; the encoders, in turn, learn features the vocabulary can represent. Both grids also share infrastructure (entropy regularization, dead-code reassignment) — one theory of discrete representation, two instantiations. **This closed loop is the novel contribution; neither half alone is.**

---

## 3. The Two-Phase Training Strategy

**Phase A — Subsystem-wise reconstruction warm start (Block-AP, generalized).**
EfficientQAT's blockwise distillation assumes transformer blocks; our "blocks" are heterogeneous subsystems. But QuantV2X's PTQ reconstruction drivers already partition the model into exactly those subsystems and cache their I/O — including the hard cases (multi-input fusion under spatial warping, `record_len` batching). We reuse that machinery and change only the optimized set: instead of AdaRound's rounding variables, we train **{FP32 shadow weights, per-channel scales, codebook levels}** against frozen FP32 teacher block outputs (MSE), sequentially encoder → backbone → fusion, on a ~1–2k-frame calibration subset. Cheap, stable, and recovers most of the naive-ternarization gap before any end-to-end step.

**Phase B — Global optimization, then quantization-parameter polish (E2E-QP analogue).**
End-to-end training with full detection loss + FP-teacher distillation (KL on logits, feature-mimic MSE on the *fused* BEV map — fusion is where multi-agent quantization errors meet), progressive activation quantization (A8→A4), live BN folded only at export, and the joint codebook phase from §2. The finale mirrors EfficientQAT's second phase: **shadow weights locked (`requires_grad=False`) — ternary assignments frozen** — while scales, levels, activation ranges, transmission codebook, and BN affine parameters train end-to-end for 2–3 epochs. This decouples *where the discrete points are* from *what they mean*, and is the cheap, stabilizing step that makes ultra-low-bit training land.

Health telemetry throughout: codeword **flip rate** (healthy ≈ 0.1–1%/step), per-layer zero-fraction (collapse alarm), code-usage perplexity.

---

## 4. Validation Metrics & Key Experiments

**Headline hypothesis — the Precision Scaling Law holds for cooperative perception, on both axes of the twin codebook:** at fixed budget, *bigger-at-lower-bits wins*.

1. **Iso-memory sweep (on-device axis):** {1× width @ INT8} vs {2× @ 2-bit} vs {2.5× @ 1.58-bit} — AP@0.5/0.7 vs weight memory on V2X-Real. Prediction: the low-bit/wide configurations dominate down to a knee near 2 bits.
2. **Iso-bandwidth sweep (on-air axis):** codebook (m, k) grid at fixed m·log₂k bits per BEV cell — AP vs bits-on-air. Tests whether the same law governs the transmitted representation.
3. **Baseline ladder:** FP / PTQ-W8A8 / PTQ-W4A8 / QAT-W2 / QAT-W1.58, each with AP, weight memory, message size, latency. Generalization: OPV2V-H, DAIR-V2X-C.
4. **Ablations that isolate the contributions:** joint vs frozen transmission codebook (the §2 claim); learnable vs fixed ternary levels; warm start vs cold end-to-end; distillation off.

**Success criteria:** QAT-W1.58 within a small pre-registered AP gap of FP while beating PTQ-W4; monotone dominance of low-bit-wide configurations on the iso-curves; joint-codebook ablation shows a strictly positive AP margin.

---

*Supporting artifacts: full engineering backlog (Plan 1), runnable quantizer blueprint (`qat_ternary_codebook_blueprint.py`, all three gradient paths self-tested), architecture breakdown with framework survey.*
