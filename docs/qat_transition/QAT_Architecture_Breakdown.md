# QuantV2X → QAT Transition: Architectural Breakdown & Implementation Plan

Prepared for the PTQ→QAT research push targeting 1.58-bit (ternary) and 2-bit weights with a learnable codebook. All file references are relative to the QuantV2X repo root.

---

## 1. Strategic Context & Vision Alignment

### 1.1 What QuantV2X does today

QuantV2X (ECCV 2026) is a fully quantized multi-agent cooperative perception system built on the OpenCOOD/HEAL lineage. The runtime pipeline per frame is:

```
per-agent sensor → modality encoder (PointPillars / SECOND-spconv / LiftSplatShoot)
                 → BEV backbone (ResNetBEVBackbone / BaseBEVBackbone)
                 → aligner (AlignNet, heterogeneity adapter)
                 → [codebook encode → transmit tokens → codebook decode]   ← the V2X channel
                 → spatial warp (normalize_pairwise_tfm / affine_matrix)
                 → PyramidFusion (multiscale collaborative fusion, occ-weighted)
                 → shrink_conv → cls / reg / dir heads
```

Two independent quantization subsystems already exist:

**(a) Model quantization — PTQ, `opencood/quant/`.** A direct descendant of the BRECQ/QDrop family: `QuantModule` wraps `Conv2d`/`ConvTranspose2d`/`Linear` (plus `QuantSpconvModule` for sparse 3D convs), `UniformAffineQuantizer` does asymmetric uniform fake-quant with MSE/minmax/entropy scale search, `AdaRoundQuantizer` learns a ±1 rounding policy, and `block_recon.py` / `pyramid_recon.py` / `encoder_recon.py` / `second_recon.py` / `lss_recon.py` do blockwise output reconstruction against cached FP activations. `QuantModel.quant_module_refactor` performs the graph surgery via the `opencood_specials` registry (`QuantPyramidFusion`, `QuantResNetBEVBackbone`, `QuantPointPillar`, `QuantSECOND`, `QuantLiftSplatShoot`, ...), with BN folding (`fold_bn.py`) and a skip list (`aligner_m*` are deliberately left unquantized).

**(b) Message quantization — the Codebook Module, `opencood/models/sub_modules/codebook.py`.** `UMGMQuantizer` is a multi-stage residual vector quantizer: features `[N, C, H, W]` are flattened to `[N·H·W, C]`, split into `m = seg_num` groups, and each group is snapped to one of `k = dict_size` learned codewords per level (3 residual levels). Assignment is made differentiable with temperature-scaled Gumbel-softmax; dead codes are recycled via `reAssignCodebook` with EMA usage frequencies. What crosses the V2X channel is codebook **indices** — `m · log2(k)` bits per BEV cell — which is the "discrete transmission token" mechanism. It is trained in stages 2/3 (`train_stage2.py`, `train_stage3.py`) with everything else frozen, minimizing feature-reconstruction MSE.

### 1.2 Why cooperative perception matters (impact)

A single vehicle's perception fails in exactly the situations that kill people: occlusion (a pedestrian behind a bus), limited field of view at intersections, sparse LiDAR returns at range, and sensor degradation. Intermediate-fusion V2X perception lets an ego vehicle fuse BEV features from other vehicles and roadside units, effectively giving it non-line-of-sight vision and a several-fold FoV expansion. The binding real-world constraints are the ones QuantV2X attacks: V2X channel bandwidth (C-V2X sidelink is a few Mbps shared across agents), end-to-end latency budgets (~100 ms before features are stale and warping errors dominate), and on-vehicle compute/memory. Quantization is the lever on all three simultaneously — smaller messages, faster inference, smaller models.

### 1.3 The core motivation for QAT (through the Precision Scaling Law lens)

The referenced paper (arXiv 2411.04330, "Scaling Laws for Precision", Kumar et al.) formalizes the intuition driving this project: quantizing a model to precision P reduces its **effective parameter count** N_eff ≈ N·(1 − e^(−P/γ)); loss follows a Chinchilla-style law in N_eff. Two consequences matter to us:

1. **Bigger-at-lower-bits wins.** At a fixed weight-memory / transmission budget (N × P = const), a larger backbone at 1.58–2 bits retains more effective capacity than a smaller backbone at INT8, down to a knee somewhere around 2 bits. For V2X this applies twice — to the on-vehicle model and, via the codebook, to the transmitted representation.
2. **PTQ degradation grows with how well-trained the model is.** Their inference-time result: post-training quantization hurts *more* as the model is trained on more data — the loss surface sharpens around the FP minimum. Our FP checkpoints are exactly such well-trained minima, which is why PTQ collapses below ~INT4. Training *in* low precision (QAT) is the regime the paper finds compute-optimal; it finds a *different* minimum that is flat with respect to the quantization perturbation.

### 1.4 Hard limits of the current PTQ pipeline below INT4

- **Hard-coded floor.** `UniformAffineQuantizer` asserts `2 <= n_bits <= 8`, and nothing in the stack understands 3-level (ternary) grids — 1.58-bit is not representable at all. The grid is uniform-affine; a ternary/codebook grid is non-uniform by construction.
- **AdaRound's premise collapses.** AdaRound optimizes a *soft ±1 rounding decision* around a fixed grid, justified by a second-order Taylor expansion valid for small perturbations. At 4 levels (2-bit) the rounding step Δ is on the order of the whole weight distribution's width; at 3 levels the "perturbation" IS the weight. The quadratic approximation, and with it the entire calibration-time reconstruction logic, stops being meaningful.
- **Local objectives, compounding global error.** Blockwise reconstruction minimizes per-block output MSE on a small calibration set. Errors from encoder blocks propagate through the aligner, codebook, spatial warp, and fusion — and in a multi-agent system every collaborator's quantization error enters the ego's fusion. At INT8 these residuals are noise; at 2 bits they are structured distortions no downstream layer was trained to absorb.
- **Frozen statistics.** BN folding (`fold_bn.py`) and activation-range search happen once, against FP-era statistics. Ternary weights shift activation distributions so far that folded BN parameters and cached ranges become wrong, and PTQ has no mechanism to move them.
- **The codebook cannot co-adapt.** The UMGM codebook was fit (stage 2/3) to *full-precision* aligner outputs. PTQ then quantizes the encoders underneath it, shifting the feature distribution the codebook tessellates — centroids sit in wrong places, code usage collapses, and reconstruction error is added on top of weight-quantization error. PTQ has no gradient path to re-fit codewords jointly.

### 1.5 How QAT solves this

- **Loss-aware weight placement.** With fake-quant in the forward pass and STE gradients, the task loss reshapes the *shadow* weight distribution: weights migrate toward codebook levels (tri-modal clustering around {−α, 0, +α} is observable within a few epochs), so clamping stops destroying information the network still relies on. The network learns a function that is *flat along the quantization direction* rather than approximating the FP function post hoc.
- **Learned quantization geometry.** Per-channel scales (absmean-initialized, gradient-trained) and learnable codebook levels (TTQ-style asymmetric ternary {−w_n, 0, +w_p}) let each layer place its 3–4 representable values where its distribution needs them — recovering a large fraction of the entropy a fixed uniform grid throws away.
- **Statistics re-adaptation.** BN stays live during QAT (folded only at export), activation ranges are EMA/learned, so all normalization statistics track the quantized regime.
- **Joint codebook optimization.** Because the UMGM quantizer is already differentiable (Gumbel-softmax), QAT can finally close the loop: detection loss + codebook loss backpropagate through ternary encoders *and* codewords simultaneously, so the transmitted token vocabulary is optimized for the quantized system end-to-end.
- **Distillation as an entropy bridge.** The FP teacher's soft predictions and fused BEV features regularize the low-capacity student, directly compensating the information-entropy loss of extreme clamping.

---

## 2. Online Repository & Alternative Framework Search

Surveyed (July 2026): EfficientQAT, OmniQuant, BRECQ, BitNet b1.58 implementations, MQBench, plus the classic ternary-QAT lineage (TWN/TTQ/LSQ) they build on.

| Framework | What it actually is | Ternary / non-uniform? | Conv/BEV-friendly? | Verdict for QuantV2X |
|---|---|---|---|---|
| **EfficientQAT** ([OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)) | Two-phase QAT for LLMs: **Block-AP** (blockwise training of *all* params — weights + scales — against FP block outputs) then **E2E-QP** (end-to-end training of quantization params only). Uniform INT, W2 minimum, group-wise on `nn.Linear`. | No — uniform grid, ≥2 bit | No — iterates `model.layers` of Llama-style transformers | **Adopt the recipe, not the code.** Block-AP ≈ our existing `block_recon` with weight gradients enabled; E2E-QP ≈ a scales-only fine-tune phase. |
| **OmniQuant** ([OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)) | Blockwise *learnable clipping* (LWC) + equivalent transformations (LET); weights stay frozen — it is gradient-assisted PTQ, not full QAT. LLM-only. | No | No | Skip as a base. LWC is a good idea for our activation quantizers. |
| **BRECQ** ([yhhhli/BRECQ](https://github.com/yhhhli/BRECQ)) | Block-reconstruction PTQ for CNNs. | No | Yes | Already in-house — `opencood/quant/` *is* a BRECQ/QDrop descendant. It is the baseline we are surpassing; its block partitioning survives as the QAT warm-start curriculum. |
| **BitNet b1.58** (paper + [microsoft/BitNet](https://github.com/microsoft/BitNet) for inference; trainable `BitLinear` implementations: [schneiderkamplab/bitlinear](https://github.com/schneiderkamplab/bitlinear), [1bitLLM reproductions], [kotak-ai/1.58BitNet](https://github.com/kotak-ai/1.58BitNet)) | Native ternary QAT: FP16 shadow weights, per-tensor **absmean** scale, `RoundClip(W/α)` → {−1,0,+1} in forward, STE backward, 8-bit activations. | **Yes — native 1.58-bit** | Linear-only as shipped; the quantizer math is layer-agnostic | **Adopt the quantizer math.** absmean scaling + ternary RoundClip + STE is exactly our weight path; we extend it to conv/spconv and make the levels learnable. |
| **MQBench** ([ModelTC/MQBench](https://github.com/ModelTC/MQBench)) | Reproducible QAT benchmark for CNNs/detection: LSQ, LSQ+, DoReFa, PACT, QDrop, APoT under one API, hardware-deployable configs, down to 2 bit. | Partially (APoT non-uniform; no ternary) | **Yes — CNN-native** | **Reference implementation for QAT correctness** (LSQ scale-gradient calibration, per-channel conv handling). Do *not* adopt its torch.fx tracing: QuantV2X's heterogeneous forward (`eval(f"self.encoder_{modality}")`, dict inputs, `record_len`-dependent control flow, spconv tensors) is not symbolically traceable. Our module-swap surgery is the right mechanism. |
| Classic ternary lineage: TWN, **TTQ** (Trained Ternary Quantization), **LSQ** | The algorithmic core of ternary QAT: TTQ learns asymmetric level magnitudes w_p/w_n; LSQ learns the step size with a calibrated gradient. | Yes | Yes | These are the ingredients our custom quantizer composes. |

### Contrast with EfficientQAT's Block-AP and the recommendation

EfficientQAT's Block-AP phase assumes a chain of *homogeneous, identically-shaped* transformer blocks with a single hidden-state input — that layout does not exist in QuantV2X. Our "blocks" are heterogeneous subsystems (spconv voxel encoder, BEV CNN backbone, pyramid fusion with `(features, record_len, affine_matrix, agent_modality_list)` inputs). But this is a solved problem in-house: `encoder_recon / pyramid_recon / second_recon / lss_recon` already partition the model into exactly those subsystems and cache their I/O for reconstruction. Block-AP is that machinery with `weight.requires_grad = True`.

**Cleanest path (hybrid):**

1. **Keep QuantV2X's own graph surgery** (`QuantModel` + `opencood_specials`) — it already understands BEV backbones, spconv, and PyramidFusion; nothing external does.
2. **Replace the quantizer**, not the wrapper: BitNet-style absmean ternary + TTQ learnable levels + LSQ-style learnable scale, implemented as a custom `autograd.Function` (Section 5) so the non-uniform codebook gets true gradients.
3. **Adopt EfficientQAT's two-phase schedule**: blockwise all-parameter warm start (reusing the recon caches), then end-to-end fine-tuning; final phase trains quant-params + codebook only (E2E-QP analogue).
4. **Use MQBench as the correctness oracle** for STE/LSQ details (gradient scaling 1/√(N·Q_max), per-channel conv semantics), not as a framework dependency.

---

## 3. Architecture & Framework Compatibility Analysis

### 3.1 Structural alignment map

| QuantV2X subsystem | PTQ handling today | EfficientQAT analogue | QAT translation |
|---|---|---|---|
| `encoder_m*` (PointPillar / SECOND spconv / LSS) | `QuantPointPillar` / `QuantSECOND` / `QuantLiftSplatShoot` + `encoder_recon` / `second_recon` / `lss_recon` | a "block" | Blockwise warm start vs FP teacher outputs; spconv needs `QATQuantSpconvModule` (note: the PTQ `QuantSpconvModule.forward` re-wraps weights in `nn.Parameter` each call — this silently breaks the autograd graph and **must** be fixed for QAT by calling spconv's functional path or assigning `.data`) |
| `backbone_m*` (ResNet/Base BEV) | `QuantResNetBEVBackbone`, `block_recon` | a "block" | Same recon loop, weights trainable |
| `aligner_m*` | **skipped** (`specials_unquantized_names`) | — | Keep FP or 8-bit; heterogeneity adapters are tiny and sensitive |
| Codebook (UMGM) | untouched by PTQ; trained in stage 2/3 | — (no analogue) | Co-trained in QAT Phase 4 (see roadmap) |
| `pyramid_backbone` (PyramidFusion) | `QuantPyramidFusion` + `pyramid_recon` (multi-input caching already solved) | a "block" | Same; fusion is where multi-agent quantization errors meet — give it the distillation feature-mimic loss |
| `shrink_conv`, heads | `QuantDownsampleConv`, `QuantModule` | lm_head (kept high precision) | Keep 8-bit (first/last-layer rule for extreme low-bit) |

Key insight: **block-wise reconstruction translates to our intermediate-fusion architecture as subsystem-wise reconstruction, and the codebase already implements the hard part** (I/O caching for multi-input fusion blocks under warping, `record_len` batching, modality routing). The delta from PTQ-reconstruction to Block-AP-style QAT inside each recon loop is: optimizer over `{shadow weights, scales, levels}` instead of `{AdaRound α, act δ}`, loss unchanged (block-output MSE / smooth-L1 vs FP reference).

### 3.2 Structural conflicts to plan around

- **Layer types**: EfficientQAT/OmniQuant only touch `nn.Linear` with group-wise quant. QuantV2X is ~95% conv (2D + sparse 3D + transposed). Per-output-channel scales are the correct granularity; group-wise is meaningless for 3×3 convs.
- **BN folding**: the PTQ flow folds BN *before* calibration. For ternary QAT, fold-then-train is unstable (folded scale absorbs BN γ/σ, which then drifts). Train with live BN → fold at export. `QuantModel(..., is_fusing=False)` already provides the unfused path.
- **FX-tracing frameworks** (MQBench) are incompatible with the dynamic heterogeneous forward; module-swap (already in-house) is the compatible mechanism.
- **Two quantizers, one system**: weight ternarization (on-device) and codebook tokens (on-air) are orthogonal but *interacting* — ternary encoders shift the feature distribution the codebook tessellates. They must be trained jointly at least in the final phase, which no external framework supports; this is the genuinely novel engineering in the project.

### 3.3 Can the training mechanics accommodate the custom 1.58-bit codebook?

Yes, cleanly, because blockwise reconstruction and end-to-end QAT are **quantizer-agnostic** — they only require (a) a fake-quant forward and (b) a gradient path. The custom `autograd.Function` in Section 5 supplies both for a non-uniform grid:

- Nearest-codeword snap in a scale-normalized domain generalizes uniform rounding (uniform grid = special case of equally spaced levels).
- STE covers the shadow-weight path; **exact** gradients (not STE) reach the codebook levels and scales, because `w_q = α · levels[idx]` is differentiable in α and `levels` for fixed assignments — the same trick VQ-VAE uses for its codebook. The vanilla `round_ste` in `quant_layer.py` cannot do this (its `.detach()` severs the level/scale path), which is precisely why a custom Function is required.
- The UMGM feature codebook keeps its own Gumbel-softmax gradient path; joint training just sums losses.

One further alignment worth exploiting: the weight codebook per layer ({−w_n, 0, +w_p}, 3 levels) and the feature codebook (k entries per segment) can share the entropy-regularization and dead-code-reassignment utilities already written in `codebook_utils.py`.

---

## 4. Step-by-Step Summer Implementation Roadmap (~12 weeks)

### Phase 0 — Scaffolding (weeks 1–2)

Create `opencood/qat/` (leave `opencood/quant/` untouched as the PTQ baseline):

- `ternary_quant.py` — `CodebookQuantSTE` (autograd.Function), `TernaryCodebookQuantizer`, `ActFakeQuant` (or reuse `UniformAffineQuantizer(leaf_param=True)` for activations).
- `qat_module.py` — `QATQuantModule` (Conv2d/ConvTranspose2d/Linear) and `QATQuantSpconvModule` (with the autograd-safe weight override fix).
- `qat_model.py` — copy `QuantModel.quant_module_refactor` semantics; same `opencood_specials` registry and skip-list; `is_fusing=False` (no BN fold).
- Config: a `qat:` section in `hypes_yaml` — per-subsystem `n_levels` (3 = ternary, 4 = 2-bit), `act_bits`, `learn_levels`, `skip_names`, distillation weights.
- **Tests**: `torch.autograd.gradcheck` on the Function (float64, small tensors); FP-bypass equivalence (`set_quant_state(False)` reproduces baseline outputs bit-exactly); state_dict save/load round-trip; a 10-step overfit test showing loss decrease with quant enabled.

### Phase 1 — Shadow weights + blockwise warm start (weeks 3–4)

**Shadow-weight structure (explicit):**
- `self.weight = nn.Parameter(org_module.weight.detach().clone())` — the FP32 master, and the *only* copy the optimizer sees. Quantized weights are re-materialized every forward and never stored. Checkpoints save shadow weights + scales + levels; discrete codes are an *export artifact*, not training state.
- Initialization from baseline: load the FP checkpoint via the existing `train_utils` loaders → run `qat_model` conversion (weights copied into wrappers) → per-channel absmean scale init: `α_c = mean(|W_c|)` (BitNet rule) → levels init `{−1, 0, +1}`.

**Blockwise warm start (Block-AP analogue):** reuse the recon drivers (`encoder_recon`, `block_recon`, `pyramid_recon`) but swap the optimized parameter set to `{shadow weights (lr≈2e-5), scales & levels (lr≈1e-4)}`, loss = block-output MSE vs cached FP outputs, ~1–2k calibration frames, sequential encoder → backbone → fusion → heads. This alone should recover most of the PTQ-at-2-bit gap and gives a sane init for end-to-end training.

### Phase 2 — End-to-end ternary QAT, weights only (weeks 5–6)

- Full detection loss (cls + reg + dir + occ, existing `opencood/loss`) with activations still FP; weight quant ON everywhere except: first conv of each encoder, `aligner_m*`, detection heads → keep 8-bit or FP (standard extreme-low-bit practice; the PTQ skip-list already encodes this instinct).
- **Distillation**: FP teacher (same architecture) → KL on cls logits, smooth-L1 on reg, plus feature-mimic MSE on the fused BEV map (post-`pyramid_backbone`). This is the main antidote to ternary entropy loss.
- Hyperparameters (EfficientQAT's W2 lessons transfer): shadow-weight lr 1–2e-5 AdamW, quant-param lr ~1e-4, cosine decay, grad-clip 1.0, 10–15 epochs on V2X-Real. Watch the **flip rate** (fraction of weights changing codeword per iteration): healthy ≈ 0.1–1%; >5% → lower lr; ~0 → raise it or unfreeze levels.

### Phase 3 — Activation quantization (weeks 7–8)

- Progressive: A8 → A6 → A4, EMA-range or LSQ-learned scales, initialized with the existing MSE search (`set_act_quantize_params`). Enable QDrop-style stochastic bypass (`prob < 1`) early, anneal to 1.0.
- BN: keep live through Phase 3; freeze BN running stats for the last 2 epochs; fold at export and validate folded-vs-unfolded equivalence at eval.

### Phase 4 — Joint codebook co-training + E2E-QP analogue (weeks 9–10)

- Unfreeze the UMGM codebook (stage-2/3 machinery already isolates its params). Total loss: `L_det + λ1·codebook_loss (MSE) + λ2·usage-entropy penalty + λ3·distill`. Anneal Gumbel temperature (e.g. 1.0 → 0.3); run `reAssignCodebook` every N iterations (utility exists); consider swapping Gumbel assignment for hard nearest-neighbor + STE at the end for train/infer consistency.
- **E2E-QP analogue (final polish)**: freeze shadow weights (ternary assignments locked), train only `{scales, levels, act ranges, codebook, BN affine}` end-to-end for 2–3 epochs — cheap, stabilizing, directly mirrors EfficientQAT's second phase.

### Phase 5 — Evaluation, scaling-law study, export (weeks 11–12)

- AP@0.5/0.7 on V2X-Real (then OPV2V-H, DAIR-V2X-C) vs: FP baseline, PTQ-W8A8, PTQ-W4A8, QAT-W2, QAT-W1.58 — plus latency and message-size columns.
- **Scaling-law experiment** (the headline result): iso-memory sweep — e.g. {1× width @ INT8} vs {2× width @ 2-bit} vs {2.5× width @ 1.58-bit}, and iso-bandwidth codebook sweeps (m, k). Plot AP vs weight-memory and AP vs bits-on-air to test the "bigger-at-lower-precision" prediction in the V2X setting.
- Export: pack ternary weights (2 bits/weight trivially; 1.6 bits via 5 trits/byte base-3 packing), codebook indices for transmission; TensorRT path via the existing `build_trt_int8.py` for A8, custom bitpacked GEMM/im2col kernel as stretch goal.

**Risk register**: spconv autograd fix is a hard prerequisite (Phase 0); ternary collapse of small layers → keep them 8-bit; BN drift → freeze-then-fold protocol; codebook usage collapse under ternary features → entropy penalty + reassignment cadence; PyTorch 1.12 pin → the blueprint below uses no post-1.12 APIs.

---

## 5. PyTorch Code Blueprint

Full runnable file: **`docs/qat_transition/qat_ternary_codebook_blueprint.py`** (self-test in `__main__` verifies all three gradient paths and a training loop; compatible with the repo's torch 1.12 env). API deliberately mirrors `opencood/quant/quant_layer.QuantModule` (`org_module`, `fwd_func`, `fwd_kwargs`, `set_quant_state`) so `QuantModel`-style graph surgery can be reused nearly unchanged.

The core — a custom `torch.autograd.Function` with an STE path for the shadow weights and **exact** gradients for the codebook and scale:

```python
class CodebookQuantSTE(torch.autograd.Function):
    """w_q = scale * levels[argmin_k |w/scale - levels_k|]   (non-uniform snap)"""

    @staticmethod
    def forward(ctx, weight, levels, scale, clip_grad=True):
        w_n = weight / scale                       # scale-normalized domain
        idx = torch.argmin(                        # nearest-codeword assignment
            (w_n.unsqueeze(-1) - levels.view(*([1]*w_n.dim()), -1)).abs(), dim=-1)
        w_q = scale * levels[idx]                  # differentiable in scale & levels
        ctx.save_for_backward(w_n, levels, scale, idx)
        ctx.clip_grad = clip_grad
        return w_q

    @staticmethod
    def backward(ctx, g):
        w_n, levels, scale, idx = ctx.saved_tensors
        # (a) STE to shadow weights: identity, clipped outside representable range
        grad_w = g.clone()
        if ctx.clip_grad:
            lo, hi = levels.min() * 1.5, levels.max() * 1.5
            grad_w = grad_w * ((w_n >= lo) & (w_n <= hi)).to(g.dtype)
        # (b) EXACT grad to codebook levels: dw_q/dlevels[k] = scale where idx==k
        flat_g = (g * scale.expand_as(g)).reshape(-1)
        grad_levels = torch.zeros_like(levels).scatter_add_(0, idx.reshape(-1), flat_g)
        # (c) EXACT grad to scale: dw_q/dscale = levels[idx]
        g_scale = g * levels[idx]
        reduce_dims = [d for d in range(g.dim()) if scale.shape[d] == 1]
        grad_scale = g_scale.sum(dim=reduce_dims, keepdim=True)
        return grad_w, grad_levels, grad_scale, None
```

Why not the existing `round_ste` trick (`(x.round() - x).detach() + x`)? It passes gradients to `x` only — the `.detach()` severs any path to quantization parameters. With a learnable non-uniform codebook, levels and scales must *also* receive gradients, and those gradients are exact (gather/scatter), not estimated — only the discrete *assignment* needs the STE.

The wrapper (see full file for `TernaryCodebookQuantizer`, `ActFakeQuant`, `convert_to_qat`, and the self-test):

```python
class QATQuantModule(nn.Module):
    def __init__(self, org_module, n_levels=3, act_bits=8, learn_levels=True):
        super().__init__()
        ...  # fwd_func/fwd_kwargs exactly as in opencood.quant.quant_layer.QuantModule
        # FULL-PRECISION SHADOW WEIGHTS — the only master copy, owned by the optimizer
        self.weight = nn.Parameter(org_module.weight.detach().clone())
        self.weight_quantizer = TernaryCodebookQuantizer(n_levels, learn_levels=learn_levels)
        self.weight_quantizer.init_scale(self.weight)   # absmean, per out-channel;
                                                        # eager: before optimizer creation

    def forward(self, x):
        # re-quantize the shadow weight EVERY forward; w_q is a temporary
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        out = self.fwd_func(x, w, self.bias, **self.fwd_kwargs)
        return self.act_quantizer(out) if self.use_act_quant else out

    @torch.no_grad()
    def export_codes(self):   # deployment: discrete transmission/storage form
        ...                   # returns (int8 codes, levels[K], per-channel scale)
```

Training-loop contract: `optimizer.step()` updates FP32 shadow weights by full-precision gradient increments (changes usually too small to flip a codeword — accumulation across steps is what flips them, which is exactly why the FP master must never be overwritten by its quantized image); scales and levels update from their exact gradients; `export_codes()` emits the ternary tensor + codebook only at deployment time.

---

### Sources

- [QuantV2X paper](http://arxiv.org/abs/2509.03704) · [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT) ([paper](https://arxiv.org/abs/2407.11062)) · [Scaling Laws for Precision, arXiv 2411.04330](https://arxiv.org/abs/2411.04330)
- [OmniQuant](https://github.com/OpenGVLab/OmniQuant) ([ICLR'24 paper](https://arxiv.org/pdf/2308.13137)) · [BRECQ](https://github.com/yhhhli/BRECQ) · [MQBench](https://github.com/ModelTC/MQBench)
- BitNet b1.58: [microsoft/BitNet](https://github.com/microsoft/BitNet) · [bitnet-b1.58-2B-4T report](https://arxiv.org/pdf/2504.12285) · [schneiderkamplab/bitlinear](https://github.com/schneiderkamplab/bitlinear) · [BitNet b1.58 Reloaded (small nets)](https://arxiv.org/pdf/2407.09527) · [When are 1.58 bits enough?](https://arxiv.org/pdf/2411.05882)
- QAT scaling: [Scaling Law for Quantization-Aware Training](https://export.arxiv.org/abs/2505.14302)
