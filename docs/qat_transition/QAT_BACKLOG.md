# PLAN 1 — QAT Engineering Backlog (QuantV2X: PTQ → 1.58/2-bit QAT)

Granular, dependency-ordered task backlog. Each task = one Jira/GitHub issue. IDs are `QAT-<epic>.<task>`. All paths relative to QuantV2X repo root. `opencood/quant/` remains untouched as the PTQ baseline; all new code lives in `opencood/qat/`.

**Prerequisite (DONE):** the `QuantSpconvModule` autograd fix — parameter re-registration removed from `forward`, quantized weight assigned as a plain tensor attribute so its `grad_fn` survives. Epic 0 ports this exact pattern into `QATQuantSpconvModule` and adds a regression test so it can never silently reappear.

---

## EPIC 0 — Scaffolding & Quantizer Core (weeks 1–2)

Dependency root. Nothing downstream starts until QAT-0.7 gate passes.

### QAT-0.1 — Package skeleton + config schema

**Create:**

```
opencood/qat/__init__.py
opencood/qat/ternary_quant.py      # Function + quantizers (QAT-0.2/0.3)
opencood/qat/qat_module.py         # dense wrappers (QAT-0.4)
opencood/qat/qat_spconv.py         # sparse wrapper (QAT-0.5)
opencood/qat/qat_model.py          # graph surgery (QAT-0.6)
opencood/qat/telemetry.py          # flip rate / grad / usage tracking (QAT-0.7)
opencood/qat/policy.py             # freeze & skip policy (Epic 2)
opencood/qat/distill.py            # teacher losses (Epic 2)
opencood/qat/qat_recon.py          # blockwise warm start (Epic 1)
opencood/qat/export.py             # code emission + trit packing (Epic 5)
tests/qat/                         # all gates below
```

**Modify:** `opencood/hypes_yaml/<experiment>.yaml` — add a `qat:` section, parsed by `qat_model.py`:

```yaml
qat:
  weight:
    n_levels: 3            # 3 = ternary (1.58b), 4 = 2-bit
    learn_levels: true     # TTQ-style {-w_n, 0, +w_p}
    channel_wise: true
    clip_grad: true
  act:
    enabled: false         # flipped on in Epic 3
    n_bits: 8
    mode: lsq              # lsq | ema
    qdrop_prob: 0.5
  skip_names: [aligner_m1, aligner_m2, aligner_m3, aligner_m4]   # stays FP (mirrors PTQ specials_unquantized_names)
  high_precision_names: [shrink_conv, cls_head, reg_head, dir_head]  # W8 fallback, not ternary
  first_conv_high_precision: true
  distill: {kl_cls: 1.0, l1_reg: 1.0, feat_mimic: 0.5, temperature: 2.0}
  codebook: {joint: false, lambda_mse: 1.0, lambda_entropy: 0.01,
             gumbel_tau_start: 1.0, gumbel_tau_end: 0.3, reassign_every: 500}
```

**Gate:** `python -c "from opencood.qat import *"` clean; YAML round-trips through the existing `yaml_utils` loader without touching PTQ configs.

---

### QAT-0.2 — `CodebookQuantSTE` (`opencood/qat/ternary_quant.py`)

The core `autograd.Function`: STE for shadow weights, **exact** gradients for levels and scale (vanilla `round_ste` in `opencood/quant/quant_layer.py` cannot do this — its `.detach()` severs the level/scale path).

```python
class CodebookQuantSTE(torch.autograd.Function):
    """w_q = scale * levels[argmin_k |w/scale - levels_k|]  (non-uniform snap)"""

    @staticmethod
    def forward(ctx, weight, levels, scale, clip_grad=True):
        # weight: FP32 shadow [Cout, ...]; levels: [K]; scale: broadcastable, > 0
        w_n = weight / scale
        idx = torch.argmin(
            (w_n.unsqueeze(-1) - levels.view(*([1] * w_n.dim()), -1)).abs(), dim=-1)
        w_q = scale * levels[idx]
        ctx.save_for_backward(w_n, levels, scale, idx)
        ctx.clip_grad = clip_grad
        return w_q

    @staticmethod
    def backward(ctx, g):
        w_n, levels, scale, idx = ctx.saved_tensors
        # (a) STE → shadow weights; clipped outside representable range (LSQ/PACT style)
        grad_w = g.clone()
        if ctx.clip_grad:
            lo, hi = levels.min() * 1.5, levels.max() * 1.5
            grad_w = grad_w * ((w_n >= lo) & (w_n <= hi)).to(g.dtype)
        # (b) EXACT → levels: dw_q/dlevels[k] = scale where idx==k (VQ-VAE-style scatter)
        flat_g = (g * scale.expand_as(g)).reshape(-1)
        grad_levels = torch.zeros_like(levels).scatter_add_(0, idx.reshape(-1), flat_g)
        # (c) EXACT → scale: dw_q/dscale = levels[idx], reduced over broadcast dims
        g_scale = g * levels[idx]
        reduce_dims = [d for d in range(g.dim()) if scale.shape[d] == 1]
        grad_scale = g_scale.sum(dim=reduce_dims, keepdim=True)
        return grad_w, grad_levels, grad_scale, None
```

**Gates (`tests/qat/test_ste_function.py`):**

1. `torch.autograd.gradcheck` in float64 **w.r.t. `levels` and `scale` only** (weight `requires_grad=False`). Rationale: those paths are exact and must pass; the weight path is an STE whose analytic grad deliberately differs from the true (a.e.-zero) derivative, so gradcheck on it is meaningless and will "fail" by design. Inputs: `weight = torch.randn(4,3,3,3, dtype=torch.float64)*0.1`, `levels = torch.tensor([-1.,0.,1.], dtype=torch.float64, requires_grad=True)`, `scale = torch.full((4,1,1,1), 0.07, dtype=torch.float64, requires_grad=True)`. Nudge weights off codeword-boundary midpoints before gradcheck (argmin is discontinuous at ties).
2. STE identity check: with `clip_grad=False`, `grad_w == g` exactly; with `clip_grad=True`, gradient is zero exactly where `|w_n| > 1.5·max|level|`.
3. Assignment correctness: hand-built 5-element weight tensor with known nearest codewords; assert `idx` matches.
4. `grad_levels` accounting: sum of `grad_levels` equals sum of `g·scale` (no mass lost by scatter).

---

### QAT-0.3 — `TernaryCodebookQuantizer` + `ActFakeQuant` (`ternary_quant.py`, continued)

```python
class TernaryCodebookQuantizer(nn.Module):
    def __init__(self, n_levels=3, channel_wise=True, learn_levels=True, clip_grad=True):
        super().__init__()
        assert n_levels in (3, 4)
        init = torch.tensor([-1., 0., 1.]) if n_levels == 3 \
               else torch.tensor([-1.5, -0.5, 0.5, 1.5])
        self.levels = nn.Parameter(init, requires_grad=learn_levels)
        self.scale = None                     # created by init_scale — EAGERLY, at wrap time
        self.channel_wise, self.clip_grad, self.inited = channel_wise, clip_grad, False

    @torch.no_grad()
    def init_scale(self, weight, channel_dim=0):
        # BitNet b1.58 absmean rule, per output channel; channel_dim parameterized
        # because spconv weight layouts put Cout in a different position (see QAT-0.5)
        if self.channel_wise:
            dims = [d for d in range(weight.dim()) if d != channel_dim]
            s = weight.abs().mean(dim=dims, keepdim=True)
        else:
            s = weight.abs().mean().view(*([1] * weight.dim()))
        self.scale = nn.Parameter(s.clamp_min(1e-8))
        self.inited = True

    def forward(self, weight):
        assert self.inited, "init_scale must run before the optimizer is built"
        return CodebookQuantSTE.apply(weight, self.levels,
                                      self.scale.clamp_min(1e-8), self.clip_grad)
```

`ActFakeQuant(n_bits, mode)` — uniform activation fake-quant; `mode='ema'` (running min/max, plain `round_ste`) for Epic 3 bring-up, `mode='lsq'` (learnable step with 1/√(N·Q_max) gradient scaling, MQBench semantics) for final runs. Include a `qdrop_prob` argument: with probability `1−p` per-forward, bypass quantization (QDrop). API-compatible with `UniformAffineQuantizer(leaf_param=True)` so `set_act_quantize_params`-style MSE init can be reused.

**Gates:** absmean init reproduces hand-computed per-channel means; `extra_repr` reports effective bits; quantizer `state_dict` save/load round-trip preserves `scale`/`levels` exactly.

---

### QAT-0.4 — `QATQuantModule` (`opencood/qat/qat_module.py`)

Dense wrapper: `Conv2d` / `ConvTranspose2d` / `Linear`. API deliberately mirrors `opencood.quant.quant_layer.QuantModule` (`org_module`, `fwd_func`, `fwd_kwargs`, `set_quant_state`) so `QuantModel`-style surgery and the recon drivers transfer.

```python
class QATQuantModule(nn.Module):
    def __init__(self, org_module, n_levels=3, act_bits=None, learn_levels=True):
        super().__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.ConvTranspose2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   output_padding=org_module.output_padding,
                                   groups=org_module.groups, dilation=org_module.dilation)
            self.fwd_func = F.conv_transpose2d
        else:
            self.fwd_kwargs, self.fwd_func = dict(), F.linear

        # FP32 SHADOW WEIGHTS — the ONLY master copy; the only thing the optimizer sees.
        self.weight = nn.Parameter(org_module.weight.detach().clone())
        self.bias = (nn.Parameter(org_module.bias.detach().clone())
                     if org_module.bias is not None else None)

        self.weight_quantizer = TernaryCodebookQuantizer(n_levels, learn_levels=learn_levels)
        # ConvTranspose2d weight is [Cin, Cout, kH, kW] → channel_dim=1
        self.weight_quantizer.init_scale(
            self.weight, channel_dim=1 if self.fwd_func is F.conv_transpose2d else 0)
        self.act_quantizer = ActFakeQuant(act_bits) if act_bits else None
        self.use_weight_quant, self.use_act_quant = True, act_bits is not None

    def forward(self, x):
        # Re-quantize EVERY forward; w_q is a temporary, never stored as state.
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        out = self.fwd_func(x, w, self.bias, **self.fwd_kwargs)
        return self.act_quantizer(out) if (self.use_act_quant and self.act_quantizer) else out

    def set_quant_state(self, weight_quant=True, act_quant=True):
        self.use_weight_quant, self.use_act_quant = weight_quant, act_quant

    @torch.no_grad()
    def export_codes(self):
        """Deployment/telemetry: (int8 codes, levels[K], per-channel scale)."""
        q = self.weight_quantizer
        w_n = self.weight / q.scale.clamp_min(1e-8)
        idx = torch.argmin(
            (w_n.unsqueeze(-1) - q.levels.view(*([1] * w_n.dim()), -1)).abs(), dim=-1)
        return idx.to(torch.int8), q.levels.detach().clone(), q.scale.detach().clone()
```

**Training-loop contract (documented in the module docstring):** `optimizer.step()` nudges FP32 shadow weights by full-precision increments; single steps rarely flip a codeword — *accumulation* flips them, which is exactly why the FP master must never be overwritten by its quantized image.

**Gates (`tests/qat/test_qat_module.py`):**

1. FP-bypass equivalence: `set_quant_state(False, False)` reproduces the wrapped module's output **bit-exactly** on random input (all three layer types).
2. Gradient reach: one backward → `weight.grad`, `weight_quantizer.levels.grad`, `weight_quantizer.scale.grad` all non-None and non-zero.
3. 10-step overfit: tiny conv net, MSE to random target, loss strictly decreases with quant enabled.
4. `state_dict` round-trip: save → load into fresh wrapper → identical forward output.
5. ConvTranspose2d scale shape: `scale.shape[1] == Cout`, all other dims 1.

---

### QAT-0.5 — `QATQuantSpconvModule` (`opencood/qat/qat_spconv.py`) — the autograd-safe sparse wrapper

Ports the applied `QuantSpconvModule` fix. The two invariants, encoded as tests: **(1)** no `nn.Parameter(...)` construction anywhere in `forward` (wrapping a graph tensor in `nn.Parameter` re-leafs it and severs `grad_fn`); **(2)** the spconv op's `weight` slot is de-registered from `_parameters` **once at init**, then receives the quantized tensor as a plain attribute each forward, preserving `grad_fn`.

```python
class QATQuantSpconvModule(nn.Module):
    """Wraps spconv SubMConv3d / SparseConv3d / SparseInverseConv3d for ternary QAT."""

    def __init__(self, org_module, n_levels=3, act_bits=None, learn_levels=True):
        super().__init__()
        self.op = org_module                       # keep spconv op for indice/algo machinery
        # 1) Take ownership of the FP32 master copy.
        self.weight = nn.Parameter(self.op.weight.detach().clone())
        self.bias = (nn.Parameter(self.op.bias.detach().clone())
                     if self.op.bias is not None else None)
        # 2) De-register from the spconv module ONCE — init-time, NEVER in forward.
        #    After this, `self.op.weight` is a plain attribute slot.
        del self.op._parameters['weight']
        if 'bias' in self.op._parameters and self.op._parameters['bias'] is not None:
            del self.op._parameters['bias']

        # 3) Locate Cout in the spconv weight layout at runtime (spconv 2.x is
        #    typically [Cout, k0, k1, k2, Cin]; do NOT hard-code — assert instead).
        shape = list(self.weight.shape)
        self.channel_dim = shape.index(self.op.out_channels)
        assert shape.count(self.op.out_channels) == 1 or self.channel_dim == 0, \
            f"ambiguous Cout in spconv weight shape {shape}; pin channel_dim in config"

        self.weight_quantizer = TernaryCodebookQuantizer(n_levels, learn_levels=learn_levels)
        self.weight_quantizer.init_scale(self.weight, channel_dim=self.channel_dim)
        self.act_quantizer = ActFakeQuant(act_bits) if act_bits else None
        self.use_weight_quant, self.use_act_quant = True, act_bits is not None

    def forward(self, x):                          # x: spconv.SparseConvTensor
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        # AUTOGRAD-SAFE ASSIGNMENT: plain tensor attribute — w keeps its grad_fn
        # (CodebookQuantSTEBackward), so backward flows through spconv's implicit
        # GEMM into shadow weight, levels, and scale. object.__setattr__ bypasses
        # nn.Module.__setattr__, which would otherwise reject/re-register a tensor
        # into _parameters.
        object.__setattr__(self.op, 'weight', w)
        if self.bias is not None:
            object.__setattr__(self.op, 'bias', self.bias)
        out = self.op(x)
        if self.use_act_quant and self.act_quantizer is not None:
            out = out.replace_feature(self.act_quantizer(out.features))
        return out

    set_quant_state = QATQuantModule.set_quant_state
    export_codes    = QATQuantModule.export_codes
```

**Gates (`tests/qat/test_qat_spconv.py`) — the hard prerequisite gate for everything downstream:**

1. Regression guard: inside a hooked forward, assert `self.op.weight.grad_fn is not None` when weight quant is on (this is the exact failure mode the PTQ fix removed).
2. End-to-end sparse backward: build a toy `SparseConvTensor` (random voxels), run wrapped `SubMConv3d` → sum → `backward()`; assert `module.weight.grad.abs().sum() > 0` **and** `levels.grad`/`scale.grad` non-zero.
3. Single-copy invariant: `sum(p.numel() for p in module.parameters())` counts the shadow weight exactly once; `'op.weight' not in dict(module.named_parameters())`.
4. FP-bypass equivalence vs. unwrapped spconv op, bit-exact (indices and features).
5. Optimizer-visibility: `AdamW(module.parameters())` → one step → shadow weight changed, `op`'s attribute refreshed next forward (no stale quantized weight).
6. Layout assert fires correctly on a mocked ambiguous shape.

---

### QAT-0.6 — `QATQuantModel` graph surgery (`opencood/qat/qat_model.py`)

Copy `QuantModel.quant_module_refactor` semantics; reuse the `opencood_specials` registry pattern (`QuantPyramidFusion`, `QuantResNetBEVBackbone`, `QuantPointPillar`, `QuantSECOND`, `QuantLiftSplatShoot`, `QuantDownsampleConv`) with QAT wrappers substituted. **No FX tracing** — module-swap only (the heterogeneous forward with `eval(f"self.encoder_{modality}")`, dict inputs, and `record_len` control flow is not symbolically traceable). **No BN folding** (`is_fusing=False` path): BN stays live through training; folding happens only in Epic 3 export.

```python
class QATQuantModel(nn.Module):
    def __init__(self, model, qat_cfg):
        super().__init__()
        self.model, self.cfg = model, qat_cfg
        self.refactor(self.model, prefix='')

    def refactor(self, module, prefix):
        for name, child in module.named_children():
            full = f"{prefix}{name}"
            if any(full.startswith(s) for s in self.cfg['skip_names']):
                continue                                       # aligners stay FP
            n_levels, act_bits = self.resolve_policy(full)     # high-precision list → W8 uniform path
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                setattr(module, name, QATQuantModule(child, n_levels, act_bits, ...))
            elif is_spconv(child):
                setattr(module, name, QATQuantSpconvModule(child, n_levels, act_bits, ...))
            else:
                self.refactor(child, full + '.')

    def set_quant_state(self, w=True, a=True):
        for m in self.model.modules():
            if isinstance(m, (QATQuantModule, QATQuantSpconvModule)):
                m.set_quant_state(w, a)

    def qat_param_groups(self, lr_w=2e-5, lr_q=1e-4):
        shadow, qparams = [], []
        for m in self.model.modules():
            if isinstance(m, (QATQuantModule, QATQuantSpconvModule)):
                shadow.append(m.weight)
                if m.bias is not None: shadow.append(m.bias)
                qparams.append(m.weight_quantizer.scale)
                if m.weight_quantizer.levels.requires_grad:
                    qparams.append(m.weight_quantizer.levels)
        # weight_decay MUST be 0 on scales/levels — decay drags levels toward 0 → ternary collapse
        return [{'params': shadow,  'lr': lr_w, 'weight_decay': 0.0},
                {'params': qparams, 'lr': lr_q, 'weight_decay': 0.0}]
```

`resolve_policy(full_name)`: config-driven — `first_conv_high_precision` keeps each encoder's first conv at W8 (uniform 8-bit quantizer or FP); `high_precision_names` (shrink_conv, heads) likewise; everything else gets `n_levels` from `qat.weight`.

**Gates:** conversion coverage report (printed table: module name → wrapper type / skipped / high-precision) matches the PTQ skip-list expectations; full-model FP-bypass produces AP identical to the FP checkpoint on 10 V2X-Real frames; checkpoint save/load round-trip; conversion of every supported modality config (PointPillar, SECOND, LSS) completes without unwrapped Conv/Linear leaks (assert via `named_modules` sweep).

---

### QAT-0.7 — Telemetry (`opencood/qat/telemetry.py`) + Epic-0 exit gate

```python
class QATTelemetry:
    """Attach to a QATQuantModel; call .step() every N iters; logs to tensorboard/wandb."""
    def __init__(self, qat_model, every=100):
        self.mods = {n: m for n, m in qat_model.model.named_modules()
                     if isinstance(m, (QATQuantModule, QATQuantSpconvModule))}
        self.prev_codes = {n: m.export_codes()[0] for n, m in self.mods.items()}

    @torch.no_grad()
    def step(self, global_iter):
        for n, m in self.mods.items():
            codes, levels, scale = m.export_codes()
            flip = (codes != self.prev_codes[n]).float().mean().item()   # FLIP RATE
            self.prev_codes[n] = codes
            zero_frac = (codes == int(levels.abs().argmin())).float().mean().item()
            log({f"{n}/flip_rate": flip,                 # healthy: 1e-3 .. 1e-2
                 f"{n}/zero_frac": zero_frac,            # collapse alarm: >0.98 or <0.02
                 f"{n}/levels": levels.tolist(),
                 f"{n}/scale_mean": scale.mean().item(),
                 f"{n}/shadow_grad_norm": (m.weight.grad.norm().item()
                                           if m.weight.grad is not None else -1.0)})
```

**Epic-0 exit gate (blocking):** all of tests QAT-0.2 → 0.6 green in CI under the repo's pinned torch 1.12 env (no post-1.12 APIs anywhere); telemetry produces a sane report on a 3-layer toy model trained 50 steps (flip rate > 0, loss down, shadow dtype float32, `weight.unique().numel() > n_levels`).

---

## EPIC 1 — Shadow-Weight Init + Blockwise Warm Start (weeks 3–4) — Block-AP analogue

### QAT-1.1 — Init pipeline (`opencood/tools/train_qat.py`, init path)

**Sequence (one function, `build_qat_model(hypes, ckpt_path)`):**

1. Build FP model via existing `train_utils.create_model` + checkpoint loader.
2. Wrap: `qat_model = QATQuantModel(model, hypes['qat'])` — shadow weights cloned from FP, absmean scales computed, levels at `{-1,0,+1}`.
3. **Only then** construct the optimizer from `qat_param_groups()` (scale Parameters must exist first — eager init makes this safe, but assert anyway).
4. Snapshot pre-training codes for telemetry baseline.

**Gate:** initial ternary snap (quant on, no training) evaluated on 50 frames logs the "PTQ-equivalent-at-1.58b" AP — this is the floor every later phase must beat, recorded in the experiment tracker.

### QAT-1.2 — `opencood/qat/qat_recon.py` — subsystem-wise reconstruction with trainable weights

Reuse the PTQ recon drivers' I/O caching verbatim (`encoder_recon`, `second_recon`, `lss_recon`, `block_recon`, `pyramid_recon` already solve multi-input caching under warping, `record_len` batching, modality routing). The delta from PTQ: the optimized set becomes `{shadow weights, scales, levels}` instead of `{AdaRound α, act δ}`; loss unchanged.

```python
def qat_block_reconstruction(qat_block, cached_inps, cached_fp_outs,
                             iters=5000, batch_size=8, lr_w=2e-5, lr_q=1e-4):
    qat_block.set_quant_state(True, False)          # weights ternary, acts FP in Epic 1
    groups = collect_param_groups(qat_block, lr_w, lr_q)   # same split as qat_param_groups
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for it in range(iters):
        idx = torch.randint(len(cached_inps), (batch_size,))
        out = qat_block(*fetch(cached_inps, idx))   # multi-input safe (pyramid case)
        loss = F.mse_loss(out, fetch(cached_fp_outs, idx))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flatten(groups), 1.0)
        opt.step(); sched.step()
```

Driver order (architectural dependency order, ~1–2k calibration frames): `encoder_m*` → `backbone_m*` → `pyramid_backbone` → (heads stay W8, skipped). Codebook untouched (frozen, FP inputs assumption still holds within calibration tolerance).

**Gates:**

1. Per-block reconstruction MSE ≤ 0.3× the initial-snap MSE for every block (log a table).
2. Non-zero shadow-weight grads in *every* wrapped layer of the block (catch dead sub-graphs, esp. inside spconv middle encoder).
3. Flip-rate over the warm start between 0.1% and 5% per 100 iters; shadow histogram develops visible tri-modal clustering (log histograms).
4. Post-warm-start full-model eval beats the QAT-1.1 floor by a recorded margin.

---

## EPIC 2 — End-to-End Ternary QAT, Weights Only (weeks 5–6)

### QAT-2.1 — Freeze/skip policy (`opencood/qat/policy.py`)

`apply_qat_policy(qat_model, hypes)` — single authority for what trains when:

```python
PHASE_E2E_WEIGHTS = dict(
    shadow=True, scales=True, levels=True,
    act_quant=False, bn_live=True,
    codebook=False,                      # UMGM frozen: requires_grad=False on all UMGMQuantizer params
    fp_islands=('aligner_m',),           # never wrapped
    w8_islands=('shrink_conv', 'cls_head', 'reg_head', 'dir_head', 'first_conv'),
)
```

**Gate:** `assert_policy(qat_model, PHASE_E2E_WEIGHTS)` — walks named_parameters and verifies every `requires_grad` flag matches the declared phase; run at the top of every training script (cheap insurance against silent unfreezing).

### QAT-2.2 — Distillation (`opencood/qat/distill.py`)

```python
class QATDistiller(nn.Module):
    def __init__(self, teacher, w_kl=1.0, w_reg=1.0, w_feat=0.5, T=2.0):
        super().__init__()
        self.teacher = teacher.eval()            # FP checkpoint, same architecture
        for p in self.teacher.parameters(): p.requires_grad_(False)
        # forward hook on teacher.pyramid_backbone output → cached fused BEV feature

    def forward(self, student_out, student_fused_feat, batch):
        with torch.no_grad():
            t_out = self.teacher(batch)          # hook captures t_fused_feat
        L  = self.w_kl  * T*T * F.kl_div(F.log_softmax(student_out['cls']/T, 1),
                                         F.softmax(t_out['cls']/T, 1), reduction='batchmean')
        L += self.w_reg * F.smooth_l1_loss(student_out['reg'], t_out['reg'])
        L += self.w_feat * F.mse_loss(student_fused_feat, self.t_fused_feat)
        return L
```

Student fused feature exposed via the same hook mechanism on the QAT model's `pyramid_backbone` (fusion is where multi-agent quantization errors meet — that is why the mimic loss sits there).

**Gate:** unit test with teacher == student (FP-bypass): distill loss ≈ 0 (< 1e-6); hook fires exactly once per forward.

### QAT-2.3 — Training script (`opencood/tools/train_qat.py`, main loop)

Total loss: `L_det (existing opencood/loss: cls+reg+dir+occ) + distill`. AdamW; shadow lr 1–2e-5, quant-param lr ~1e-4; cosine decay; grad-clip 1.0; 10–15 epochs V2X-Real; telemetry every 100 iters; eval + checkpoint every epoch. Checkpoints save shadow weights + scales + levels (discrete codes are an export artifact, never training state).

**Gates (Epic 2 exit):**

1. Flip rate in the healthy band (≈0.1–1%); runbook wired to telemetry: >5% → halve lr_w; ≈0 for 500 iters → raise lr_w or verify levels unfrozen.
2. Levels drift asymmetric (`w_p ≠ w_n` per layer) — evidence TTQ freedom is being used.
3. No layer with zero_frac > 0.98 (ternary collapse); offender list auto-promoted to W8 via config override.
4. W1.58/A-FP AP@0.5 on V2X-Real within a pre-registered gap of FP baseline (set target after QAT-1.2 numbers land); must beat PTQ-W4 baseline.

---

## EPIC 3 — Activation Quantization (weeks 7–8)

### QAT-3.1 — Progressive activation enable

Modify: `policy.py` (phase `A8 → A6 → A4`), `train_qat.py` (resume-and-continue). Init act ranges with the existing MSE search (`set_act_quantize_params` machinery) on 64 calibration batches before unfreezing; then EMA or LSQ per config. QDrop `prob=0.5` at enable, annealed to 1.0 over 2 epochs.

**Gates:** A8 enable costs < 0.5 AP immediately after range init (else ranges mis-initialized); each precision drop retrains to within its own pre-registered budget before the next drop.

### QAT-3.2 — BN protocol: live → freeze → fold

- Live BN through all training phases (fold-then-train is unstable at ternary: folded scale absorbs γ/σ which then drifts).
- Last 2 epochs of each stage: freeze BN running stats (`momentum=0`, `eval()` on BN only).
- Export: fold via a QAT-aware port of `fold_bn.py` **into the shadow weights**, then re-snap codes; validate folded-vs-unfolded output equivalence < 1e-4 relative on 20 frames.

**Files:** `opencood/qat/export.py::fold_bn_qat`, test `tests/qat/test_fold_export.py`.

---

## EPIC 4 — Joint Codebook Co-Training + E2E-QP Analogue (weeks 9–10)

### QAT-4.1 — UMGM joint phase

**Modify:** `train_qat.py` (phase flag), `policy.py`. Unfreeze `opencood/models/sub_modules/codebook.py::UMGMQuantizer` params (stage-2/3 machinery already isolates them). Loss: `L_det + λ1·codebook MSE + λ2·usage-entropy penalty + λ3·distill`. Gumbel τ annealed 1.0 → 0.3; `reAssignCodebook` every `reassign_every` iters (utility exists, EMA usage frequencies); optional final swap to hard nearest-neighbor + STE for train/infer consistency. Weight-side and feature-side codebooks share entropy/dead-code utilities from `codebook_utils.py`.

**Gates:** code-usage perplexity per segment ≥ 0.5·k after co-training (no collapse under ternary features); AP not degraded vs. Epic 3 exit; message reconstruction MSE improves vs. frozen-codebook ablation (run both — this is also a paper ablation).

### QAT-4.2 — E2E-QP analogue (final polish)

```python
def freeze_for_e2eqp(qat_model):
    for m in qat_model.model.modules():
        if isinstance(m, (QATQuantModule, QATQuantSpconvModule)):
            m.weight.requires_grad_(False)        # ternary assignments locked
            m.weight_quantizer.scale.requires_grad_(True)
            m.weight_quantizer.levels.requires_grad_(True)
```

Train `{scales, levels, act ranges, codebook, BN affine}` end-to-end, 2–3 epochs, low lr.

**Gates:** flip rate == 0 exactly (weights frozen ⇒ codes cannot move; asserts the freeze is real); AP ≥ Epic-4.1 exit AP; this checkpoint is the release candidate.

---

## EPIC 5 — Evaluation, Scaling Law, Export (weeks 11–12)

### QAT-5.1 — Export (`opencood/qat/export.py`)

- `export_model(qat_model, path)`: fold BN → `export_codes()` per layer → pack. Ternary: 2 bits/weight trivially, or true 1.6 bits via base-3 packing (5 trits/byte, 3⁵ = 243 ≤ 256):

```python
def pack_trits(codes):                    # codes ∈ {0,1,2}, int8
    flat = codes.flatten().to(torch.int64)
    pad = (-flat.numel()) % 5
    flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int64)])
    g = flat.view(-1, 5)
    packed = (g * torch.tensor([1, 3, 9, 27, 81])).sum(1).to(torch.uint8)
    return packed, codes.shape            # unpack_trits inverts via divmod chain
```

- Manifest JSON: per-layer {shape, levels, scale dtype/shape, packing, bits}; codebook indices path for transmission.

**Gates:** pack → unpack → codes bit-exact; exported-model reload reproduces eval AP; reported storage bytes match `Σ ceil(numel/5) + scales + levels` within 1%.

### QAT-5.2 — Evaluation matrix (`opencood/tools/eval_qat_matrix.py`)

Rows: FP baseline, PTQ-W8A8, PTQ-W4A8, QAT-W2A8, QAT-W1.58A8, QAT-W1.58A4. Columns: AP@0.5/0.7 (V2X-Real, then OPV2V-H, DAIR-V2X-C), weight memory (MB), message size (bytes/frame at configured m·log2 k), latency. One YAML per row under `opencood/hypes_yaml/qat_sweeps/`.

### QAT-5.3 — Scaling-law sweeps (headline experiment)

- **Iso-memory:** {1× width @ INT8} vs {2× width @ 2-bit} vs {2.5× width @ 1.58-bit} — width multiplier as a hypes param on backbone channels.
- **Iso-bandwidth:** codebook (m, k) grid at fixed `m·log2(k)` bits/cell.
- Deliverable: AP vs weight-memory and AP vs bits-on-air curves + CSVs.

**Gates:** every point trained with the identical recipe (Epics 1–4 scripts, config-only differences); seeds fixed; curves reproduce within ±0.3 AP on re-run of one point.

### QAT-5.4 (stretch) — Deployment path

TensorRT via existing `build_trt_int8.py` for A8; custom bitpacked GEMM/im2col kernel for ternary weights.

---

## Cross-cutting risk register → owning task

| Risk | Owning task | Mitigation encoded |
|---|---|---|
| spconv autograd regression | QAT-0.5 gate 1 | `grad_fn` assert in CI, forever |
| Lazy scale init after optimizer build | QAT-0.4/1.1 | eager `init_scale` + assert in `build_qat_model` |
| Ternary collapse of small layers | QAT-2.3 gate 3 | zero_frac alarm → W8 promotion |
| BN drift under folding | QAT-3.2 | live→freeze→fold protocol + equivalence test |
| Codebook usage collapse | QAT-4.1 | entropy penalty + reassign cadence + perplexity gate |
| weight_decay on quant params | QAT-0.6 | hard-coded 0.0 in `qat_param_groups` |
| torch 1.12 pin | Epic-0 exit gate | CI runs pinned env; no post-1.12 APIs |
