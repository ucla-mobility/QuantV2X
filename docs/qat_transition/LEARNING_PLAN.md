# QAT Learning Plan — Rewrite Epic 0 Yourself (Tutored Track)

Audience: Michelle. Cadence: ~1–2 hrs/day alongside deadline work.
Rule #1: **never delete or edit `opencood/qat/`** — it is the working
implementation AND your answer key. All your code goes in
`opencood/qat_mine/` (create it; same file names).
Rule #2: **the existing test suite is your oracle.** After each exercise:

```bash
# one-time: make a test copy pointed at YOUR package
cp -r tests/qat tests/qat_mine
# (Windows PowerShell) (Get-ChildItem tests/qat_mine/*.py) | ForEach-Object {
#   (Get-Content $_) -replace 'opencood\.qat', 'opencood.qat_mine' | Set-Content $_ }
# (bash) sed -i 's/opencood\.qat\b/opencood.qat_mine/g' tests/qat_mine/*.py
python -m pytest tests/qat_mine/<the relevant file> -v
```

When your rewrite passes the same gates as mine, you own the concept.

---

## Day 0 (~1 hr) — Ground truth first. No writing.

1. Run the gates against MY implementation (your conda env, CPU fine):

   ```bash
   python tests/qat/numpy_reference_check.py      # 12 [ok] lines
   python -m pytest tests/qat -v                  # expect ~50 pass, 1 skip
   ```

   If anything fails, stop and bring me the output — do not build on sand.

2. Reading order (30 min, skim — the point is the map, not mastery):
   `opencood/qat/__init__.py` docstring → `ternary_quant.py` module
   docstring (the math model, eq. (1)) → `qat_spconv.py` top docstring
   (the bug story) → `docs/qat_transition/QAT_BACKLOG.md` Epic 0.

3. Answer in one sentence each, in a notes file (self-test, no peeking):
   - Why must the optimizer never see the quantized weights?
   - Which two parameters get *exact* gradients and why is that possible?
   - What exactly does `nn.Parameter(w_q)` destroy?

---

## Phase 1 (Day 1, ~2 hrs) — The math, on paper then numpy

**Goal:** derive the quantizer before reading any PyTorch.

**Socratic ladder — work these in order, on paper:**
1. You must represent a weight tensor with 3 values per channel:
   `s·{−1, 0, +1}`. Given a weight w and scale s, which value should it map
   to? (Write the rule. You will invent `argmin_k |w/s − c_k|`.)
2. Sketch w_q as a function of w (fix s=1). What does the derivative look
   like? Where is it nonzero? Why is that fatal for gradient descent?
3. Now hold the *assignment* fixed. Is w_q differentiable in the level
   values c_k? In s? Write both derivatives. (You should get
   ∂w_q/∂c_k = s·1[assigned to k] and ∂w_q/∂s = c_assigned.)
4. Why is "hold the assignment fixed" legitimate? (Hint: for which weights
   does an infinitesimal change of c_k change the assignment? What measure
   does that set have?)

**Exercise 1.1:** implement `forward(w, levels, scale) -> (w_q, idx)` and
`backward(g, ...) -> (grad_levels, grad_scale)` in pure numpy, from your
paper derivation. Do NOT look at `numpy_reference_check.py` yet.

**Verify:** your functions against central finite differences (write the
finite-difference loop yourself — it is 10 lines and you will reuse the
skill forever). Then diff your approach against
`tests/qat/numpy_reference_check.py`.

**Checkpoint question:** in check #4 of that file, why must grad_scale use
the *level*, not the weight? (If you can answer this cold, Phase 1 is done.)

---

## Phase 2 (Day 2, ~2 hrs) — autograd.Function: the STE

**Goal:** rewrite `CodebookQuantSTE` from a blank file.

**Get-there ladder:**
1. Warm-up: write `round_ste` yourself from its spec ("forward: round,
   backward: identity"). Understand the trick:
   `(x.round() − x).detach() + x` — forward the two x's cancel; backward
   only the undetached x contributes.
2. Now try to use that trick when `levels` is also an input. Convince
   yourself it CANNOT deliver gradients to `levels` (everything inside
   `.detach()` is invisible to autograd). This failure is *why* the custom
   Function exists — feel the wall before you climb it.
3. Read the `torch.autograd.Function` API contract: `forward(ctx, *args)`,
   `ctx.save_for_backward`, `backward(ctx, grad_out)` returning one grad
   per forward input (None for non-tensors).
4. Write it: forward = your numpy forward in torch; backward = your numpy
   backward + the STE for `grad_w` (identity, then the 1.5× clip window).
   Use `torch.scatter_add_` where numpy used `np.add.at`.

**Verify:** `python -m pytest tests/qat_mine/test_ste_function.py -v`

**Trap you will likely hit** (by design): gradcheck failing because your
test weights sit near a cell boundary — reread `_safe_weights` and
understand why the exact-gradient claim is only "almost everywhere".

---

## Phase 3 (Day 3, ~1.5 hrs) — Quantizer + dense wrapper

**Goal:** rewrite `TernaryCodebookQuantizer` and `QATQuantModule`.

**Get-there ladder:**
1. Design question first: the scale's *shape* depends on the weight it will
   quantize. Where can it be created? What goes wrong if a Parameter is
   created *after* `torch.optim.AdamW(model.parameters())`? (Answer: it is
   silently never updated — the optimizer captured the param list already.)
   This single fact dictates the eager-`init_scale` architecture.
2. Second design question: `F.conv2d(x, w, b, ...)` takes the weight as an
   argument. Why does that make the dense wrapper EASY (no attribute
   surgery), and what property of spconv makes the sparse one HARD?
3. Write the quantizer (absmean per-channel init — derive the `dims=`
   argument for `channel_dim=0` vs `1` yourself), then the wrapper:
   capture fwd_func/fwd_kwargs from the org module, clone shadow weight,
   quantize-every-forward, FP bypass path.
4. ConvTranspose2d check: print `nn.ConvTranspose2d(4, 8, 2).weight.shape`.
   Which dim is the output channel? Now you know why `channel_dim=1`.

**Verify:** `python -m pytest tests/qat_mine/test_qat_module.py tests/qat_mine/test_quantizers.py -v`

---

## Phase 4 (Day 4, ~1.5 hrs) — The spconv trap (the most important day)

**Goal:** reproduce the bug with your own hands, then rewrite the fix.

**Micro-lab first (15 min, throwaway script):**

```python
import torch, torch.nn as nn
w = nn.Parameter(torch.randn(3))
wq = w * 2                       # any op: wq has grad_fn, is_leaf=False
p  = nn.Parameter(wq)            # "attach" it the buggy way
print(wq.grad_fn, wq.is_leaf)    # MulBackward0, False
print(p.grad_fn,  p.is_leaf)     # None, True   <- the graph is GONE
p.sum().backward()
print(w.grad)                    # None. Silent. No error. THE bug.
```

Then reproduce the *fix* path: a module `m` with a registered parameter,
`del m._parameters['weight']`, assign `m.weight = wq`, backward, observe
`w.grad` is populated. Now read `nn.Module.__setattr__` (torch source,
~30 lines) to see WHY the deletion is needed first.

**Exercise:** rewrite `QATQuantSpconvModule` against the mock in
`tests/qat/conftest.py` (read the mock — it documents spconv's contract:
weight read at call time, `out_channels`, `replace_feature`). Include the
channel-dim resolver; write down the failure it prevents before coding it.

**Verify:** `python -m pytest tests/qat_mine/test_qat_spconv.py -v`

---

## Phase 5 (Day 5, ~1.5 hrs) — Graph surgery + telemetry

**Get-there ladder:**
1. Explore: `for n, c in model.named_children(): print(n, type(c))` on the
   toy model in conftest. Then recurse. Where must the `setattr` happen —
   on the child or the parent? Why does `named_modules()` NOT work for
   swapping (you need the parent handle)?
2. Write the recursive swap with the three policies (skip prefix / high
   precision / first-encoder-conv). Keep my coverage-report idea: an
   unreviewable surgery is an untrustworthy one.
3. Telemetry: define flip rate precisely BEFORE coding (fraction of
   *codes* changed between snapshots — why codes and not shadow values?).
   Then write the tracker.

**Verify:** `python -m pytest tests/qat_mine/test_qat_model.py tests/qat_mine/test_telemetry.py -v`
Then the full suite: `python -m pytest tests/qat_mine -v` — all green means
you have independently reimplemented Epic 0.

---

## The actual pipeline — component map (what exists vs what's missing)

Runtime dataflow (per frame):

```
per-agent sensor → encoder_m* (PointPillar / SECOND-spconv / LSS)
                → backbone_m* (BEV CNN) → aligner_m* [FP island]
                → codebook (UMGM) → V2X channel → decode
                → spatial warp → pyramid_backbone (fusion)
                → shrink_conv → cls/reg/dir heads [HP islands]
```

| Pipeline stage | QAT status (today) |
|---|---|
| Quantizer math + wrappers + surgery + telemetry (Epic 0) | ✅ written; CPU tests pending YOUR run; GPU spconv test pending |
| FP-checkpoint → QAT conversion entry (`tools/train_qat.py` init path) | ❌ Epic 1 |
| Blockwise warm start (`opencood/qat/qat_recon.py`, reusing `opencood/quant/*_recon.py` caches) | ❌ Epic 1 |
| E2E training loop + FP-teacher distillation (`policy.py`, `distill.py`, main loop) | ❌ Epic 2 |
| Activation quant schedule (A8→A4) + BN freeze-then-fold | ❌ Epic 3 (ActFakeQuant itself exists) |
| UMGM codebook co-training + E2E-QP polish | ❌ Epic 4 |
| Export/bitpacking, eval matrix, scaling-law sweeps | ❌ Epic 5 (trit-pack math verified) |

**Answer to "are we done with pipeline making?": No — Epic 0 is the
foundation (~the hardest correctness work), but no training entry point
exists yet.** You can convert a model and backprop through it; you cannot
yet *run a QAT training job*. That is Epic 1–2, next.

---

## GPU-day runbook (when the box is back)

Step 0 — asset inventory (you said "not sure what's there"):

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import spconv; print(spconv.__version__)"
find . -name "*.pth" -o -name "*.ckpt" | head -20      # FP checkpoints?
grep -rn "root_dir" opencood/hypes_yaml/v2x_real/*.yaml | head -5
ls <that root_dir>                                     # dataset present?
```

Step 1 — re-run the gates on the GPU env:
`python -m pytest tests/qat -v` (the real-spconv test un-skips) and
`python docs/qat_transition/test_spconv_grad_fix.py` (the definitive
end-to-end CUDA check for the attachment fix).

Step 2 — first REAL result: convert an actual model and read the surgery.

```python
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.qat import QATQuantModel, QATTelemetry

hypes = yaml_utils.load_yaml("<a v2x_real config>", None)
model = train_utils.create_model(hypes)          # + load FP ckpt if found
qmodel = QATQuantModel(model, hypes.get("qat"))
qmodel.coverage_report(print)                    # <- READ every line
```

What to look for: aligners/codebook SKIPPED, encoder first convs `[first-conv
HP]`, heads K=255, everything else K=3, zero leaks.

Step 3 — the "initial snap" floor (needs checkpoint + dataset): run the
existing inference/eval script on `qmodel` with quant ON, no training.
This AP number is the PTQ-equivalent-at-1.58-bit floor — the baseline every
subsequent epic must beat. Log it somewhere permanent.

If there is **no FP checkpoint**: ask the PhD lead / check the repo's
release page before training one yourself — an FP V2X-Real baseline is
days of GPU time and almost certainly exists already.

Step 4 — one overfit batch through the real model (10 steps, one frame,
loss must drop; telemetry flip rate > 0). This is Epic 1's smoke test and
your green light to start `qat_recon.py`.

---

## Next steps summary

1. Today: Day 0 (run gates, guided read).
2. Daily 1–2 hr: Phases 1–5 in `opencood/qat_mine/`.
3. GPU day: runbook above; capture coverage report + snap-floor AP.
4. Then: Epic 1 (`qat_recon.py` — use `docs/qat_transition/QAT_BACKLOG.md`
   §EPIC 1 as the spec; a fresh chat can be bootstrapped with
   `docs/qat_transition/HANDOFF_PROMPT.md`).
