# Handoff Prompt — paste this as the first message of a new chat

(Everything between the lines is the prompt. Works standalone; the project
files it references are all in the repo.)

---

You are continuing an in-progress engineering effort on the QuantV2X repo
(cooperative V2X perception, OpenCOOD lineage): transitioning from PTQ
(`opencood/quant/`, BRECQ/QDrop-style — DO NOT MODIFY, it is the frozen
baseline) to 1.58/2-bit Quantization-Aware Training.

## Current state (Epic 0 of 6 implemented, by a prior session)

Implemented package `opencood/qat/`:
- `ternary_quant.py` — `CodebookQuantSTE` (autograd.Function: STE to shadow
  weights, EXACT gradients to codebook levels & per-channel scales),
  `TernaryCodebookQuantizer` (K=3 ternary / K=4 / generic-K uniform for
  high-precision islands; absmean per-channel scale init),
  `ActFakeQuant` (EMA/LSQ uniform activation fake-quant with QDrop).
- `qat_module.py` — `QATQuantModule` wraps Conv2d/ConvTranspose2d/Linear;
  FP32 shadow weights are the ONLY master copy, re-quantized every forward.
- `qat_spconv.py` — `QATQuantSpconvModule`: spconv wrapper. CRITICAL
  invariant: never construct `nn.Parameter` in forward; the op's weight is
  de-registered from `_parameters` once at `__init__` and the quantized
  tensor is assigned as a PLAIN attribute each forward so `grad_fn`
  survives (a static test greps forward's source to enforce this).
- `qat_model.py` — `QATQuantModel`: recursive module-swap surgery (NO
  torch.fx — the model's forward is not traceable), skip-list FP islands
  (aligners, codebook), high-precision islands (heads, first encoder
  convs, K=255), `qat_param_groups()` (shadow lr 2e-5 / quant-params lr
  1e-4, weight_decay hard-coded 0.0 — decay collapses ternary codebooks),
  `coverage_report()`.
- `qat_config.py` — `qat:` YAML section parser (defaults/validation;
  `yaml_utils.py` untouched). `telemetry.py` — `QATTelemetry` (flip rate,
  zero-frac collapse alarm, severed-grad tripwire).

Tests: `tests/qat/` (~50 tests + `numpy_reference_check.py`, torch-free
math oracle — already passed). Run with `python -m pytest tests/qat -v`
from repo root; sparse tests use a CPU mock of spconv from `conftest.py`.

Reference docs (read before writing code):
- `docs/qat_transition/QAT_BACKLOG.md` — the full Epic 0–5 task backlog
  with specs and verification gates. THIS IS THE SPEC; follow it.
- `docs/qat_transition/QAT_Architecture_Breakdown.md` — architecture and
  framework-survey rationale.
- `docs/qat_transition/LEARNING_PLAN.md` — the user's tutoring curriculum.

## Hard constraints
- torch 1.12 pin: no post-1.12 APIs.
- Never modify `opencood/quant/` (PTQ baseline) or `opencood/qat_mine/`
  (the user's own rewrite exercises — hands off).
- BN stays live during QAT; folding only at export.
- The user (Michelle) is learning this codebase: explain the "why" in code
  comments and in chat; prefer teaching over doing when she asks.
- Every new component needs tests in `tests/qat/` before it counts as done.

## Immediate next steps (in order)
1. If not yet done: have the user run `python -m pytest tests/qat -v` and
   fix any environment-specific failures (Epic 0 exit gate).
2. GPU-day runbook in LEARNING_PLAN.md: asset inventory (FP checkpoint?
   V2X-Real dataset?), real-model conversion + coverage report review,
   "initial snap" AP floor measurement.
3. Then Epic 1 per QAT_BACKLOG.md: `opencood/qat/qat_recon.py` — blockwise
   warm start reusing the PTQ recon drivers' I/O caching
   (`opencood/quant/encoder_recon.py`, `block_recon.py`,
   `pyramid_recon.py`), optimizing {shadow weights, scales, levels} against
   cached FP block outputs; then the `tools/train_qat.py` init path.

Start by reading QAT_BACKLOG.md and the current `opencood/qat/` source,
then confirm your understanding of the shadow-weight contract and the
spconv attachment invariant before writing anything.

---
