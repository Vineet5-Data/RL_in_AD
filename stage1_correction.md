# Stage-1 Correction — Run 0b Spec (post-de-leak eval failure)

> Supersedes the Run 0 definition in `stage2_correction.md`. Root-cause analysis of the de-leaked retrain's eval results (`eval_stage1.txt`), followed by coder-ready changes. Compute constraints unchanged (RTX 4060 8GB / Kaggle single GPU ≤12h per session and 30hr per week with T4x2 GPU or P100, 30GB RAM, 42GB SSD).

---

## 1. What happened

De-leaked retrain (`stage1_retrain_deleaked.log`) trained correctly — contrast 1.64 → ~0.7, full cosine decay, no instability, no leak signature. **The training run is not the failure. The task formulation is.**

Honest eval (`eval_stage1.txt`):

| eval | Hit@1 | random | leaked (dead) |
|---|---|---|---|
| Porto in-city | 0.738 | 0.174 | 0.920 |
| Porto held-out | 0.671 | 0.164 | 0.921 |
| T-Drive OOD | 0.345 | 0.157 | 0.821 |

Why this blocks Stage 2 mathematically: prior-rollout path accuracy compounds per step. 0.67^15 ≈ 0.002 — Stage-3 planning gates unreachable. Retrain with a better formulation is required before Stage-2 Run 1.

---

## 2. Root cause (verified, three independent lines)

### 2.1 The model is geometry-blind by construction

Full input audit:

| Input path | Contains position? |
|---|---|
| GPS side (`gps_encoder._gps_feats`) | yes — centroid-local nlat/nlon, bearing sin/cos, speed, dt |
| Candidate features (`gps_encoder.py:72`, post-7C `Linear(4,d)`) | **no** — only [sinθ, cosθ, oneway, log1p(speed_kph)] |
| Stage-0 road_z (`roadgraph/io.py:77-91`) | **no** — log_length, heading sin/cos, curvature, oneway, lanes, speed, bridge, tunnel, degrees, highway embedding. No x/y/lat/lon anywhere |

Target = `argmin d_perp` — a pure metric comparison at meter scale. No input carries candidate position, so GPS↔candidate proximity is **not computable from the inputs**. Change 7C removed the leak by removing the *only* geometric channel, leaving heading match + road-class priors + sequence continuity. That information set caps near the observed 0.67.

OOD collapse consistent: Beijing kills road_z embedding-identity continuity (zero-shot graph) and 60s sampling kills sequence continuity → only heading/class priors remain → 0.345.

### 2.2 The pointwise argmin target is noise-dominated (cache measurement, 300k rows)

```
d1 (nearest) m:    p25 1.3   p50 2.8   p75 5.5   p95 15.7
margin d2-d1 m:    p25 0.0   p50 2.8   p75 11.3
P(margin < 3m)  = 0.471      P(margin < 5m)  = 0.536
P(margin < 10m) = 0.669      P(margin < 15m) = 0.762
```

GPS noise σ ≈ 10m. For **67% of fixes the argmin label is decided by less than the noise magnitude**; 25%+ are exact ties (bidirectional segment pairs share identical d_perp — tie broken by cache sort order, i.e., an artifact). A hard one-hot argmin target is therefore a noisy, partly arbitrary label. Part of the 0.671 is the model being *penalized for disagreeing with label noise*.

### 2.3 What was NOT the cause (do not touch)

- Optimizer/LR/epochs/batch — training converged cleanly; contrast ~0.7 ≈ the information ceiling of the inputs.
- Eval scripts — 7A permutation, 7E segment-id Hit@1, strict ckpt load all verified correct.
- Changes 7A/7E themselves — keep permanently.

---

## 3. The fix — restore geometry as INPUT, kill circularity via masking, de-noise the TARGET

Principle: d_perp is a legitimate deployment-time observable (HMM emission feature — computed from map + GPS, no ground truth involved). The leak was never "d_perp as input"; it was "target = argmin of a visible input, scored on that same target." Fix the training signal, not the model's eyesight.

### Change 8 — candidate geometry restored + feature-dropout masking

`gps_encoder.py`:
- Candidate features → `[d_perp/50 (0 when masked), geo_mask_flag, sinθ, cosθ, oneway, log1p(speed_kph)]` → `cand_pos = nn.Linear(6, d)`.
- New forward arg `geo_mask: (B,T) bool`. Where True: zero the d_perp feature for ALL K candidates at that fix and set flag=1. Masking touches **inputs only** — target-side d_perp always read from cache.

`stage1.py`:
- Per batch sample `geo_mask = (rand(B,T) < 0.3) & seq_mask_valid`.
- InfoNCE loss on **all** valid fixes (masked + visible). Visible term goes trivially low fast (trains the deployment/copy path — intended); masked term carries the honest learning pressure (context + neighbor geometry → current road).
- Log split columns: `contrast_masked`, `contrast_visible`.

### Change 9 — soft emission targets (replaces hard one-hot argmin)

- Target distribution per fix: `softmax(-d_perp² / (2σ²))` over valid candidates, σ = 10m (GPS noise scale). Invalid slots masked out.
- Loss = CE against this soft distribution (5-line change from one-hot CE).
- Rationale: exactly the HMM emission likelihood. Handles the 25% ties (both directions of a two-way road get ~equal mass; heading input resolves direction) and the 47% sub-3m margins (near-equal mass instead of arbitrary hard label). This is also the distribution Stage-2's posterior actually wants from z1.
- Keep hard-argmin Hit@1 computable at eval for continuity with old logs — but never gate on it alone.

### Change 10 — eval protocol v2 (all three eval scripts)

Two passes per checkpoint:
1. **Visible pass** (`geo_mask` all False) — deployment metric. Expect near-ceiling; sanity check, not a learning metric.
2. **Masked pass** (seeded 30% mask, score ONLY masked fixes) — the honest context-learning metric. Mirrors training distribution.

New metric both passes: **tolerant Hit@1** — hit if chosen candidate's true d_perp ≤ d1 + 5m. Removes tie/label-noise penalty. Report matrix: {visible, masked} × {hard, tolerant} × {in-city, holdout, T-Drive}.

---

## 4. Run 0b plan

Everything else IDENTICAL to the de-leaked run: batch 16, 5 epochs, lr 3e-4 cosine, bf16, same cache (no rebuild — d_perp already stored), 7A permutation on, mgm/next losses unchanged.

| Gate (holdout unless noted) | Threshold | On fail |
|---|---|---|
| Visible tolerant Hit@1 | ≥ 0.95 | wiring bug (mask leaked into eval / feature misalignment) — abort, debug, do NOT change formulation |
| Masked tolerant Hit@1 | ≥ 0.70, target 0.75–0.85 | ablate: (i) hard targets + masked inputs, (ii) mask rate 15% / 50%; if still < 0.70 → trigger Run 0c |
| T-Drive visible tolerant | ≥ 0.70 | if still ~0.35 → geometry restore not reaching eval path OR Beijing cache issue — Phase-1 again |
| `contrast_visible` | → < 0.1 within 2k steps | if stays high: features misaligned |
| `contrast_masked` | settles 0.4–0.8, ≠ contrast_visible | if ≈ visible ≈ 0: mask flag not applied — abort |
| NaN | anywhere | abort |

### Stage-2 interactions

- Stage-2 consumes **visible** z1 (no masking at inference) — z1 now carries emission-distribution info.
- Posterior road CE → 0 will RETURN (z1 contains the answer again). Expected and benign — already excluded from gates; prior-road aux (stage2_correction Change 4) is the honest signal, prior cannot see future d_perp.
- Stage-2 road targets: adopt Change-9 soft targets + tolerant Hit@1 for Change-4/5 metrics.
- Stage-3 entry gates in `stage2_correction.md` (prior Hit@1@15 ≥ 0.5) were calibrated on leak-inflated numbers — **recalibrate from Run 0b's masked metric** (per-step p → path p^H; log per-horizon, set gates from measured p, don't invent).

## 5. Run 0c — contingent / thesis-final (do NOT start now)

FMM/HMM Viterbi pseudo-labels (sequence-consistent, resolves the 67% noisy-margin fixes via route continuity). Required for thesis eval harness anyway (methodology: Porto FMM pseudo-GT) — build the FMM harness in parallel on Kaggle/WSL, but retrain on FMM targets only if (a) Run 0b masked gate fails, or (b) final thesis re-baseline. Optional lever for OOD at same time: sampling-rate augmentation (random subsample to 30–60s) for T-Drive transfer. Keep OFF in Run 0b — one formulation change at a time.

## 6. Do-not-do list

- Do NOT revert 7A (permutation) or 7E (segment-id Hit@1).
- Do NOT feed raw d_perp without the mask machinery (that's the original leak).
- Do NOT gate anything on visible-pass hard Hit@1 (vanity metric — near-copyable).
- Do NOT touch optimizer/LR/epochs/batch this run.
- Do NOT rebuild the candidate cache.
- Do NOT start FMM retrain before Run 0b verdict.

---

## 7. Run 0b RESULTS (2026-07-03) — VERDICT: PASS. Formulation fixed, no Run 0c.

Final ckpt, Change-10 matrix (tolerant Hit@1):

| eval | visible tol | masked tol (honest) | masked hard |
|---|---|---|---|
| Porto in-city | 1.000 | 0.851 | 0.662 |
| Porto held-out | 1.000 | 0.823 | 0.613 |
| T-Drive OOD | 1.000 | 0.540 | 0.343 |

Gate check against §4: visible tolerant 1.000 ≥ 0.95 → wiring correct. Holdout masked tolerant 0.823 ≥ 0.70 → PASS, inside the 0.75–0.85 target band. T-Drive visible tolerant 1.000 ≥ 0.70 → PASS. **Run 0c NOT triggered** — FMM harness remains a thesis-eval deliverable only, not a retrain trigger.

Per-checkpoint masked-tolerant curve (gates apply to final ckpt only; step1000 is an early snapshot, not a gate subject):

| ckpt | in-city | holdout | T-Drive |
|---|---|---|---|
| step1000 | 0.582 | 0.535 | 0.460 |
| step15000 | 0.789 | 0.752 | 0.554 |
| final | 0.852 | 0.823 | 0.540 |

Curve evidence: honest metric monotone-earned over training (0.535→0.823 holdout) — a leaky/copyable metric would start high at step1000; it doesn't. In-city↔holdout gap stays ~3pts at final → no memorization. T-Drive masked dips 0.554→0.540 late (mild Porto specialization, ungated); step15000 would trade −7pts holdout for +1.4pts OOD — rejected, final ckpt selected.

Reading the two low-looking numbers (neither is a failure, neither is gated):
- **Masked hard 0.662/0.613** — the noise-dominated argmin metric. §2.2 predicts a hard ceiling near 0.75 from ties (25%) and sub-noise margins (67%); the tolerant→hard gap (0.851→0.662) is that tie/noise mass showing up exactly where expected. Diagnostic only.
- **T-Drive masked 0.540** — honest zero-shot context-learning number, no gate. Beijing graph is unseen and 60s sampling destroys the sequence continuity the masked path relies on; the deployment path (visible — d_perp always computable at inference) sits at 1.000. 0.540 vs 0.157 random = 3.4× honest OOD lift. Optional future lever: sampling-rate augmentation (§5). Not required for Stage 2.

**Per-step honest anchor for Stage-2/3 calibration: p = 0.823 (holdout masked tolerant); 0.851 in-city.** Independence floors: p^5 ≈ 0.38, p^8 ≈ 0.21, p^15 ≈ 0.054. Recurrent prior + route continuity should beat these floors — measure per-horizon in Stage-2 Run 1 and set final gates from the measured curve, don't invent.

**Next step: Stage-2 Run 1 per `stage2_correction.md`** (Changes 1–3, 5, 6, 7B) on this Stage-1 checkpoint; all Stage-2 road metrics adopt Change-9 soft targets + tolerant Hit@1; recalibrated gates written into `stage2_correction.md` (Gate 9 + Stage-3 entry).
