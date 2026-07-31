# Stage-2 Fix Proposal v2 — Post-29k-Run Corrections

> Self-contained analysis (v1 proposal retired; its validated math — free bits, KL balancing, unimix, eval gate — is restated in §2.1). Grounded in: direct execution of the trained `stage2_porto_step29000.pt` checkpoint against the real candidate cache, `stage2_full_run.log`, DreamerV3 (arXiv:2301.04104) paper verification, and `eval_stage2.py` code review. Compute constraints unchanged (RTX 4060 8GB / Kaggle T4-P100 ≤12h single GPU); every change below is parameter-count-trivial (≤ +2.2M params worst case).

---

## 1. Corrections to the Evidence Base

### 1.1 "Bug 1 — has_cand always False" is RETRACTED (evaluation_findings.md is wrong on this point)

Direct execution of the unmodified pipeline (real `porto_n200000_r50_k10.npz` cache — 98.7% of rows have ≥1 valid candidate — real Stage-0/1/2 checkpoints, real batches):

- `has_cand` mean = 0.55–0.65 per batch; `n_road` = 567–668 per batch — **never** floored to 1.
- Untrained RSSM on same data: road CE ≈ 2.31 ≈ ln(10) = random over K=10, matching the training log's step-0 value (2.2791).
- Trained checkpoint: road CE = 0.00000000 (exact at 8 decimals); log shows a genuine multi-hundred-step decay 2.28 → 0.105 (step 50) → ~0.0001 (step ~1000).

**The candidate data path is functionally correct end-to-end** (`LengthCapped` truncation, collate, `cand_pad` in `gps_encoder.py:120`, `has_cand` in `stage2.py:85`). The Coder agent must NOT spend time on this; no data-path fix is needed.

### 1.2 Actual road→0 mechanism (v1 §1.1 revised)

Road CE collapsed to exact 0.0 (not v1's cited ~0.06) via the posterior shortcut: `posterior(h, z1_t)` sees the same-timestep Stage-1 embedding whose contrastive target is the **identical quantity** as Stage-2's road label (argmin d_perp over the same K≈10 candidates). 44.4-nat latent vs ≤ln(10)≈2.3-nat task → lossless transmission after ~1k steps of dense gradient.

Framing changes from "benign" to: **expected, fully collapsed, and fully uninformative.** Practical prescription unchanged from v1 — keep the term (harmless), exclude it from all gating — but with higher confidence: posterior road CE is structurally blind to prior quality, so a prior-side road signal is now mandatory (§2.3).

### 1.3 GPS plateau is a decode-precision ceiling, not KL collapse and not capacity shortage

- KL at plateau = 0.60–0.74 nats — healthy-ish, not collapsed → free bits will NOT break the plateau (it remains cheap insurance).
- Raw capacity check: absolute position @50m inside Porto bbox needs ~14.6 bits; 16×16 latent has 64 bits. Capacity is not the binding constraint.
- Binding constraint: straight-through hard categorical samples are poor carriers of fine continuous residual — positions straddling a class boundary get discontinuous codes; within-class offsets get zero gradient. Same limitation documented in VQ-VAE/codebook literature; the original World Models MDN-RNN (arXiv:1803.10122) used a continuous mixture head for exactly this reason.
- ol@4 losing to constant-velocity (0.364 vs 0.305 km) while ol@8/16 win is a **symptom of the same ceiling**: RSSM decode error is roughly horizon-constant (~decode floor), cv error compounds super-linearly with horizon. One fix, not two.

---

## 2. Method (changes on top of v1 §2)

### 2.1 Keep from v1 verbatim (validated against DreamerV3 paper, arXiv:2301.04104 Eq. 4/5, Table W.1)

Free bits 1.0 nat on both KL terms; β_dyn = 0.5, β_rep = 0.1; unimix 1% on BOTH prior and posterior; LR 1e-4 + 1k warmup + grad clip 100 (current code: balancing 0.8/0.2 exists at `stage2.py:130`, free bits absent, lr 3e-4 — all three change to v1 spec); KL healthy band 2–6 nats; stop-loss list v1 §4.

### 2.2 PRIMARY fix — continuous residual GPS head (Option C)

Add a small mixture-density head on `(h, z)`:

- 5-component 2D diagonal-Gaussian MDN predicting the fine GPS offset; total output dim = 5·(1 + 2 + 2) = 25. Parameter cost < 1M (model currently ≈2.63M) — VRAM negligible on both GPUs.
- Target: residual between true position and the coarse decode (or previous-fix-relative displacement — Coder picks whichever integrates cleaner with the existing Huber path; document the choice).
- Loss: standard MDN NLL, weight λ_mdn = 1.0; keep the existing Huber term at 1.0 as the coarse anchor.
- Rationale: routes fine-position gradient through continuous parameters instead of through the categorical bottleneck — directly attacks §1.3's ceiling and, by the same mechanism, the ol@4-vs-cv gap.

### 2.3 MANDATORY addition — prior-side road signal

New auxiliary loss: decode road CE from the **prior** sample (ẑ ~ p(z|h_t), no z1 access): `L_prior_road`, weight 0.5. This is the only training-time pressure making the GRU/prior road-predictive — the posterior road term provably provides none. Also required: v1 §3.5 eval harness (prior-rollout Hit@1@{1,5,15}) stays the checkpoint gate.

### 2.4 SECONDARY fix — 32×32 latent (Option A), gated

32 categoricals × 32 classes is DreamerV3's universal default at every model scale (XS→XL); 16×16 was a down-scaling. Cost: ≈2.63M → ≈4.79M params (~57MB with Adam states) — trivial on this hardware. Hold this for Run 2 (§3) — apply only if Run 1's plateau persists; it adds capacity slack but does not by itself remove the hard-sampling discreteness issue.

### 2.5 REJECTED — symlog two-hot GPS head (Option B)

Misapplied precedent: DreamerV3 uses two-hot classification for reward/value ONLY; continuous vector observations use symlog + MSE. GPS behaves like an observation, not a reward. Do not implement.

### 2.6 Eval harness fix (before trusting ol@N as a gate)

`eval_stage2.py:162` reuses `cand_segment_id[:, T_WARM, 0]` unchanged for every imagined step — stale road conditioning during rollout. Fix: advance the conditioning segment with the model's own imagined position (or drop road conditioning during imagination and note it). Until fixed, treat ol@N numbers as biased; report ol@N vs cv@N side-by-side in every eval log.

---

## 3. Run Plan (Coder agent)

**Run 1 (one Kaggle session):** v1 KL spec (free bits + 0.5/0.1 + lr 1e-4 warmup) + MDN residual head (§2.2) + prior-road aux loss (§2.3) + eval fixes (§2.6). Architecture stays 16×16, B=16, L-curriculum unchanged. Stage-0/1 stay frozen.

**Run 2 (only if gate below fails):** add 32×32 latent (§2.4), rerun.

### Gates and stop-loss (extends v1 §4)

| Condition | Action |
|---|---|
| cl_gps < 0.10 km by 15k steps (Run 1) | continue to 29k; success band |
| cl_gps still > 0.15 km at 15k, MDN NLL flat | trigger Run 2 (32×32) |
| ol@4 ≤ cv@4 after eval fix at 20k | decode ceiling persists — Run 2 |
| prior-road Hit@1@1 < 0.85 at 20k | do NOT enter Stage 3; escalate (action/heading conditioning on GRU input — new proposal round) |
| L_prior_road destabilizes gps/speed (loss spikes >2× for >500 steps) | halve λ_prior_road to 0.25 once; if repeat → drop to eval-only and flag |
| KL pinned at 1.0-nat floor > 2k steps | free-bits wiring bug — abort, fix, restart |
| NaN anywhere | abort immediately |

**Stage-3 entry gate (unchanged in spirit from v1, now measurable):** prior Hit@1@1 ≥ 0.95 AND Hit@1@15 ≥ 0.5 AND cl_gps ≤ 0.10 km AND ol@{8,16} beat cv, all on held-out Porto with the fixed eval harness.

---

## 4. Source Anchors

- DreamerV3: arXiv:2301.04104 (Eq. 4/5 KL; Table W.1 hyperparams; 32×32 universal; two-hot = reward/value only; obs = symlog+MSE)
- MDN precedent: Ha & Schmidhuber arXiv:1803.10122 / arXiv:1809.01999
- Discrete-vs-continuous latent precision: arXiv:2503.00653, TD-MPC2 arXiv:2310.16828
- Empirical basis: direct checkpoint execution (step29000), `stage2_full_run.log`, cache `porto_n200000_r50_k10.npz` (98.7% rows ≥1 candidate)
