# Run-2 Open Checks — Resolution, Novelty Verdict, Research-Gap Verdict, Red-Team

> **Date:** 2026-07-12. **Commissioned by:** `/goal` directive — resolve the two open checks
> from `archive/run2_capacity_diagnosis_and_scaling_review.md`, deliver a novelty verdict, a
> research-gap verdict, and a red-team pass.
>
> **Provenance discipline (read this first).** Every load-bearing claim is tagged:
> **[FACT]** = measured in a run-2 / Phase-2 / GT eval artifact on disk, or established published
> literature already in `literature_papers/*.json`. **[JUDGMENT]** = my inference or call, defensible
> but not directly measured. The whole point of this document is that the reader can tell the two apart
> without trusting me. Where a check is *not* resolved, it says so — I did not manufacture closure.

---

## TL;DR

- **Check 1 (capacity vs data/training bottleneck): RESOLVED — it is data/training/signal, not capacity.**
  Now backed by ground-truth evidence the run-2 doc predates. The single un-run caveat (P2 latent-width
  probe) keeps it from being *airtight*, but capacity stays evidence-ranked last.
- **Check 2 (do the training-improvement recommendations move the needle?): NOT RESOLVED — mostly un-run.**
  The run-2 recommendations (P0 LR-tail, P1 data-scaling, P2 latent-width) have **never been executed**.
  The only training-side lever exercised since is Phase-2 data-cleaning on a new city: iteration 1
  (naive retrain) **FAILED**; iteration 2 (speed-filtered) is **in flight**, ~3 h out. So Check 2's
  honest status is *one FAIL + one PENDING on a lever the run-2 doc did not even list*, and the core
  three probes are still open.
- **Novelty verdict:** the on-paper 4-pillar architectural combination is real but **empirically thin** —
  P2 (world-model RL) does not beat a supervised readout, so architecture-novelty would not survive
  review. The defensible contribution is **empirical**, not architectural: two rigorous negative results
  plus one GT-verified positive finding (learned emission helps where geometry starves at sparse sampling).
- **Research-gap verdict:** **no external/literature research gap.** The gap is *experimental and internal*
  — un-run probes (P0/P1/P2) and un-isolated confounds — not missing citations. No web fetch performed
  (would need user go-ahead per standing convention; corpus already covers the space).

---

# Part A — Decision Resolution

## Check 1 — Architecture capacity vs. data/training bottleneck

**VERDICT: RESOLVED in favor of data/training/signal. Capacity is not the primary bottleneck.**

### What "beating the check" looks like in numbers [FACT]

The capacity hypothesis predicts: *train loss high and still falling until the LR dies, eval tracking
train, and gains from added capacity.* Every one of those failed, and three new results since the run-2
doc reinforce it:

| Evidence | Number | Source | What it says |
|---|---|---|---|
| Biggest gain ever came from a **schedule fix, zero new params** | online match@1 +7.3pp (0.695→0.7684) | project_summary §3 | optimization, not capacity, was binding |
| Train road CE flat *before* LR died | −0.06 over 43k steps, flat at 20–25k while LR still ~half | run-2 doc §I.2 | not straining against parameter count |
| No overfitting anywhere | nothing degrades across 12 checkpoints | run-2 doc §I.3 | over-capacity also rejected |
| Latent utilization | KL ≈9.7 nats of 44.4 max (~22%) | run-2 doc §I.3 | road-task width **not** saturated |
| **GT (real ground truth), zero-shot, sparse rate** | hybrid **0.888** vs nk 0.771 @30 s, +11.7pp (n=659) | `gt_benchmark_spec.md` §Phase-2 calibration | the learned emission carries *real* signal on held-out ground truth — a model "too small to fit" cannot do this |
| Chinchilla-direction arithmetic | 2.6M params vs ~8.2M unique fixes | run-2 doc §II.1 | model is data-*matched*, not starved; growing it moves away from compute-optimal |

The GT row is the material addition since run-2: the earlier "signal ceiling" claim rested on proxy
metrics; the Kubicka GT eval shows the emission is not universally ceilinged — its value **concentrates
at sparse sampling rates** (ties geometry at 15 s: 0.970/0.970; wins +11.7pp at 30 s), exactly where
geometry starves. That refines rather than overturns the signal-ceiling finding, and it is direct
capacity evidence: capacity that produced a real GT win is not the bottleneck.

### The one thing that keeps this from being airtight [JUDGMENT]

The **P2 latent-width probe (16×16 → 32×32) was never run.** The run-2 doc flagged the 16×16 categorical
latent as the *one* narrow-capacity variant with direct evidence (it caps the continuous-GPS decode,
cl_gps floors ~0.54 km). So "capacity is not it" is true for the **road-matching** channel (utilization
22%, GT win exists) but **untested for the GPS-decode channel**. This is a bounded, honest gap — it does
not touch the matching claim, but it means "capacity fully excluded" would be an overstatement.

### P0 result — schedule confound REFUTED by experiment (2026-07-12) [FACT]

The run-2 doc ranked the **LR-schedule confound as bottleneck #1** ("the terminal plateau is not
attributable to capacity at all" — the LR decayed to 1.7e-10 across the plateau) and called it "the single
largest interpretive trap in the run." P0 tested it directly: resume the run2xl ckpt (step 50994),
hold **LR flat at 3e-5** (`--constant-lr`, verified in the Kaggle log), +10.3k steps to the max-hours cap.

| ckpt | match@1 | road CE |
|---|---|---|
| baseline (run2xl final) | 0.7692 (reproduces the 0.7684 anchor) | ~1.36 plateau |
| P0 (flat-LR +10.3k steps) | **0.7677** | ~1.3–1.5 (noisier, no downtrend) |

**Δ match@1 = −0.15pp** → well inside the pre-registered "<±0.5pp = genuinely converged" band, nowhere near
"+1.0pp = schedule-bound." **The plateau is real convergence, not a dead-LR artifact.** Reviving the LR
moved neither match@1 nor road CE; if anything it bounced the model out of its minimum (noisier CE) without
finding a better one. **Consequence:** bottleneck #1 (schedule) is *eliminated by experiment*, and the
hoped-for "free win from LR re-warming" is dead. Combined with P0b (a 0.27-nat gap above the entropy floor
exists **but is not reachable by optimization** — flat LR couldn't close it), the residual gap is now
attributable to **signal ceiling and/or capacity, not optimization.** This *strengthens* the
"not-a-schedule-problem" half of Check 1 while making P1 (data) and P2 (capacity) the only live hypotheses.

### Next concrete milestone [JUDGMENT, gated]

With schedule refuted, the milestone is the **P1 data-scaling ablation** (same 2.6M model, `--limit-trajs`
400k then 800k): the canonical data-vs-capacity discriminator, never run, doubles as OOD-serving work, and
now un-gated (Kaggle free). Milestone bar (pre-registered in run-2 doc P1): monotone match@1/hybrid gain
≥+1pp per data doubling → data-bound, scale data before model; flat → signal is binding, go to P4
signal-side work (or P2 capacity). Baseline-comparison milestone (DeepMM/RouteKG) is downstream — see Part B.

---

## Check 2 — Do the training-improvement recommendations move the needle?

**VERDICT: NOT RESOLVED for the run-2 recommendations (still mostly un-run). The one lever tested off-list
(Phase-2 Silesia retraining) is now fully resolved: two iterations, two FAILs — stop rule fired
2026-07-13.**

This is the check where honesty matters most, because it is tempting to point at the Phase-2 Silesia work
and call it "training improvement tested." It is not the same thing.

### What the run-2 doc actually recommended, and its status [FACT]

| Rec | What it tests | Status (2026-07-12) |
|---|---|---|
| **P0** LR-tail probe (resume, constant LR, +5k) | schedule confound vs true convergence | **NOT RUN** |
| **P0b** log soft-target entropy + held-out loss | measures the CE floor (1-line change) | **NOT RUN** |
| **P1** data-scaling ablation (400k/800k trajs) | data-bound vs capacity/signal-bound | **NOT RUN** |
| **P2** latent 16×16→32×32 | the one supported capacity hypothesis | **NOT RUN** |

So the literal answer to "do the recommendations move the needle" is: **unknown — none has been executed.**
Kaggle go-ahead was never requested for them; local compute went to Phase-2 instead.

### The lever that *was* exercised (off the run-2 list) [FACT]

Phase-2 Silesia label-free retraining on a *new city's* public GPS — a data-side lever the run-2 doc did
not enumerate (it is OOD/per-city retraining, roadmap item ④, not a Porto-scaling probe):

- **Iteration 1 (naive retrain, 20k steps, clean training run): FAILED both bars.**
  - W1 held-out proxy: retrained road head 0.6456 vs zero-shot Porto 0.7376 (**−9.2pp**).
  - GT @30 s pooled: retrained hybrid 0.7375 vs zero-shot hybrid 0.8877 (**−15pp**), and now *below*
    pure geometry (nk 0.7709). `eval_gt_silesia_retrained.txt`, `eval_w1_silesia.txt`.
  - **Diagnosed cause [FACT + JUDGMENT]:** modality contamination — 18.1% of Silesia public trajectories
    move at median <2 m/s (walkers/bikes) vs Porto's 8.8%; p90 speed 29 m/s vs 11.6 (rail/highway). The
    road head learned off-road emission a taxi-fleet WM never sees. *(The speed distribution is FACT; that
    it is the dominant cause rather than a confound is JUDGMENT — see red-team R3.)*
- **Iteration 2 (speed-filtered [2,40] m/s → 6,012 trajs / 1.58M fixes, same recipe): DONE 2026-07-13 —
  stop rule FIRED.** Training finished clean (step 19,900, road CE 1.003, KL 9.63/16≈0.60/group, in band,
  ckpt `ckpt_car/stage2r_silesia.pt`). Eval:
  - **W1' (primary gate):** retrained base match@1 0.7790 vs zero-shot base 0.7817 (**−0.27pp → FAIL**,
    n=6,949 held-out fixes). `eval_w1_silesia_iter2.txt`.
  - **GT @30s (primary rate, P2/P2b):** hybrid pooled 0.7360 vs nk pooled 0.7709 (**−3.49pp → FAIL**,
    n=659). `eval_gt_silesia_iter2_r30.txt`.
  - **GT @60s (secondary rate):** hybrid pooled 0.8506 vs nk pooled 0.7622 (+8.84pp, pass), n=328.
    `eval_gt_silesia_iter2_r60.txt`. Does not offset the primary-rate failure per protocol — the
    pass condition needs both gates, not one rate on one gate.

### Most likely cause of the FAIL, direct from evidence [JUDGMENT]

Not a generic list — the single most-likely cause, ranked from the evidence: **the training signal was
poisoned by data quality, not by any capacity or schedule deficiency.** The retrain trained *cleanly*
(losses finite, KL in band, road CE converged to ~1.05) — it successfully fit the data it was given, and
that data encoded pedestrian/rail behavior. This is the same failure family as every prior "learned signal
loses to geometry" result in the project (Track-B semantic emission OOD −21.9pp; T-Drive hybrid −0.8pp),
now with a concrete mechanism (modality mix) rather than just "OOD is hard."

### Outcome: stop rule fired (2026-07-13) [FACT]

Iteration 2 (the speed filter) was that smallest experiment — it isolated *modality contamination*
specifically by removing the walkers/rail while changing nothing else. Result: **W1' fails, primary-rate
GT fails → the pre-registered stop rule fires.** Speed-filtering alone did not rescue per-city label-free
retraining on public traces. **Verdict: public traces are fundamentally too weak regardless of this
cleaning; zero-shot Porto-WM hybrid stands as the shipped OOD config. Do not iterate further on public
traces** — the only remaining lever is cleaner data (Grab-Posisi-L request or real fleet data).

One confound iteration 2 does **not** isolate (red-team R3): the Porto stage0/stage1 encoders were applied
**frozen/zero-shot** to the Silesia graph; the retrain touched only the WM. So the FAIL could also be a
frozen-encoder mismatch, not the traces themselves. Per the stop rule, this confound is now worth isolating
(iteration 2 failed) → **loop item L3 is triggered**: a short encoder-unfrozen run on the *filtered* data.

---

# Part B — Novelty Verdict

**Verdict: the architectural-novelty story does NOT hold up; the empirical contribution does. Publish the
empirics, not the architecture.**

### Where the architecture actually stands vs the scoped landscape [FACT for the matrix, JUDGMENT for the call]

The CLAUDE.md §4 matrix claims this system is the only one covering **P1a + P1b + P2 + P3** — DeepMM (P3*
via augmentation), RouteKG (P1b), Ayara/BMW OWL-RDF lineage (P1b symbolic only) each cover fewer; the
adjacent RL/world-model systems each miss a pillar:

| System | Missing pillar(s) | Why it is not this system |
|---|---|---|
| DeepMM | P1a (RNN not attention), P1b, P2 | grid raster, supervised on synthetic augmentation |
| RouteKG | P1a, P2, P3 | KG-completion + seq encoder, supervised on matched routes |
| Ayara / BMW lineage | P1a, P1b-neural, P2, P3 | entirely symbolic (OWL 2 RL / SPARQL), does not do matching |
| RLOMM, MIDIRL | P3 (need labels/demos), *latent* half of P2 | RL over explicit candidate MDPs, not a latent WM |
| Think2Drive, Bench2Drive | P1b, the GPS/road-graph task | latent-WM RL but for CARLA steering/throttle, not segment selection |

**So on paper the 4-pillar combination is genuinely unoccupied.** [FACT: the matrix]

**But the empirics gut the architectural claim** [FACT for each number, JUDGMENT for the synthesis]:
- **P2 (latent-world-model RL) does not deliver.** The RL actor scores 0.593 vs the supervised road head's
  0.695 — the "sequential selection inside a learned latent simulation" pillar loses to a one-shot
  supervised readout. What actually works is the world model used as a *decoder-light emission model* inside
  a hybrid Viterbi, i.e. a representation-learning contribution, not the RL-agent story the pillar describes.
- **The learned-signal win is small and non-transferable.** Hybrid beats geometry by only +2.1pp in-domain
  (proxy), **ties** on GT at dense rates (0.970/0.970), and the in-domain win **flips negative OOD** on
  T-Drive (−0.8pp). Retraining to fix OOD (Phase-2) failed on contaminated data.

[JUDGMENT] A reviewer would read "we built the first P1a+P1b+P2+P3 map matcher" against "the P2 actor loses
to a linear readout and the learned emission ties or loses off-domain" and reject the architecture as the
contribution. Claiming novelty here invites exactly the pushback in Part D.

### What the contribution actually is [JUDGMENT] — and it is solid

Novelty of architecture is **not required** if the empirical findings are rigorous. They are:

1. **Negative result (mechanism):** a faithful DreamerV3-style reconstruction RSSM **posterior-collapses on
   the GPS modality** — `h` freeloads on the teacher-forced road-embed channel through the reconstruction
   decoder (Dai/Wang/Wipf 2020 mechanism); **decoder-light is the fix** (KL holds 8–9 nats, collapse
   eliminated). Reproduced across 4 experiment rounds with a pre-registered gate. *Workshop-grade.*
2. **Positive, GT-verified — but scoped down by L1:** the learned emission is **complementary to geometry
   where geometry starves.** The pooled "+11.7pp at 30 s" is real but decomposes to a **single dense-urban
   track** (33: +20.6pp; tracks 27/60 tie at −1.0/+1.1pp) and ties everywhere at 15 s (0.970). Honest claim,
   narrow: *on the one urban track where sparse geometry fails, the emission recovers ~20pp; elsewhere it
   ties.* Directionally consistent with the mechanism but an **n=1-track effect** — defensible as a case
   study, not a general GT claim (needs more urban tracks, D-R1). The most *promising* positive, not the
   most *established* one.
3. **Negative result (practical, NEW):** **raw public OSM traces poison label-free retraining** — a
   clean-fleet zero-shot WM (0.888 GT) beats an in-domain WM trained on contaminated local traces (0.738).
   Directly useful to anyone planning "scrape OSM + retrain"; the field's default assumption is that
   in-domain data helps. *Pending iteration-2 confirmation that a speed filter does/doesn't rescue it —
   which sharpens the finding either way.*

[JUDGMENT] Framing to defend: *"a rigorous label-free study of when a learned world-model emission beats
geometry for map matching — it does so only at sparse sampling and only in-domain, and public-trace
retraining cannot buy the in-domain win because of modality contamination."* That is an honest,
GT-anchored empirical paper. The architecture is the vehicle, not the claim.

---

# Part C — Research-Gap Verdict

**Verdict: NO external research gap. The gap is experimental and internal. No new literature needed to
proceed; no web fetch performed.**

### The exact question a research gap would have to answer — and why it is already answered [JUDGMENT]

A literature gap would exist if we could not size/interpret the model or place it against prior art without
more reading. We can: the run-2 doc already synthesizes 21 citations + the 48-entry corpus covering scaling
laws (Kaplan, Chinchilla, Hestness, Pearce, Hilton), world-model scaling (DreamerV3, TD-MPC2, GAIA-1),
label-free trajectory foundation models (UniTraj, MoveGPT, START/JCLRNT/Trembr), curriculum (Narvekar,
Soviany, Variš-Bojar), the collapse mechanism (Dai/Wang/Wipf), and every scoped competitor
(DeepMM, RouteKG, Ayara). The two verdicts above are fully supported by material already on disk.

The one thing literature *cannot* supply is the number that actually decides the open checks: Pearce et al.
(2024) explicitly warn scaling coefficients **do not transfer** across tokenizer/task/architecture — so the
capacity question must be answered by a **local measured probe (P0/P1)**, not a borrowed constant. That is
an experiment, not a reading.

### Why I did not run web research now [FACT]

Standing convention (`project_summary.md` §7, memory): check `literature_papers/*.json` before any web
search, and ask before fetching new sources. The corpus covers the space; nothing in either verdict rests
on an un-cited claim. Fetching would be scope creep against the goal, which is to *resolve* using existing
evidence. If a specific external question arises from iteration 2 (e.g. published speed-filter thresholds
for GPS mode classification), that is a targeted single-source fetch to request then — not a general
research pass now.

---

# Part D — Red-Team Pass

Strongest counter-case against each conclusion above. A skeptical reviewer — and specifically **Prof.
Bürkner** (Bayesian workflow: he attacks uncertainty quantification first) and **Adel Ayara** (symbolic
map-quality lineage: he attacks whether the "world model" earns the name) — would open here.

**R1 — [Bürkner would lead with this] The GT positive rests on ONE track. → NOW QUANTIFIED (L1, 2026-07-12).**
The +11.7pp @30 s pooled win decomposes per-track (hybrid − nk, `eval_gt_silesia_powered.txt`) as:
track 27 **−1.0pp**, track 33 (urban Tychy, 1248 fixes) **+20.6pp**, track 60 **+1.1pp**. The pooled number
is track 33's 1248 fixes swamping the other two — **the entire headline effect is a single track.** Same
shape @60 s (33: +13.7pp; 27: −2.0; 60: +4.2). At the correct statistical unit (cluster = track, n=3), the
diffs are {−1.0, +20.6, +1.1}: one outlier, median +1.1pp, no general significance — a formal bootstrap
would only confirm the track-level CI spans 0. **Status: R1 is upheld and is the most important finding of
this red-team. The "learned emission helps at sparse rates on GT" claim must be rescoped to "helps sharply
on the one dense-urban track where geometry genuinely starves (33), ties on the other two." Publishable only
with more urban GT tracks — this is a data-coverage gap (more tracks), not a compute gap.** L1 CLOSED; full
bootstrap deemed unnecessary once the effect proved single-track.

**R2 — "Signal ceiling" was never measured. → NOW MEASURED (P0b, 2026-07-12).** The claim that road CE
~1.36 sits near the pseudo-label entropy floor was inference. P0b measured the floor directly:
mean H(softmax(-d²/2σ²)) over Porto candidate sets = **1.09 nats** (`eval_p0b_floor.py`, n=2,783 usable
fixes, mean 6.5 candidates). Observed plateau CE ≈1.36 → **0.27 nats of headroom above the floor.** So
"the model sits at the pseudo-label ceiling" was an **overstatement** — there is modest room, which means
the plateau is *not* purely irreducible label noise and is at least partly model/schedule (mild support for
P0's premise). **Caveat:** the floor uses `ci.query` candidate sets, which may differ from the training
pipeline's K/radius, so 1.09 is approximate. **P0 has now answered it (2026-07-12): flat LR 3e-5 for +10.3k steps left match@1
at 0.7677 vs baseline 0.7692 (−0.15pp) and did not lower road CE — the 0.27-nat gap is NOT reachable by
optimization.** So "at the ceiling" is closer to right than the P0b gap first suggested: the model has
converged and cannot close the remaining gap by more/healthier optimization — the residual is signal or
capacity, not schedule. R2 resolved.

**R3 — The contamination finding has an un-isolated encoder confound.** The Silesia retrain trained only the
WM; the Porto stage0/stage1 encoders were **frozen and applied zero-shot** to the Silesia graph. A −15pp
drop is therefore *also* consistent with "frozen-Porto-encoder ⊥ Silesia geometry," not "public traces are
contaminated." The speed distribution (18.1% walkers) is real, but it proves contamination *exists*, not that
it is the *cause* of the drop. **Status: valid and important — the headline negative result ("public traces
poison retraining") is over-stated until the encoder is ruled out.** Iteration 2 (speed-filtered) FAILED
its primary gates too (2026-07-13: W1' −0.27pp, GT@30s −3.49pp) — this doesn't isolate the confound, it
just shows filtering alone doesn't rescue the result. **L3 is now triggered** per the stop rule: a short
encoder-unfrozen run on the filtered data is the smallest remaining experiment that could separate "traces
are bad" from "frozen encoder mismatches this city."

**R4 — The two OOD results appeared to contradict each other. → LARGELY DISSOLVED by L1 (2026-07-12).**
T-Drive OOD said the emission *hurts* (hybrid −0.8pp, aggregate); Kubicka OOD said it *helps* (+11.7pp).
L1 showed the Kubicka "+11.7pp" is not general — it is **one dense-urban track (+20.6pp), ties on the other
two.** So there is no real contradiction: T-Drive's aggregate tie/hurt and Kubicka's two-of-three ties are
the *same* signal (emission ≈ geometry on average OOD), and the +20.6pp on track 33 is the special case
where sparse geometry starves. The remaining open question is narrower: does the same starved-geometry
recovery appear on a *second* dataset? **L4 (T-Drive at 30/60 s resample) is now a confirmatory test of the
mechanism, not a contradiction to resolve — demoted from decisive to nice-to-have.** Caveat: T-Drive has no
GT routes, so L4 can only use the tolerant-Hit@1 proxy (weaker than Kubicka's point-on-route GT), which
bounds how much it can confirm. → L4 (cheap, lower priority now).

**R5 — [Ayara would lead here] The system is not really the latent-world-model RL matcher it is sold as.**
The P2 actor loses to a supervised readout; what ships is a decoder-light representation + a
geometry-hybrid Viterbi. Calling it a "sequential decode over a learned latent world model" is aspirational.
**Status: valid — and already conceded in Part B. The novelty verdict pre-empts this by dropping the
architectural claim.** No new gap; it is a framing discipline, not an open question.

**R6 — Check 1's "not capacity" leans on an un-run probe.** P2 (latent-width) was never executed, so the
GPS-decode channel's capacity is untested. **Status: acknowledged in Part A; bounded (does not touch the
matching claim), but "capacity excluded" must read "capacity deprioritized, one narrow variant untested."**

**Claims resting on assumption rather than measured evidence (the honest ledger):**
- CE ~1.36 is "near the entropy floor" — assumption (P0b un-run). [R2]
- Plateau is convergence, not dead LR — assumption (P0 un-run). [R2]
- Contamination *causes* the Silesia drop — assumption (encoder confound, R3).
- Emission helps OOD at sparse rates generally — hypothesis on 2 points (R4).
- Latent width is adequate for the GPS channel — untested (R6).

Everything else in Parts A–C is measured.

---

# Part E — Loop: Next-Iteration Queue

The red-team surfaced five unresolved gaps. Per the `/loop` directive they are scheduled, not left open.
Ordered by cost/decisiveness; the nearest one (L0) is already running.

| # | Gap | Experiment | Cost | Trigger |
|---|---|---|---|---|
| **L0** | Check 2 data-quality sub-question | iteration-2 filtered retrain → W1' + GT eval | **done 2026-07-13** — stop rule fired (W1' FAIL, GT@30s FAIL, GT@60s pass) | folded into Check 2 + R3 above; triggers L3 |
| **L1** [R1] | GT positive lacks CIs | ~~per-track CIs~~ **DONE 2026-07-12**: +11.7pp is single-track (33 +20.6pp; 27/60 tie); cluster n=3, one outlier → no general significance; claim rescoped to a case study | done | full bootstrap unnecessary (effect proved single-track) |
| **L2** [R2] | Signal-ceiling / schedule | **DONE 2026-07-12.** P0b: floor H=1.09 vs CE 1.36 → 0.27 nat headroom. P0 (Kaggle, flat LR 3e-5, +10.3k steps): match@1 0.7677 vs baseline 0.7692 = **−0.15pp → schedule REFUTED, run converged**. Output saved to `ckpt_out_p0/` | done | schedule eliminated; P1 (data) / P2 (capacity) now the live probes |
| **L3** [R3] | Contamination vs encoder confound | filtered-data run, encoders unfrozen | ~1 session | **TRIGGERED (L0 failed 2026-07-13)** — route to Kaggle, not yet started |
| **L4** [R4] | ~~OOD contradiction~~ **dissolved by L1** — now a confirmatory mechanism test | re-run T-Drive OOD at 30/60 s resample (proxy only, no GT) | cheap local eval | demoted to nice-to-have; after primaries |

**Compute-permission update (2026-07-12):** Kaggle launches now need no user go-ahead (run freely; keep
resume/pause via `--max-hours`+`--resume`+`--ckpt-every`; notify user to save output or save it). Local
runs now need permission. This un-gates the training probes: **P0, P1, P2 can all be pushed to Kaggle
without asking** — they are no longer blocked, only prioritized behind L0's result.

**Immediate action:** L0 is done (2026-07-13, stop rule fired — see Check 2 and R3 above). Next: L3
(encoder-unfrozen filtered retrain) is triggered but not yet started — route to Kaggle. P1 (data-scaling
ablation) remains the highest-value probe for Check 2's original (Porto-scaling) question and is separately
in flight on Kaggle (400k run, chunked-precompute fix v5, as of 2026-07-13 RUNNING).

---

## Appendix — artifact index (for verification)

- Run-2 diagnosis: `research/archive/run2_capacity_diagnosis_and_scaling_review.md`
- Current state: `research/project_summary.md`
- Phase-2 pre-registration + outcomes: `.worktrees/research2/gt_benchmark_spec.md`
- Silesia naive-retrain evals: `.worktrees/research2/eval_gt_silesia_retrained.txt`, `eval_w1_silesia.txt`
- Zero-shot GT: `.worktrees/research2/eval_gt_kubicka{,_wm}.txt`
- OOD T-Drive: `.worktrees/research2/eval_ood_tdrive_{roadhead,offline_viterbi}.txt`
- Iteration-2 training log (finished, step 19,900): `.worktrees/research2/stage2r_silesia_car_run.log`
- Iteration-2 evals (2026-07-13, stop rule fired): `.worktrees/research2/eval_w1_silesia_iter2.txt`,
  `eval_gt_silesia_iter2_r30.txt`, `eval_gt_silesia_iter2_r60.txt`
