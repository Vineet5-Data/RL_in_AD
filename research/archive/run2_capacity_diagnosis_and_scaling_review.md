# Is the Architecture Too Small? Run-2-XL Capacity Diagnosis and a Scaling-Literature Review

> **Date:** 2026-07-11. **Scope:** answers two commissioned questions: (I) does the Run-2-XL evidence in `methodology-comparison.md` §8 support the hypothesis that the current training architecture is too small for the label-free raw-GPS task, and (II) what does the literature say about sizing/improving training for a transformer-GNN-encoder + latent-world-model setup trained label-free on GPS trajectories.
>
> **Provenance discipline:** every claim is tagged. **[Lit]** = established published literature, cited. **[Corpus]** = already-verified entries in `literature_papers/*.json`. **[This project]** = own analysis of Run-2-XL logs (`ckpt_kaggle_run2xl_s{1,2}/*.log`), eval tables (`eval_run2xl_roadhead.txt`), and code (`stage2r.py`, `world_model2.py`), performed for this document. Nothing here relaunches training; per `critique_and_next_steps.md` §6.2 the WM track is closed at 60k and any new run below is a proposal requiring its own written success bar and (for Kaggle) explicit user go-ahead.

---

## Verdict up front

**The evidence does NOT support "architecture too small" as the primary bottleneck. It partially supports one narrow variant — the 16×16 categorical latent is a deliberate 4× down-scale of DreamerV3's universal default and measurably caps the fine-GPS channel — but the dominant bottlenecks, in evidence-ranked order, are:**

1. **Learning-rate schedule confound** — the final 20k steps trained at LR ≤ 1e-6 decaying to 1.7e-10; the terminal plateau is not attributable to capacity at all. *(strongest direct evidence)*
2. **Label-free signal ceiling** — training targets are geometry-derived soft pseudo-labels whose cross-entropy has an irreducible entropy floor; the learned emission adds only +2.1pp over a 2-parameter geometric baseline when combined, and loses to it head-to-head.
3. **Unused data headroom** — training used 200k of ~1.7M available Porto trajectories (`--limit-trajs 200_000` default); the model is not data-starved *relative to its size*, but the cheapest scaling axis (8.5× more data, zero architecture change) has never been exercised.
4. **Latent width** (the narrow "too small" variant) — real but bounded: the stochastic channel carries ~9.7 nats/step of a 44.4-nat capacity (~22% utilization), so raw width is not saturated for the road task; only the continuous-GPS decode is structurally starved by discreteness.
5. **Raw parameter count** — least evidenced. No metric in the run behaves the way parameter starvation looks (see §I.3).

---

# Part I — Diagnosis of Run-2-XL [This project]

## I.1 The system under test, quantified

| Quantity | Value | Source |
|---|---|---|
| Model | RSSM2 decoder-light: GRU h_dim=512, stochastic latent 16 groups × 16 classes (256-d), MLP hidden 256, road_dim 256 | `models/world_model2.py:93-166` |
| Parameter count | **≈2.6M** (31.3MB checkpoint = fp32 model + AdamW first/second moments ≈ 3 tensors/param × 4B) | run2xl s2 log; `stage2r.py:289-292` saves `opt.state_dict()` |
| Training data | 200,000 Porto trajectories (default `--limit-trajs 200_000`), ≈41 fixes/traj → **≈8.2M GPS fixes** | `stage2r.py:366`; fixes/traj from eval ratio 20,411/500 |
| Training exposure | 60k steps × batch 16 × L∈{16→64 @15k} ≈ **50M fix-visits ≈ 6 epochs** | run config, §8 of methodology-comparison |
| Eval | 500 held-out trajs, 20,411 fixes, tolerant Hit@1; gate doc acknowledges ±2–3pp ckpt sampling noise | `eval_run2xl_roadhead.txt`; critique §6.3 |

## I.2 What the training curves actually show

Sampled training-log lines (deduplicated across the two chained Kaggle sessions):

| step | total | road CE | prior_road CE | zgps (huber) | kl (nats) | lr |
|---|---|---|---|---|---|---|
| 16,950 | 12.518 | 1.426 | 1.546 | 0.168 | 9.68 | (cosine, declared over 60k) |
| 25,950 | 12.552 | 1.445 | 1.569 | 0.196 | 9.70 | |
| 35,950 | 12.365 | 1.344 | 1.486 | 0.176 | 9.70 | |
| 45,950 | 12.396 | 1.395 | 1.560 | 0.121 | 9.69 | |
| 55,950 | 12.129 | 1.230 | 1.370 | 0.108 | 9.70 | |
| 59,950 | 12.361 | 1.363 | 1.515 | 0.142 | 9.70 | **1.7e-10** |

Eval trajectory over the same span (`eval_run2xl_roadhead.txt`): match@1 0.634 @5k → 0.743 @20k → 0.755–0.768 band from 25k to final (≈+1.5pp over the last 35k steps); prediction hit@1/5/15 sits in a 0.55–0.65 no-trend noise band from the *first* checkpoint at 5k onward; cl_gps (z-only decode) 0.97 → 0.60 @20k → 0.54 final.

Three readings, in decreasing confidence:

1. **Train loss is flat from ~20k onward** (total −0.16 over 43k steps, within line noise; road CE −0.06). Flat-and-high train loss with *no* overfitting gap is the classic *underfitting-shaped* signature — but see the two confounds below before concluding capacity.
2. **Confound A — the LR was dead.** The cosine schedule (declared over the full 60k horizon, correctly, per the Run-1 curriculum-bug fix) decays through 1e-6 into 1e-10 territory across exactly the region where both train loss and match@1 plateau. A plateau under a dying LR is uninformative about capacity: the optimizer was *instructed* to stop moving. This is the single largest interpretive trap in the run, and it is checkable for ~2 GPU-hours (§III, P0).
3. **Confound B — the objective has an entropy floor.** The road head trains against Change-9 *soft* targets derived from geometric candidate scoring, and cross-entropy against soft targets is bounded below by the entropy of the target distribution itself **[Lit: Müller, Kornblith & Hinton 2019, arXiv:1906.02629 — soft targets carry irreducible entropy; minimum CE = H(target) > 0]**. Road CE plateauing at ≈1.36 nats (perplexity ≈3.9 over ~10-candidate sets whose uniform baseline is ln 10 ≈ 2.30) is consistent with approaching the floor of a *pseudo-label* objective, not with a model too small to fit it. The floor's exact value was never measured — logging `H(soft_target)` per batch would settle it in one line of code.

## I.3 Hypothesis-by-hypothesis test

| Hypothesis | Prediction if true | Observed | Verdict |
|---|---|---|---|
| **Architecture too small (global)** | Train loss high & falling until LR dies at *every* scale of the curve; eval tracks train; gains from any added capacity | Train loss flat *long before* LR died (already flat at 20–25k when LR was still ~half); Run-1→Run-2-XL gained **+7.3pp online with zero architecture change** (schedule fix only) | **Not supported as primary** |
| **Overfitting / over-capacity** | Eval degrades late; train→0; gap widens | No metric degrades across 12 checkpoints; train road CE stuck at 1.36 nats; no gap dynamics visible | **Rejected** |
| **Schedule/optimization bottleneck** | Step-changes in eval when schedule fixed; plateau located where LR dies | Exactly observed: match@1 jumped 0.6895→0.7429 across the repaired 15k curriculum switch; terminal plateau coincides with LR < 1e-6 | **Supported (proven once already)** |
| **Label-free signal ceiling** | Learned emission ≈ or < geometric baseline; combining helps a little; more training doesn't move it | WM emission alone 0.7684 (argmax) / 0.7268 (Viterbi) vs NK geometry 0.8447; hybrid only +2.1pp (0.8655); prediction hit@N flat from 5k→60k under 12× more optimization; same shape as Track-B C2/C3 (learned Stage-1 emission lost to geometry in-domain AND OOD) | **Supported** |
| **Latent width too small (narrow variant)** | z-channel saturated (KL near capacity); GPS decode capped | KL ≈9.7 nats of 16·ln16 = 44.4 max (~22% used → road channel NOT saturated); but cl_gps floors at ~0.54km and §6 finding 5 already attributed this to the 16×16 discrete bottleneck; DreamerV3 uses 32×32 at *every* scale incl. its smallest **[Corpus: dreamerv3_diverse_domains.json]** | **Partially supported — GPS channel only** |
| **Data quantity/coverage bottleneck** | Loss/metrics improve with more data at fixed model | Never tested: 200k of 1.7M Porto trajs used; 6 epochs of re-exposure rather than fresh data | **Untested — cheapest open axis** |

Additional negative check [This project]: posterior entropy ≈27 nats ≈ 1.69/group of max 2.77 — the latent is neither collapsed nor saturated; capacity-utilization arguments cut *against* "just make z wider" for the matching task.

**Bottom line for Part I:** the run behaves like a small-but-adequately-sized model that (a) was optimized on a schedule that stopped it, and (b) is squeezing a pseudo-label objective toward its entropy floor, and whose learned emission carries genuinely limited signal beyond geometry — not like a model straining against parameter count. The one structural under-sizing with direct evidence is the 16×16 latent for the continuous-GPS channel, which is a *deliberate, documented* down-scale from DreamerV3's 32×32 universal default.

Limitations of this diagnosis: no held-out *loss* was ever logged (only task metrics per checkpoint), so train/val loss-gap analysis — the textbook instrument — is unavailable; the soft-target entropy floor was not logged; single seed throughout; ±2–3pp checkpoint noise on 500-traj eval bounds the resolution of every claim above.

---

# Part II — What the literature says [Lit] / [Corpus]

## II.1 Scaling laws: how to size a model to a dataset

- **Power-law scaling of loss in model size N, data D, compute C** is established for autoregressive language models (Kaplan et al. 2020, arXiv:2001.08361) and pre-dated by empirical power laws across machine translation/speech/vision (Hestness et al. 2017, arXiv:1712.00409). **Compute-optimal allocation** revised the balance sharply toward data: Chinchilla (Hoffmann et al. 2022, arXiv:2203.15556) found N and D should scale *equally*, with ≈20 tokens/parameter at optimum — most prior models were oversized for their data.
  - **Applied here [This project]:** 2.6M params vs ≈8.2M unique fixes (≈50M fix-visits) puts the run *near or above* the Chinchilla-style data-per-parameter regime, not below it. Naively, an 8M-fix corpus "affords" a ~0.4M-param compute-optimal model; even granting domain differences, nothing in the ratio screams "model far too small for this data." Growing the model without growing the data moves *away* from the optimum.
- **These coefficients do NOT transfer blindly.** Pearce et al. 2024 (arXiv:2411.04434, *Scaling Laws for Pre-training Agents and World Models*) confirm the same power-law *forms* hold for world modeling and behavior cloning in embodied domains, but show the coefficients are **heavily dependent on tokenizer, task, and architecture** — uniform LLM rules of thumb mis-size agentic models. Any sizing decision here should come from a measured local scaling probe (§III), not a borrowed constant.
- **RL-specific scaling:** Hilton, Tang & Schulman 2023 (arXiv:2301.13442) introduce *intrinsic performance* (compute-to-reach-return) and show it scales as a power law in model size and environment interactions across Procgen/Dota 2/MNIST-horizon environments; optimal model size grows as a power law in compute budget. Notably they find optimal RL model sizes are *smaller* than supervised intuition suggests for a given compute level.
- **World-model scaling precedent:** DreamerV3 (Hafner et al. 2023, arXiv:2301.04104; Nature 2025 version) shows **monotonic improvement in both final performance and data-efficiency from ~12M to ~400M params** with a single fixed hyperparameter set — the strongest literature prior *for* trying a bigger world model. Two caveats: (i) our 2.6M model sits *below* their smallest published configuration, so we are off the low end of their measured curve; (ii) their gains are on reward-rich control benchmarks, not a pseudo-label matching objective near its entropy floor — the signal ceiling in §I.3 is a mechanism their curves never had to face. TD-MPC2 (Hansen et al., ICLR 2024, arXiv:2310.16828) reports robust 1M→317M scaling for *decoder-free* world models **[Corpus]**, and GAIA-1 (Wayve, arXiv:2309.17080) demonstrates a 9B-param generative world model specifically for driving — evidence that the *family* scales when the training signal is rich (video prediction), again not label-free matching.

## II.2 "Is my model too small?" — established diagnostics

1. **Data-scaling ablation (the canonical discriminator).** Train the *same* model on {¼, ½, 1×} of the data (or the same data budget at 1×, 2×, 4×): if held-out performance improves with data at fixed capacity, you are data-bound; if it saturates while train loss stays high, you are capacity- or signal-bound (Hestness et al. 2017; standard practice in Kaplan et al. 2020). This is the single highest-information/lowest-risk experiment available to this project and has never been run.
2. **Train/val gap taxonomy.** High train loss + small gap = underfitting *or* irreducible noise; low train + large gap = overfitting; both flat under decaying LR = schedule-confounded (this run). Double-descent work (Nakkiran et al. 2019, arXiv:1912.02292) warns that "bigger overfits" intuitions fail in modern regimes — over-parameterization is usually benign, so fear of overfitting is *not* a reason to stay small.
3. **Irreducible-error accounting.** Scaling-law fits always include an irreducible term E (Hoffmann et al. 2022); for pseudo-label training the floor includes the pseudo-label noise/entropy itself (Müller et al. 2019). A model at that floor cannot be improved by capacity — only by better labels/signal. Measure the floor before buying parameters.
4. **Width/utilization checks for latent-variable models.** KL between posterior and prior measures the information the latent actually carries; comparing it to channel capacity (groups × ln classes) distinguishes "latent too narrow" from "latent under-used" — this project's own posterior-collapse literature (Dai, Wang & Wipf, ICML 2020 **[Corpus]**) is the same instrument pointed at the opposite failure mode.

## II.3 Curriculum and staged training for label-free sequence models

- Curriculum learning in RL is a mature framework (Narvekar et al., JMLR 2020, arXiv:2003.04960 — task sequencing when the target task is too hard to learn directly) and in supervised learning generally (Soviany et al., arXiv:2101.10382).
- **Sequence-length curricula specifically** — exactly what this project's L=16→64 switch is — are standard for transformer/sequence training, and length behaves as a *domain* of its own: models overfit to trained lengths, and staged length growth stabilizes optimization (Variš & Bojar 2021, arXiv:2109.07276). The Run-2-XL result (match@1 +5.3pp across the repaired switch) is a textbook local confirmation. Literature-consistent extensions: more than two stages (16→32→64→128) and re-warming the LR at each stage boundary rather than letting one global cosine span all stages — the observed dead-LR tail is an artifact of the single-cosine choice.
- This project's staged pipeline (stage0/stage1 self-supervised pretrain → stage2r WM) mirrors the trajectory-representation-learning literature's pretrain-then-specialize recipe (START, JCLRNT, Trembr **[Corpus]**), which is the P3-compatible way to add training signal without labels.

## II.4 Architecture scale in the map-matching / trajectory literature

- **DeepMM** (Zhao et al., SIGSPATIAL 2019 / TMC 2020 **[Corpus]**): RNN seq2seq over grid tokens; the papers do not report a parameter count, but the architecture class (2-layer GRU seq2seq + embeddings) is single-digit-millions — same size class as this project's WM. Its >10pp win over classical HMM came from *synthetic data augmentation* (label side), not from scale. *(Size estimate is [This project] inference, marked uncertain.)*
- **RouteKG** (Tang et al., T-ITS 2025, arXiv:2310.03617 **[Corpus]**): KG-embedding + sequence encoder, again small-model class; its gains are attributed to injected relational structure (signal side), not capacity.
- **The trajectory-domain frontier scales data and coverage, not just parameters.** UniTraj (arXiv:2411.03859, NeurIPS 2025) pretrains on WorldTrace — 2.45M trajectories, billions of points, 70 countries — and attributes zero-shot robustness to data diversity and resampling/masking pretext tasks; MoveGPT/TrajMoE (arXiv:2505.18670) scales mobility foundation models via spatially-aware mixture-of-experts across cities. The pattern across the domain: **wins come from data diversity + self-supervised signal design first, parameters second.** For this project that ordering matches Part I's evidence exactly (and OOD generalization — `practical_roadmap.md` item 2 — is precisely what data diversity buys).
- Candidate additional corpora already on disk or public: remaining ~1.5M Porto trajs (on disk), T-Drive (on disk, Beijing), Geolife (on disk), Grab-Posisi (Huang et al., SIGSPATIAL 2019 workshop; Southeast Asia; not currently in `data/`).

## II.5 Regularization–capacity tradeoffs relevant here

- In the over-parameterized regime, capacity increases are typically benign (Nakkiran et al. 2019) — the risk of a 2–5× wider model is wasted compute, not degraded accuracy. Weight decay is already present (AdamW).
- The project's real regularization constraints are structural and already documented: free-bits/KL-balancing and unimix (DreamerV3 recipe **[Corpus]**) govern latent information flow; `--norm batch` was the MuDreamer-motivated stabilizer (Burchi & Timofte 2024 **[Corpus]**) and worked. Any width change must re-verify these (free-bits floor arithmetic changes with group count — 0.5·G+0.1·G nats).

---

# Part III — Prioritized next steps [This project]

Ordered by information-per-cost, each with the success bar to write before launch (project convention). P0–P1 are diagnostic probes that *justify or kill* the expensive changes; do not reorder. Kaggle launches require user go-ahead; the WM track's 60k close-out (critique §6.2) means every item below is a *new pre-registered experiment*, not "round 3."

| # | Experiment | What it discriminates | Cost | Success bar (pre-registered) |
|---|---|---|---|---|
| **P0** | **LR-tail probe:** resume the final 60k ckpt, constant LR 3e-5, +5k steps, no other change | Schedule confound vs true convergence. If match@1 moves, the plateau was the cosine tail, and *all* capacity conclusions from the plateau are void | ~2h single T4 session (or local 4060) | match@1 gain ≥ +1.0pp over 0.7684 (above ckpt noise) → schedule-bound confirmed; < ±0.5pp → run genuinely converged |
| **P0b** | **Log the floor:** add `H(soft_target)` and a fixed held-out *loss* (not just task metrics) to stage2r logging | Whether road CE ≈1.36 is at/near the pseudo-label entropy floor; enables real train/val gap reads forever after | 1-line code change, free | n/a (instrumentation) |
| **P1** | **Data-scaling ablation:** same 2.6M model, same 60k recipe, `--limit-trajs` 400k and 800k (two points) | Data-bound vs capacity/signal-bound — the canonical test (§II.2.1). Also directly serves the OOD/product goals | 2 sessions per point (existing `--max-hours`+`--resume` machinery) | monotone match@1/hybrid gain ≥ +1pp per doubling → data-bound (scale data before model); flat → capacity or signal is binding, proceed P2 |
| **P2** | **Latent-width probe:** 16×16 → 32×32 (DreamerV3's universal default), params ≈+1–2M, fresh 60k run | The one *supported* narrow capacity hypothesis; primary expected effect on cl_gps and the WM emission's hybrid contribution | 2 sessions | hybrid Viterbi > 0.8655 or cl_gps ≤ 0.40km; if neither, the width hypothesis is dead too |
| **P3** | **Joint scale-up** (h 512→1024, hidden 256→512, 32×32 latent ≈ DreamerV3-XS ~8–12M class) — only if P1 or P2 shows headroom | Whether the DreamerV3 monotonic-scaling prior transfers to this task | 2–3 sessions | pre-register against whichever metric P1/P2 moved; must beat that cheaper config, not just 60k baseline |
| **P4** | **Signal-side work** (only path forward if P0–P2 all flat): richer self-supervised objectives (START/JCLRNT-style contrastive auxiliaries on the frozen encoder), cross-city data diversity (T-Drive/Geolife/Grab-Posisi) per the UniTraj/MoveGPT pattern | Breaks the label-free signal ceiling, which capacity cannot | project-scale, not one run | define per sub-proposal; the bar that matters is hybrid > 0.8655 and OOD retention |

**Why this justifies (or blocks) the eventual architecture changes:** the expensive change everyone reaches for first — "make the model bigger" — is the *fifth*-ranked hypothesis by this run's evidence and is explicitly gated here behind two cheap probes (P0, P1) plus one bounded structural test (P2). If P0 moves the metric, a schedule fix (multi-stage curriculum with LR re-warming, §II.3) captures the win at zero parameter cost. If P1 moves it, data scaling is cheaper than architecture and also serves the OOD product goal. Only a flat P0+P1 with a positive P2 constitutes actual evidence for scaling the architecture — at which point DreamerV3's monotonic curve (§II.1) makes P3 a well-grounded bet instead of a guess.

---

## References

1. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020. arXiv:2001.08361
2. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022. arXiv:2203.15556
3. Hestness et al., *Deep Learning Scaling is Predictable, Empirically*, 2017. arXiv:1712.00409
4. Hilton, Tang, Schulman, *Scaling Laws for Single-Agent Reinforcement Learning*, 2023. arXiv:2301.13442
5. Pearce et al., *Scaling Laws for Pre-training Agents and World Models*, 2024. arXiv:2411.04434
6. Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), 2023. arXiv:2301.04104; Nature (2025) 10.1038/s41586-025-08744-2
7. Hansen, Su, Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024. arXiv:2310.16828
8. Burchi & Timofte, *MuDreamer: Learning Predictive World Models without Reconstruction*, 2024. (corpus: `mudreamer_no_reconstruction.json`)
9. Hu et al. (Wayve), *GAIA-1: A Generative World Model for Autonomous Driving*, 2023. arXiv:2309.17080
10. Nakkiran et al., *Deep Double Descent*, 2019. arXiv:1912.02292
11. Müller, Kornblith, Hinton, *When Does Label Smoothing Help?*, NeurIPS 2019. arXiv:1906.02629
12. Narvekar et al., *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*, JMLR 21, 2020. arXiv:2003.04960
13. Soviany et al., *Curriculum Learning: A Survey*, 2022. arXiv:2101.10382
14. Variš & Bojar, *Sequence Length is a Domain: Length-based Overfitting in Transformer Models*, EMNLP 2021. arXiv:2109.07276
15. *UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces*, NeurIPS 2025. arXiv:2411.03859
16. *MoveGPT / TrajMoE: Scaling Mobility Foundation Models with Spatially-Aware Mixture of Experts*, 2025. arXiv:2505.18670
17. Zhao/Feng et al., *DeepMM*, SIGSPATIAL 2019 / IEEE TMC 2020. (corpus: `deepmm.json`)
18. Tang et al., *RouteKG*, IEEE T-ITS 2025. arXiv:2310.03617 (corpus: `routekg.json`)
19. Dai, Wang, Wipf, *The Usual Suspects? Reassessing Blame for VAE Posterior Collapse*, ICML 2020. (corpus: `dai_wang_wipf_posterior_collapse_blame.json`)
20. Newson & Krumm, *Hidden Markov Map Matching Through Noise and Sparseness*, SIGSPATIAL 2009. (corpus: `newson_krumm_hmm_map_matching.json`)
21. Huang et al., *Grab-Posisi: An Extensive Real-Life GPS Trajectory Dataset in Southeast Asia*, SIGSPATIAL 2019 (PredictGIS workshop).

*Uncertainty register: DeepMM/RouteKG parameter counts are architecture-class estimates, not reported figures (§II.4); MuDreamer arXiv ID cited via corpus entry rather than re-verified; the soft-target entropy floor value for this project's Change-9 targets has never been measured (P0b exists to fix that).*
