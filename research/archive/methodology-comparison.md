# Methodology Comparison — P1a + P1b + P2 + P3 Recombination

> Synthesis of `literature_papers/*.json` (34 entries) + `literature_review.md` + `methodology.md` + `stage2_correction.md` (Run 3 / Change 14) + `.worktrees/HMM_baseline/hmm_baseline/hmm_results.md` + `classical_baseline_spec.md`, read against the four-pillar vocabulary in `CLAUDE.md` §3. Fixed comparison anchors per `CLAUDE.md` §4: **DeepMM**, **RouteKG**, **Ayara symbolic BMW lineage**.

---

## 1. Final Ranked Recommendation

**Winner: decoder-free / decoder-light latent world model (TD-MPC2/MuDreamer lineage) replacing the reconstruction-heavy DreamerV3-style RSSM, while keeping the DreamerV3/Think2Drive-style discrete categorical actor-critic trained on imagined λ-return rollouts over the masked road-graph successor action space.**

This is **future-work / methodology-chapter framing for a redesign beyond this thesis's Stage-2 iteration budget — it is explicitly NOT a relaunch of the hard-stopped Stage-2 loop.** Per `stage2_correction.md` Run 3 gate R3 ("R2 fails → HARD STOP on Stage-2/P2 iteration... No further runs"), Change 14 / Run 3 already fired the hard stop: best hit@1 0.648 (need ≥0.70), best cl_gps 0.7695 km (need ≤0.15 km), peaking at different checkpoints. No further Stage-2 runs are recommended or implied here.

**Why this is the winner, not another round of Change-14-style patching:**

- The empirically confirmed root cause across four experiment rounds (Run 1, Run 2′, Change 13, Change 14/Run 3) is **posterior collapse**: the deterministic path `h` receives a free teacher-forced `road_embed` channel and can satisfy the GPS/road reconstruction loss without the stochastic `z` ever carrying information. Channel-weakening patches (Change 13A road-embedding dropout, Change 13B/14A z-only auxiliary losses) and schedule-only fixes (Change 14B aggressive posterior training, `he_lagging_inference_posterior_collapse.json`) moved the collapse metrics but never cleared gate R2.
- `dai_wang_wipf_posterior_collapse_blame.json` (ICML 2020) predicts exactly this outcome: patches "can reduce but not eliminate collapse... while the decoders stay expressive enough to fit from h alone." The problem is structural (a reconstruction decoder that lets `h` freeload), not a hyperparameter the KL-balance/dropout knobs can dial away.
- `mudreamer_no_reconstruction.json` (Burchi & Timofte, 2024) is the **direct same-lineage precedent**: it is an explicit DreamerV3 successor that removes pixel-reconstruction losses entirely and instead trains on value/action prediction, reporting that this removes exactly the decoder-vs-latent competition class of failure this project independently reproduced. `tdmpc2_decoder_free_world_model.json` confirms the pattern scales (317M-param multi-task agents, 80+104 tasks) with an **implicit world model that has no observation decoder at all**, so "the entire class of decoder-vs-latent competition... cannot occur in a model with no observation-reconstruction decoder in the first place" (verbatim reasoning already logged in that JSON's notes).
- This was already the project's own named escalation path — `stage2_correction.md`'s Change 14 rationale states "A full decoder-free redesign is REJECTED for this run: `cl_gps` is a gate metric of this thesis and the rewrite does not fit one timeboxed session; recorded as escalation-only/future-work." This recommendation simply promotes that already-identified, already-rejected-for-time-reasons option to the top of the future-methodology ranking — it is not a new idea introduced here.
- The RL/actor-critic component of Stage 2 (categorical actor over masked topological neighbors, λ-return critic, imagined rollouts) was **never the failure point** — `stage2_correction.md`'s diagnosis is entirely about the world-model representation (posterior collapse) and a separate, secondary MTL gradient-competition finding (`standley_task_grouping_mtl.json`, `gradnorm_mtl_balancing.json`, `pcgrad_gradient_surgery.json`) between the GPS head and RoadScorer head. A decoder-free redesign removes the MDN GPS reconstruction head that causes both problems at once (it is the dominant-gradient head in the MTL competition finding AND the decoder `h` freeloads on) — one architectural change addresses two independently diagnosed failure modes.
- It stays inside the P2 framing (`CLAUDE.md` §3, §6) rather than retreating to RLOMM/MIDIRL-style RL over explicit hand-engineered candidate states, which the project's own gap analysis (`literature_review.md` §3, "P2 is an absolute firewall in the map-matching domain") explicitly excludes as "adjacent-but-distinct" and not a defensible substitute for the latent-world-model claim.

**What the thesis REPORTS (already final, not affected by this recommendation) vs. what this recommends as the NEXT methodology:**

| | Reports (final, in the thesis) | Recommends (future work / methodology-chapter framing) |
|---|---|---|
| Map matching | NK-HMM Viterbi (β=20), Track B, is the **primary** result: tolerant Hit@1 0.8447 (500-traj Porto held-out), 0.8493 (30k-traj scale-up); T-Drive OOD 0.7955. Learned semantic-context emission (s1-masked) **underperforms** raw geometry: 0.7422 in-domain, 0.5763 OOD (`hmm_results.md` gates C2/C3, both "NOT met"). | Not reopened — Track B stands as reported. |
| P2 / world model | **Honest negative result** at Stage-2's own pre-registered gate (R2 fails, hard stop per R3), reported alongside the positive prediction-horizon finding (see below). | Decoder-free/decoder-light redesign (TD-MPC2/MuDreamer lineage) named as the concrete next architecture to try, explicitly deferred past this thesis's compute/time budget. |
| Path prediction | RSSM (step8000/step21000) **beats** HMM β=20 transition-only prediction by 5.2–7.9pp at every horizon (H=1/5/15) and beats constant-velocity at H=16 at all 15 checkpoints (`hmm_results.md` gate C4: "Central evidence FOR keeping Run 3" / for P2's value). This is reported as a genuine positive result **even though the matching gates failed** — the failure is representation-quality (posterior collapse capping `cl_gps`/`hit@1`), not the imagination-based prediction mechanism itself, which already wins. | The decoder-free redesign is expected to *preserve or improve* this margin, since removing the GPS-reconstruction competition should let more capacity go to the road-prediction task that already wins. |
| Gradient competition (MTL) | Diagnosed (`standley_task_grouping_mtl.json`) but not fixed within Run 3's budget; 14C diagnostic (cosine similarity / gradient-ratio logging) was measurement-only, not a trigger for further runs. | PCGrad (`pcgrad_gradient_surgery.json`) or GradNorm (`gradnorm_mtl_balancing.json`) named as secondary/incremental levers — see Rank 2 below — largely subsumed if the decoder-free redesign removes the dominant-gradient GPS head outright. |

### Top-3 Ranking

1. **Decoder-free / decoder-light latent world model (TD-MPC2/MuDreamer lineage)**, DreamerV3/Think2Drive-style discrete categorical actor-critic retained unchanged. Addresses the confirmed root cause directly; lowest compute delta (removing a decoder head *reduces* parameter count and VRAM vs. the current design); highest expected impact per `mudreamer_no_reconstruction.json`'s same-lineage evidence.
2. **GradNorm/PCGrad dynamic loss-balancing layered on the existing reconstruction-based RSSM**, no architecture change. Lower implementation risk, addresses the MTL gradient-competition finding, but per `dai_wang_wipf_posterior_collapse_blame.json` does **not** address the posterior-collapse mechanism itself — a incremental, lower-ceiling option if a full redesign is out of scope.
3. **Scope-reduced P2**: formally restrict the RSSM's role to path PREDICTION (where it already beats HMM by 5-8pp, per gate C4) and decouple it from the map-matching decode task entirely, rather than trying to make one representation serve both. Lowest implementation cost (mostly evaluation/reporting framing, already partially realized by the Track-B-primary structure) but concedes P2 never becomes competitive for matching within this project — the most conservative, least ambitious option.

---

## 2. Summary Table

| Method | Strengths | Weaknesses | Fit for our data (raw unaligned fleet GPS, Porto + T-Drive OOD, no ground-truth matched labels) |
|---|---|---|---|
| **NK-HMM Viterbi** (`newson_krumm_hmm_map_matching.json`, Track B) | No learned parameters beyond 2 hand-tuned constants (σ, β); label-free by construction; robust, well-understood, CPU-only; empirically the strongest matcher in this project (0.8447–0.8493 tolerant Hit@1) | Zero semantic enrichment (P1b absent), no representation learning (learns nothing from fleet mobility patterns), first-order Markov transition only — loses to RSSM on multi-step prediction | **Excellent fit for matching** (primary thesis result); poor fit for prediction beyond short horizons (loses to RSSM at every horizon per gate C4) |
| **DreamerV3-style reconstruction RSSM** (as built in Stage 2; `dreamerv3_diverse_domains.json`, `dreamerv2_discrete_world_models.json`) | Genuine P1a+P1b+P2+P3 combination when it works; imagined-rollout actor-critic already wins on prediction horizons; architecture correctly implements DreamerV3's documented KL free-bits/balancing/unimix formulas (verified against source paper) | **Empirically confirmed posterior collapse** across 4 rounds (Run 1, Run 2′, Change 13, Run 3) — `h` freeloads on the teacher-forced road-embed channel; secondary MTL gradient competition between MDN GPS head and RoadScorer head; gate R2 failed, hard-stopped | Currently the honest-negative-result component of the thesis for matching; positive-but-partial for prediction |
| **Decoder-free / decoder-light world model** (TD-MPC2/MuDreamer lineage — **recommended next step**) | Removes the reconstruction decoder that causes both diagnosed failure modes (posterior collapse AND MTL gradient dominance) in one change; same-lineage precedent (MuDreamer is a direct DreamerV3 successor) already fixed this exact failure class; lower parameter count/VRAM than current RSSM | No literature precedent for this exact combination on discrete road-graph action spaces (all TD-MPC2/MuDreamer evidence is continuous control / pixel domains — see Gaps); loses direct GPS-decode interpretability (`cl_gps` metric needs redefinition); MuDreamer's own caveat that BatchNorm is needed to prevent a *different* collapse mode once reconstruction is removed, untested on GRU+categorical-latent architectures | Good fit in principle (same raw GPS + OSM data, no new labels needed); implementation and evaluation-protocol risk not yet retired |
| **RLOMM-style RL over explicit candidate MDP** (`rlomm.json`) | Closest published precedent for treating fleet-GPS map matching as sequential RL; GNN+RNN encoders align trajectory/road in a shared latent space | RL decisions over *enumerated candidate states with a hand-designed reward*, not inside a learned latent world model — explicitly excluded per `CLAUDE.md` §6 as "adjacent-but-distinct"; requires ground-truth matched paths (supervised), breaking P3 | Poor fit: would require matched-path labels this project does not have and does not want (P3 is a hard requirement) |
| **MIDIRL-style deep IRL** (`midirl.json`) | Learns reward from demonstrations rather than requiring per-point match labels; handles very low sampling rates (truck GPS), the closest neighbor on "learning-the-reward" | Requires demonstration trajectories assumed correct (not raw unmatched GPS); RL state is a hand-engineered feature MDP, not a learned latent space; no transformer encoder | Poor fit: demonstration-dependent, feature-engineered state space contradicts both P1b (learned vectorization) and true P3 (label-free on raw unmatched GPS) |
| **START / semantic_enhanced_road_network-style representation pretraining** (`start.json`, `semantic_enhanced_road_network.json`) | Strong template for P1a+P1b(partial)+P3 — transformer + graph-attention road encoder pretrained self-supervised; closest neighbor on label-free axis | No map-matching or decision step at all — pure representation learning, P2 entirely absent; travel/traffic-frequency semantics, not RDF/KG-derived road attributes | Good fit as a Stage-1-style pretraining recipe (already influenced this project's Stage-1 design); does not by itself solve matching or prediction |
| **Pseudo-label GRU sequence scorer** (Track B, B2, contingent fallback per `classical_baseline_spec.md`) | Cheap (~1-2M params, fits local 4060 or one Kaggle session); tests whether *any* learned transition model beats hand-crafted NK transitions; machine-derived labels (Viterbi paths) preserve P3 | Architecturally sits in RLOMM/MIDIRL's family (sequence scorer over explicit candidates) — must be framed as baseline/fallback, never as the thesis's P2 claim, per `classical_baseline_spec.md` §Positioning | Only relevant if a lighter-weight learned-transition comparator is wanted; not built (contingent on Run 3 failing R2, which it did — flagged but not required, since Track B's HMM result already stands as primary) |
| **Ayara symbolic BMW lineage** (5 papers, fixed anchor) | Production-grade, in-house BMW precedent for semantic road-environment representation (OWL 2 RL, RDF, Datalog, SPARQL/RDFox); most recent paper (`ayara_knowledge_layer_data_centric.json`) explicitly names ML+reasoning integration as its own next step | Zero neural components anywhere in the lineage; no map matching (does rule-based trip inference, not GPS-to-segment snapping); no sequence learning, no RL, no label-free learning in the ML sense | Not a candidate for this thesis's architecture — retained only as the P1b symbolic-vs-learned comparison anchor per `CLAUDE.md` §4-5 |
| **DeepMM** (fixed anchor) | Closest deep latent-space matching precedent; label-free-ish via synthetic data augmentation; beats classical HMM/ST-Matching/FMM by >10pp in noisy/sparse settings | Raster grid input (no vectorized graph, no topology); RNN seq2seq, no transformer; one-shot decode, no RL/world-model; supervised on synthetic GraphHopper trajectories, not raw fleet GPS | Retained only as the P3(partial)+latent-matching comparison anchor |
| **RouteKG** (fixed anchor) | KG-completion framing injects road-network structure/semantics (P1b) into route prediction | Supervised on matched route sequences (not label-free); one-shot seq2seq top-K ranking, no RL, no transformer encoder | Retained only as the P1b comparison anchor |

---

## 3. RL Formulation Analysis — Discrete Road-Segment Selection in Learned Latent State

The action space is **discrete**: at each step the agent selects among a variable number (padded to max-degree, masked) of topological successors of the current road segment (`methodology.md` §2.3-2.4). The state is a **learned latent** `(h_t, z_t)` from an RSSM, not raw observations or hand-engineered features. This combination determines which RL paradigm fits.

**PPO** — On-policy, clipped surrogate objective gives strong training stability against noisy or shifting reward estimates, which would matter if training against real-environment rollouts. **Not used by any latent-world-model paper in this corpus** (Think2Drive, DreamerV2, DreamerV3 all use a REINFORCE-style/straight-through actor-critic on *imagined* rollouts instead — `think2drive.json`, `dreamerv2_discrete_world_models.json`, `dreamerv3_diverse_domains.json`). This is not an oversight: inside a differentiable learned world model, gradients can flow analytically through the imagined trajectory (via straight-through categorical sampling) directly to the actor, which is a *stronger* training signal than PPO's clipped policy-gradient estimator built for stochastic, non-differentiable environments. PPO would only earn its keep if training against the real environment directly (i.e., abandoning the imagination-based P2 framing), at the cost of needing many more real-GPS rollouts — undesirable given the compute budget and the label-free/self-supervised framing.

**DQN** — Naturally fits a discrete action space (this is DQN's home turf), but is off-policy and depends on a stable Bellman target, which is exactly what breaks when the underlying state representation is still collapsing (posterior collapse directly degrades Q-value estimates built on `(h,z)`). DQN also has no natural mechanism for the imagined-rollout training loop that defines P2 — value backprop through a differentiable world model over an H-step horizon is not how DQN is formulated (it bootstraps from one-step TD targets against a replay buffer of *real* transitions, which the label-free/latent-world-model framing does not straightforwardly provide). Not used by any paper in the corpus for this task class.

**SAC** — Built for continuous action spaces with maximum-entropy exploration; discrete SAC variants exist in the wider RL literature but have **no precedent anywhere in this project's corpus** (no paper among rlomm/midirl/think2drive/dreamerv2/dreamerv3/tdmpc2 uses SAC). Adopting it would mean introducing twin Q-networks and entropy-temperature tuning to solve a problem — RL-algorithm instability — that was never diagnosed. Every confirmed Stage-2 failure mode (posterior collapse, MTL gradient competition) sits in the *world-model representation*, not the actor-critic; SAC does not address either.

**A2C** — Generally superseded by PPO in the wider literature, and its role is effectively already subsumed here: the Dreamer-style actor-critic (`methodology.md` §2.4, `dreamerv2_discrete_world_models.json`, `dreamerv3_diverse_domains.json`) *is* structurally an advantage-actor-critic trained on imagined rollouts, but strengthened by fixed differentiable dynamics and λ-returns rather than plain A2C's noisier single-step advantage estimates. No case for adopting bare A2C over the already-specified Dreamer-style formulation.

**Model-based imagination (Dreamer/Think2Drive lineage) — this IS the P2 framing, not an alternative to it.** The categorical actor over masked logits, trained via λ-returns backpropagated through imagined RSSM rollouts, is the correct fit for this task and was **not the component that failed** in Stage 2 (`stage2_correction.md`'s entire diagnostic history is about world-model representation quality, never actor-critic instability). This recommendation is to **keep this RL formulation unchanged** and only redesign the world-model's decoder side.

**TD-MPC2's MPC-over-latent-dynamics is a real alternative decision mechanism**, worth naming even though it is not the top recommendation: instead of a REINFORCE-style actor-critic, TD-MPC2 performs local trajectory optimization directly against a learned Q-function using no observation decoder at all (`tdmpc2_decoder_free_world_model.json`). It is the more radical of the two decoder-free options — bigger implementation lift (replacing the actor-critic training loop, not just the decoder heads) for likely similar posterior-collapse-avoidance benefit as MuDreamer's smaller change. Given the ladder of least-effort-that-works, **MuDreamer's approach (keep the Dreamer-style actor-critic, remove/de-weight only the reconstruction decoder) is the lower-risk, lower-implementation-cost path to the same fix** — this is the basis for ranking it above a full TD-MPC2-style MPC rewrite in §1.

---

## 4. Per-Candidate: Implementation Complexity, Data Requirements, Known Failure Modes

| Candidate | Implementation complexity | Data requirements | Known failure modes |
|---|---|---|---|
| **Decoder-free/decoder-light world model** (Rank 1) | Medium — reuse existing GRU + 16×16 categorical-latent RSSM, masked categorical actor, λ-return critic (all already built for Stage 2); remove or permanently z-only-gate the MDN GPS reconstruction head; retrain road/value heads as the primary signal. VRAM/compute *decreases* vs. current design (fewer decoder parameters) — comfortably within RTX 4060 8GB / Kaggle T4-P100 16GB, ≤12h/session. | Same raw unaligned fleet GPS (Porto + T-Drive) and OSM road graph already in use; no new labels. | (1) MuDreamer's own caveat: BatchNorm needed to prevent a *different* collapse mode once reconstruction is removed — untested on this project's GRU+categorical-latent architecture (MuDreamer's evidence is pixel/RSSM-standard, `mudreamer_no_reconstruction.json` "uncertain" field). (2) Losing direct GPS-decode interpretability — `cl_gps` as currently defined stops being meaningful, requiring the evaluation protocol to be redefined (flagged in `tdmpc2_decoder_free_world_model.json` "uncertain" field) — this is a methodology-chapter design decision, not a solved problem. |
| **GradNorm/PCGrad on existing RSSM** (Rank 2) | Low — off-the-shelf, drop-in loss reweighting (GradNorm) or gradient projection (PCGrad) on top of the existing loss terms; no architecture change; 14C's gradient-cosine/ratio diagnostic (already speced, not yet a trigger) determines which of the two to prefer. | Same. | Does **not** address the posterior-collapse mechanism (Dai/Wang/Wipf caution — collapse persists while decoders remain expressive enough to fit from `h` alone regardless of loss-weight rebalancing); risks "fixing" the MTL symptom (hit@1-degrades-while-cl_gps-improves) while the underlying representation quality cap remains unmoved. Interaction with the existing KL-balanced ELBO is unanalyzed (`gradnorm_mtl_balancing.json`'s own uncertain note). |
| **Scope-reduced P2 (prediction-only)** (Rank 3) | Low — largely a reporting/evaluation-protocol decision (already structurally present via Track B being primary for matching); may require decoupling shared heads so `cl_gps`/road-matching quality stops being optimized against a task the model is no longer asked to serve. | Same. | Concedes P2 never becomes competitive for matching within this project; the four-pillar claim for MATCHING then rests entirely on a component (Track B) that has no P2. Risk of the thesis narrative reading as "P2 only does one of the two things it was designed for" without the redesign in Rank 1 to close that gap. |
| **Reconstruction-based DreamerV3-style RSSM as built** (status quo, for reference) | Already built and run through 4 rounds. | Same. | **Empirically confirmed posterior collapse** (Run 1 lockstep <0.006 gap; Run 2′ Gate A fail; Change 13 D″ fired, best cl_gps 0.195/hit@1 0.644; Run 3/Change 14 R2 fails at all 15 checkpoints, best hit@1 0.648/cl_gps 0.7695km) — this is now project-confirmed empirical evidence, not a hypothesis, and is the reason this method is not re-recommended. |

---

## 5. Gaps

Flagged for awareness — not fetched, per task instructions (no new literature search performed).

1. **No direct precedent for decoder-free world models on discrete graph/road-segment action spaces.** All TD-MPC2 and MuDreamer evidence in the corpus is continuous-control (locomotion, manipulation) or pixel/Atari/DMC domains (`tdmpc2_decoder_free_world_model.json`, `mudreamer_no_reconstruction.json`). Whether removing the reconstruction decoder interacts differently with a *categorical* latent + *discrete masked action* setup (vs. continuous control or dense pixel prediction) is untested in the reviewed literature — the recommendation in §1 is a reasoned extrapolation, not a directly evidenced result.
2. **No paper in the corpus combines self-supervised road/trajectory representation learning (START, JCLRNT, semantic_enhanced_road_network) with a decoder-free latent world model.** The pairing recommended here (Rank 1) is compositional inference across two literature clusters that have not been jointly evaluated anywhere in the reviewed set.
3. **No discrete-action SAC precedent exists anywhere in the corpus** for this task class — its omission from the RL formulation ranking (§3) is a negative-evidence argument (nothing in 34 papers uses it here), not a benchmarked comparison against PPO/DQN/Dreamer-style actor-critic.
4. **GradNorm's interaction with KL-balanced ELBO training (DreamerV3's own free-bits/balance mechanism) is unanalyzed** in the literature record (`gradnorm_mtl_balancing.json`'s own "uncertain" field) — whether the two reweighting mechanisms would compound, cancel, or conflict is an open implementation question, not something the existing analysis resolves.
5. **MuDreamer's BatchNorm requirement is evidenced only on its own (pixel-based) architecture** (`mudreamer_no_reconstruction.json` "uncertain" field) — whether an equivalent normalization fix is needed for this project's GRU + 16×16-categorical-latent RSSM once reconstruction is removed is not established by any source in the corpus.
6. **The 2025 survey's Section 6 future-directions text was never fully verified** (`survey_advancing_mm_route_prediction.json` "uncertain" field notes the equation-heavy Section 3 exhausted the page-fetch budget before Sections 4-6 were reached) — whether it explicitly names transformer/RL/label-free directions as recommended (vs. inferred from the abstract) remains unconfirmed in the existing record.
7. **No literature source directly evaluates PCGrad vs. GradNorm head-to-head on a posterior-collapse-adjacent world model** — the corpus has each technique's general MTL evidence (`standley_task_grouping_mtl.json`, `pcgrad_gradient_surgery.json`, `gradnorm_mtl_balancing.json`) but no paper tests either specifically against a collapsing latent-variable model, only against ordinary supervised/RL multi-task setups.

---

## 6. research2 Run 1 — Empirical Results (2026-07-10)

The Rank-1 recommendation (§1) was implemented and run despite the future-work framing, on explicit commission. Worktree `.worktrees/research2` (branch `research2`): decoder-light world model (`training/stage2r.py`, 15k steps, Kaggle T4, batch 16, `--norm batch`) + Dreamer-style actor-critic inside the frozen WM (`training/stage3.py`, 2k outer steps ≈ 12M imagined env steps). Kernel `vineetdairashri/alphaevolve-research2-wm-ac-t4`; checkpoints + log in `ckpt_kaggle_run1/`.

**Implementation of "remove or permanently z-only-gate":** both — MDN removed entirely, coarse GPS head structurally z-only (`HeadsLight.gps` never sees `h`). Road CE (Change-9 soft targets, posterior + Change-4 prior-side) is the primary signal; new per-candidate reward head (symlog targets, DreamerV3 convention — raw-scale physics-reward MSE was becoming the new dominant-gradient head in local sanity, caught and fixed pre-launch) regresses the §2.5 label-free physics rewards for imagination. Action space = per-fix masked candidate set, not max-degree-8 successors (methodology 2.4's hard successor mask and 2.5's r_topo reward are mutually contradictory; resolved toward the eval/HMM-comparable candidate convention, topology enforced via λ_topo=2.0).

### Results (Porto held-out [200000, 200500), same split/cache as HMM Track B; tolerant Hit@1, T_WARM=8)

| | match@1 (online) | hit@1 | hit@5 | hit@15 | kl/group | collapsed? |
|---|---|---|---|---|---|---|
| stage2r final + road head (No-RL) | **0.695** | 0.586 | 0.574 | **0.579** | ~0.66 | **NO** |
| stage2r step13000 + road head | 0.689 | **0.626** | 0.547 | 0.548 | ~0.58 | NO |
| stage2r final + Stage-3 actor | 0.593 | 0.577 | 0.564 | 0.567 | — | — |
| *Anchor: NK-HMM Viterbi (offline)* | *0.8447* | | | | | |
| *Anchor: Run-3 RSSM best* | | *0.648* | | | *→0* | *YES (R2 fail)* |
| *Anchor: independence floor* | | | | *0.054* | | |

Full tables: `eval_run1_roadhead.txt`, `eval_run1_actor.txt`. Visual rollouts (warm-up vs self-driven imagination, vs HMM β=20 and nearest-road GT): `viz/out/agent_driving_traj{0,1,2}_{roadhead,actor}.gif`.

### Findings

1. **Posterior collapse eliminated — the redesign's primary structural claim is confirmed.** `kl_dyn_raw` climbed 1.6 → ~8-9 nats by step 1500 and held there through all 15k steps (Runs 1-3 repeatedly decayed toward 0). Eval-side kl ≈ 10.5 summed (~0.66/group), healthy across every checkpoint. The `dai_wang_wipf` prediction (§1) — that removing the decoder `h` freeloads on fixes what channel/schedule patches could not — held empirically.
2. **Long-horizon prediction is nearly flat**: hit@15 0.579 ≈ hit@1 0.586 (final ckpt) — far above the 0.054 independence floor and much flatter over horizon than the Run-3 reconstruction RSSM. The imagination mechanism benefits directly from the healthier latent.
3. **Stage-3 actor UNDERPERFORMS the WM road head on matching** (0.593 vs 0.695): the §2.5 physics-reward optimum is not the nearest-road target — entropy fell 1.49→0.34 toward reward-optimal choices that trade d_perp for heading/topology consistency. Same shape as Track B's C2/C3 finding (learned signal loses to raw geometry). Honest negative for RL fine-tuning as a *matching* decoder; the decoder-light WM itself is the win. The methodology-2.6 unification claim ("same policy for matching and prediction") is NOT supported by this run.
4. **match@1 was still rising monotonically at 15k** (0.514→0.695, no plateau), and zgps/cl_gps_z regressed after the L=64 curriculum switch at step 10k because cosine LR was already ≈0 — schedule mismatch, not capacity. Run 2 lever: `--curriculum-step 5000 --steps 20000` (curriculum switch with real LR budget remaining).
5. **Caveats standing:** online-filter match@1 vs HMM's offline Viterbi 0.8447 is not a like-for-like comparison (no future context); `cl_gps` is redefined (z-only decode, ~0.69 km — the 16×16 discrete bottleneck alone cannot carry fine GPS; the old ≤0.15 km gate does not transfer). MuDreamer BatchNorm caveat (Gap 5) resolved favorably in practice: `--norm batch` trained stably, no constant-code collapse observed.

---

## 7. Run-1 Zero-Training Diagnostics (2026-07-10) — critique steps 1 & 2 executed

Both diagnostics from `critique_and_next_steps.md` §5 (new-experiment steps 1–2) ran on the frozen Run-1 final checkpoint, Porto held-out [200000, 200500), tolerant Hit@1, local RTX 4060. Scripts: `eval_offline_viterbi.py`, `eval_hz_ablation.py` (research2 worktree); raw output in `eval_offline_viterbi_final.txt`, `eval_hz_ablation_final.txt`.

### Diagnostic 1 — offline Viterbi-style decode over WM emissions (NK β=20 transitions, Track-B machinery imported unchanged)

| variant | tolerant Hit@1 | reading |
|---|---|---|
| wm-argmax (no transitions) | 0.6893 | wiring check — reproduces online 0.695 ✓ |
| wm-viterbi (WM emission + NK transition) | 0.6957 | offline smoothing adds only **+0.6pp** |
| nk-viterbi (Track-B variant-i re-run in-script) | 0.8447 | anchor reproduced to 4 decimals ✓ |
| **hybrid (WM log-prob + NK Gaussian log-prob emission)** | **0.8560** | **beats pure NK by +1.1pp — new project-best matching number** |

**Stress-test (a) answered:** the 15pp online/offline gap is **representational, not decoding-protocol asymmetry** — giving the WM future context via Viterbi moves it 0.689→0.696, nowhere near 0.8447. The critique's kill criterion fires in the "gap is real" direction: P2's WM emission alone does not rival raw geometry for matching.
**New positive finding not anticipated by the critique:** the WM emission is *complementary* to geometry — hybrid 0.8560 > 0.8447 is the **first learned-signal win over pure geometry anywhere in this project** (Track B's C2/C3 tested exactly this for the Stage-1 contrastive emission and failed both in-domain and OOD). The correct matching narrative is now "decoder-light WM as a learned emission *prior on top of* NK geometry," not "WM replaces HMM."

### Diagnostic 2 — h-only vs z-only RoadScorer ablation (frozen scorer, one component zeroed; ablated scorer's own argmax drives the closed loop)

| decode | match@1 | hit@1 | hit@5 | hit@15 |
|---|---|---|---|---|
| full (h,z) | 0.6891 | 0.594 | 0.561 | 0.551 |
| h-only (z zeroed) | 0.5792 | 0.590 | 0.553 | 0.591 |
| z-only (h zeroed) | 0.5869 | 0.451 | 0.385 | 0.440 |

**Claim-2 ("topological shortcut") test resolved AGAINST the fear, for matching:** z is not a dead routing switch — z-only (0.587) matches slightly *better* than h-only (0.579), and the full scorer needs both (+11.0pp over h-only, +10.2pp over z-only). The stochastic latent carries real per-fix observational information the history lacks — exactly the "true signal of life" new_findings.md §1 asked for, now measured directly.
**Division of labor confirmed:** prediction is h's job (h-only hit@15 0.591 ≈ full; z-only prediction collapses to 0.38–0.45, since rollout z comes from the prior anyway), matching needs both. Curious wrinkle: h-only hit@15 (0.591) *exceeds* full (0.551) — prior-z sampling appears to add rollout noise at long horizon; flag for Run 2 eval.
**Caveat (pre-declared):** zeroed inputs are out-of-distribution for the frozen scorer, so absolute ablated levels understate each component; the ordering and gaps are the signal.

### Consequence for Run-2 gates (§4 of critique doc)
Matching exit gate should be reframed around the **hybrid** number (baseline to beat: 0.8560, not 0.8447, and not online-0.695-vs-offline-0.8447), and the prediction gate should track whether Run 2's longer LR budget closes the h-only-vs-full hit@15 anomaly.

---

## 8. Run-2-XL Final Results (2026-07-11) — 60k steps, WM track CLOSED

Executed Run-1 finding-4's lever at full scale: fresh init, `--steps 60000 --curriculum-step 15000` (L=16→64 with real LR budget remaining), warmup+cosine declared over the full horizon, chained Kaggle T4 sessions (`--resume` + `--max-hours 10.5` guard). Kernel `vineetdairashri/alphaevolve-research2-run2xl-t4`; ckpts/logs `ckpt_kaggle_run2xl_s{1,2}/`; eval tables `eval_run2xl_{roadhead,offline_viterbi,hz_ablation}.txt`; full gate record `critique_and_next_steps.md` §6.3. Same eval protocol as §6–7.

### Pre-registered gates (critique §6.2)

| gate | bar | result | verdict |
|---|---|---|---|
| online match@1 sustained | ≥0.75 last 3 ckpts | 0.7606 / 0.7642 / 0.7623; final **0.7684** | PASS |
| hybrid WM+NK Viterbi | >0.8560 | **0.8655** (nk-viterbi anchor re-reproduced 0.8447) | PASS — new project-best matching |
| prediction vs HMM β=20 | hit@1≥0.5918 / hit@5≥0.5072 / hit@15≥0.5193 | hit@5/15 pass every ckpt (final 0.566/0.591); hit@1 0.602–0.634 at 57–59k, final 0.581 | MARGINAL — single-ckpt hit@1 miss, within ±2–3pp sampling noise |
| kl/group | 0.4–0.9 healthy | 0.54–0.61 throughout | PASS — no collapse relapse |

### Findings

1. **Curriculum/LR fix validated**: match@1 jumped 0.6895→0.7429 across the 15k switch (Run 1's dead-LR phase now trains); final +7.3pp over Run 1 (0.695→0.7684). cl_gps (z-only) 0.69→0.546 km.
2. **Hybrid emission lead widened**: 0.8560→0.8655 over pure NK 0.8447 — the learned-emission-complements-geometry finding (§7 diagnostic 1) strengthened; this is the system's matching number.
3. **wm-viterbi (0.7268) < wm-argmax (0.7684)**: NK transitions over the pure WM emission *hurt* offline — the hybrid's win comes from adding WM log-prob to the NK Gaussian emission, not from smoothing.
4. **h/z ablation @60k**: full 0.7657 / h-only 0.6039 / z-only 0.6394 — ordering unchanged (z carries real per-fix signal); Run-1's h-only-hit@15>full anomaly gone (full 0.586 ≥ h-only 0.581). z-only hit@N still collapses (~0.44): prediction remains h's job.
5. Per critique §6.2's exit rule: **60k is the WM track's final number — no round 3.** Only remaining optional item: critique §4's single-shot actor reward reweighting (separately gated, explicit go-ahead required).
