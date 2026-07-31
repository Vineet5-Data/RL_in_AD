# Classical Baseline Spec — Track B (NK-HMM family + Stage-1 learned emissions)

> Companion to `stage2_correction.md` (Change 14 / Run 3). CPU-only — starts immediately, runs in parallel with Run 3 on separate compute budgets.
> Pillar mapping (CLAUDE.md §7): the baseline harness serves the evaluation protocol (methodology.md §4.3, which lists OSRM/FMM/ST-Matching as required comparators — none built yet). The learned-emission variant (B1-iii) demonstrates P1a+P1b+P3 value inside a classical decoder, with P2 deliberately absent — that absence is the point of the comparison, per §Positioning below.

---

## Why now

1. **Required comparator regardless of Stage-2's fate.** methodology.md §4.3 commits to classical baselines; the thesis needs this table either way.
2. **If Run 3 fails Gate R2** (`stage2_correction.md`), this becomes the primary quantitative result: an honest four-way comparison — classical vs. classical+learned-emissions vs. RSSM vs. (contingent) pseudo-label GRU.
3. **Label-free story preserved end-to-end.** NK-HMM/Viterbi needs no human labels; pseudo-labels derived from it are machine-derived from raw GPS + map only, so P3 survives intact. Only P2 is at stake in Run 3 — never P3.

## Positioning (novelty guard — read before writing any thesis text from Track B results)

Per `literature_review.md`: a supervised sequence scorer over explicit candidates (B2 below) sits architecturally in RLOMM/MIDIRL's family ("RL/sequence models over explicit candidate states"). It must always be framed as a BASELINE / fallback contribution (P1a+P1b+P3 — a matching head on learned emissions, which the START/Toast-class representation papers lack), never as the thesis's P2 contribution. The four-pillar claim lives or dies with Run 3, and that is written down in `stage2_correction.md` Gate R3 — Track B never silently replaces it.

---

## B0 — assets already in place (verify each before coding, none should need rebuilding)

- **Candidate caches** with per-fix K=10 candidates: fields are exactly `[n_rows, k, segment_id, d_perp_m, heading_deg, speed_kph, oneway, highway_id, mask]` — **no fix coordinates in the cache**. Porto train (`porto_n200000_r50_k10.npz`), Porto held-out (built by `.worktrees/evaluation/build_porto_holdout_cache.py`), T-Drive (`.worktrees/evaluation/build_tdrive_cache.py`). The HMM state space is exactly this cache — no new candidate search needed. Fix coordinates come from the processed parquets (`data/processed/porto/part-000.parquet`; T-Drive equivalent under `data/processed/`) via the same `TrajectoryGraphDataset`/`collate_fn` loader `eval_stage2.py` uses — its batches carry row-aligned coords.
- **Stage-1 Run-0b checkpoint** (`stage1_porto.pt`): visible-pass candidate scores are a calibrated emission distribution (Change 9 trained them against softmax(−d_perp²/2σ²), σ=10m); masked-pass scores are context-only emissions.
- **Held-out split**: unique traj_ids at indices [200000, 200500) in parquet insertion order (`SKIP_TRAJS=200_000`, `N_TRAJS=500`), matching `eval_stage2.py` and `build_porto_holdout_cache.py` verbatim; T_warm=8. (NOT a "last-5%" rule — it is the 500 trajectories immediately AFTER the 200k training subset; implementing anything else breaks cross-method comparability.)
- **Road graph**: read the on-disk parquets directly — `data/osm/porto/segments.parquet` (`segment_id, u, v, length_m`; T-Drive under `data/osm/tdrive/`) plus `data/osm/<city>/line_graph_edges.parquet` (segment-to-segment turn connectivity). That pair is the natural weighted digraph for d_route. Do NOT go through `roadgraph/io.py::load_pyg()` — it returns z-scored features (log-length, no raw lengths, no u/v endpoints) and drags geopandas + torch_geometric into a CPU-only job.

## B1 — NK-HMM Viterbi over the candidate cache

New script: `.worktrees/evaluation/baseline_hmm.py` (beside `eval_stage2.py`).

**State space** per fix t: the cached K≤10 candidates (mask-aware).

**Emission — three variants behind one `--emission` flag:**

| variant | definition | role |
|---|---|---|
| (i) `nk` | Gaussian on cached d_perp, σ=10m: p ∝ exp(−d_perp²/(2σ²)) | literally Newson & Krumm's emission; identical in form to Stage-1's Change-9 soft target |
| (ii) `s1-visible` | Stage-1 forward, geo_mask all False; softmax over candidate logits | sanity anchor — trained to copy (i), so must ≈ (i); if it doesn't, the export path is broken |
| (iii) `s1-masked` | Stage-1 forward, geo_mask all True (d_perp zeroed everywhere): context/heading/graph-only emissions | the honest learned-emission variant; ONLY here does beating (i) mean the learned encoder adds value beyond raw geometry |

**Transition (Newson & Krumm 2009):** p ∝ (1/β)·exp(−|d_route(c_{t−1}, c_t) − d_gc(o_{t−1}, o_t)|/β)

- d_gc: great-circle distance between consecutive fixes (coords from the parquet-backed loader per §B0 — NOT available in the .npz cache).
- d_route: shortest path between candidate segments over the segments + line-graph digraph (networkx 3.6.1 — installed and confirmed importable; `python-igraph` is NOT installed, optional `pip install python-igraph` speedup only; Dijkstra cutoff = 2×d_gc + 500 m; unreachable pairs → probability floor 1e-12).
- β: hand-tuned constant in NK 2009 (no principled estimator claimed there) — sweep {1, 3, 5, 10, 20} m on a 50-trajectory dev slice carved from the TRAINING split (never the held-out 500); pick by tolerant Hit@1.
- Gap handling: split trajectories at dt > 3× median dt; decode segments independently. (Our simplification, in the spirit of NK's junk-point robustness — `literature_papers/newson_krumm_hmm_map_matching.json` records no explicit gap-splitting convention in NK 2009; do not attribute this heuristic to the paper.)

**Decode:** log-space Viterbi over ≤10 states/fix — O(T·K²) per trajectory. The K²=100 route-distance queries per step dominate; memoize d_route per (seg_a, seg_b) pair in a dict (hit rate is high — consecutive fixes share candidates). 500 held-out trajectories ≈ hours-scale on laptop CPU; embarrassingly parallel via `multiprocessing` if slow. This is FMM's insight (precomputed route-distance tables) applied lazily; full FMM (UBODT) only if the memoized version is too slow — do not start with the C++ FMM build on Windows.

**B1-P — prediction mode (the classical counterpart of the RSSM prior rollout — required for a fair comparison):**

`eval_stage2.py`'s hit@k is a PREDICTION metric (8 posterior warm-up steps, then prior-only rollout, no future observations). Viterbi is offline smoothing — comparing it directly to RSSM hit@k is apples-to-oranges. Add a `--predict` mode:
- Forward-algorithm filtering up to t=T_warm (=8, same as eval_stage2), then propagate the state distribution H steps through the transition kernel ONLY (no emissions).
- The NK transition needs d_gc of future fixes, which prediction mode can't see → substitute expected displacement = speed_{T_warm}·Δt (constant-velocity displacement estimate). Hacky but defensible; document it.
- Report tolerant Hit@1 at horizons {1, 5, 15} against the same convention as `eval_stage2.py`. This row is the honest classical prediction floor the RSSM must beat to justify P2.

**Metrics** (same definitions as existing evals): tolerant Hit@1 (chosen candidate's true d_perp ≤ d1+5m), hard Hit@1, MHE. Route-level agreement between methods reported as descriptive statistics only.

**Ground-truth caveat (do not bury in the thesis):** there is NO human GT anywhere; methodology.md §1.3's convention is FMM/HMM pseudo-GT. Headline metric is per-fix tolerant Hit@1 (defined purely from geometry — no circularity); route-level "accuracy vs variant-(i) paths" is descriptive, since variant (i) defines the reference.

**Report matrix:** {(i), (ii), (iii)} × {Porto held-out, T-Drive} × {tolerant Hit@1, hard Hit@1, MHE} for matching; {(i) predict-mode} × horizons {1,5,15} vs RSSM step21000's hit@{1,5,15} (0.644/0.559/0.598, from `evaluation_findings.md`) for prediction. Include Stage-1's per-fix masked tolerant Hit@1 (0.823 holdout) for context, clearly labeled as the **30%-mask-regime** number — it is NOT the correct no-transition floor for variant (iii), which runs geo_mask ALL-True (100% masked, a regime Stage-1 never trained on, with no neighbor geometry visible either). Compute the true variant-(iii) floor as Stage-1's per-fix argmax under geo_mask-all-True on the same held-out split, and compare the variant-(iii) HMM against THAT — beating it is what quantifies the transition model's added value.

## B2 — pseudo-label GRU sequence scorer (CONTINGENT — build ONLY if Run 3 fails Gate R2)

- **Labels:** variant-(i) Viterbi paths on the Porto training split. Machine-derived from unlabeled GPS + map → P3 preserved.
- **Model:** 1-layer GRU (hidden 128) over [Stage-1 z1_t ‖ candidate features]; per-step candidate-scoring head (reuse the RoadScorer form); CE against the Viterbi label. ~1–2M params — trains on the local 4060 or one Kaggle session.
- **Purpose:** tests whether ANY learned transition model beats hand-crafted NK transitions before re-litigating world models (the user-proposed "small supervised sequence scorer" step); doubles as the thesis's P1a+P1b+P3 fallback contribution.
- **Positioning guard applies** (§Positioning above): baseline/fallback framing only.

## Gates

| # | Condition | Action |
|---|---|---|
| C1 | variant (ii) deviates from variant (i) by > 3pp tolerant Hit@1 | Stage-1 export/wiring bug — fix before trusting anything downstream |
| C2 | variant (iii) ≥ variant (i) on Porto held-out | learned context emissions add value over raw geometry → strong thesis material either way |
| C3 | variant (iii) ≥ variant (i) on T-Drive | learned emissions transfer OOD → headline claim for the P1a/P1b chapters |
| C4 | predict-mode Hit@1@15 vs RSSM's 0.598: HMM ≥ RSSM | the world model adds nothing over a first-order transition kernel at H=15 → strengthens the Gate-R3 negative-result writeup; HMM ≪ RSSM → P2's value is real even while its gates fail — central evidence either way |

## Compute

CPU-only, hours-scale, local machine — zero Kaggle GPU quota. Porto full 1.7M trips NOT needed: held-out 500 (eval) + dev 50 (β sweep) + the already-cached 200k subset (contingent B2 training only).
