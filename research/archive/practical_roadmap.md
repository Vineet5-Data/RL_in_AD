# Practical Roadmap (reframed from thesis, 2026-07-11)

> Deliverable: a usable map-matching + path-prediction system. Numbers below are the Run-2-XL / Track-B state of the art (see `critique_and_next_steps.md` §6.3). Ordered by product value per unit effort.

## 1. Consolidate the winning pipeline into one package — FIRST
The best matcher (hybrid Viterbi 0.8655) currently lives as an eval script stitched across two worktrees (`research2` models + `HMM_baseline` Viterbi machinery) with sys.path surgery. Productize:
- One repo/package: `matcher/` with `match_offline(traj) -> segments` (hybrid Viterbi), `match_online(fix_stream) -> segment` (WM road head), `predict(traj, horizon) -> segments` (WM prior rollout).
- Bundle: Run-2-XL final ckpt, stage0/stage1 ckpts, per-city graph artifacts (segments.parquet, line_graph_edges.parquet), candidate-cache builder.
- CLI + minimal API. Effort: days, no training.

## 2. OOD generalization test — TOP RISK, do before promising multi-city
HMM's Porto-learned semantic emission dropped −21.9pp on T-Drive (Beijing). The Run-2-XL WM has NEVER been tested off-Porto. If the WM emission collapses OOD, the practical product for a new city is NK-HMM (0.8447-class) until retrained. Run the existing eval harness on T-Drive candidates (local GPU, zero training). Decision point: per-city retraining recipe vs city-agnostic claims.

## 3. Latency + footprint benchmark
Offline: full 500-traj hybrid Viterbi ran ~105 s on RTX 4060 (≈5 traj/s incl. WM pass + memoised Dijkstra) — fine for batch fleets; measure properly, profile RouteDist cache growth. Online: per-fix WM step cost unmeasured — required number for any streaming claim. CPU-only inference path worth one test (batch=1 GRU is small).

## 4. Retraining recipe (per-city onboarding)
Document + script the full chain: OSM graph build → stage0/stage1 self-supervised pretrain → stage2r decoder-light WM (60k steps ≈ 2 Kaggle T4 sessions with `--max-hours` guard + `--resume`). This is the product's "new city" cost. Currently reconstructable only from scattered worktree knowledge.

## 5. Optional: one bounded actor-reward reweighting attempt
Unchanged from §4 actor rule (frozen WM, single shot, bar = match@1 within 3pp of 0.7684). Only worth it if a product case needs sequential decision-making beyond argmax decode (e.g., beam alternatives, controllable trade-offs). Otherwise park.

## Explicitly deprioritized (was thesis work)
- Thesis chapter writing (§5 thesis-writing steps 1-3 in `critique_and_next_steps.md`) — void.
- Honest-negative narrative polish — keep the record, spend no further effort on it.
- Novelty-gap defense vs literature — §4/§6 of CLAUDE.md now serve as competitive landscape only.
