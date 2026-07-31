# CLAUDE.md

> Persistent context for agentic sessions on map matching + path prediction. Read in full before classifying new literature, proposing architecture changes, or logging design decisions.
>
> **Reframed 2026-07-11 (user decision): this is a PRACTICAL PROJECT** The deliverable is a working map-matching + path-prediction system, not chapters or a novelty-gap defense. Historical docs keep their original thesis-era wording as the experimental record — do not retro-edit them. Layout (consolidated + keep-what-worked sweep 2026-07-11): `research/project_summary.md` = the ONE active doc (its §6 = workspace layout); `research/archive/` = full record, unedited (retired docs + `worktree_artifacts/`). Active worktrees — all load-bearing, do not retire: `research2` (WM + `matcher.py` package), `HMM_baseline`, `data-preprocess`, `kaggle` (ckpt store). Retired worktrees (`evaluation`, `Stage2_kaggle`, `stage2_new`): checkouts removed, code preserved on their git branches; failed-track artifacts in `research/archive/worktree_artifacts/` and `.worktrees/research2/archive/`.

## 1. Project Identity

- **Working title:** AI-Driven Map Matching and Path Prediction on Semantically Enriched Road Networks
- Role: You are a senior ML engineer building a deployable map-matching and path-prediction pipeline from raw, unaligned fleet GPS.
- Success is measured by product metrics — matching accuracy, prediction accuracy, latency, generalization to new cities, retraining cost, novelty claims.

## 2. Current System (what actually works, 2026-07-11)

| Capability | Best component | Number | Where |
|---|---|---|---|
| Offline (batch) map matching | **Hybrid Viterbi**: WM road-head emission + NK Gaussian geometry emission, NK transitions | **0.8655** tolerant Hit@1 (project best; pure NK-HMM 0.8447) | `.worktrees/research2/eval_offline_viterbi.py` + `.worktrees/HMM_baseline/hmm_baseline/baseline_hmm.py` |
| Online (streaming) map matching | WM road head + NK emission/transition ("hyb+topo", 2026-07-11) | **0.8122** match@1, 5.9% disconnected jumps (bare head: 0.7684 / 14.9%) | `.worktrees/research2/matcher.py` (Run-2-XL final ckpt) |
| Path prediction | WM prior rollouts | beats HMM β=20 at all horizons (hit@5/15 every ckpt; hit@1 0.60-0.63 band) | same |
| Classical fallback (no GPU, no training) | NK-HMM Viterbi β=20 | 0.8447 offline / 0.5918 online | `.worktrees/HMM_baseline/hmm_baseline/` |

Known negatives (recorded, don't re-litigate): reconstruction-RSSM (faithful DreamerV3 port) posterior-collapses on this modality — decoder-light is the working recipe; RL actor underperforms the road head for matching (0.593, one optional bounded reweighting attempt remains); Porto-learned semantic context did not transfer OOD for the HMM variant (T-Drive −21.9pp) — the Run-2-XL WM has NOT yet been tested OOD.

Roadmap + next steps: `research/project_summary.md` §5 (full detail in `research/archive/practical_roadmap.md`).

## 3. Canonical Vocabulary — The Four Pillars

Fixed architecture vocabulary. Use these IDs verbatim in notes, design-decision logs, and commit messages — don't paraphrase or rename them.

| ID | Pillar | Definition |
|---|---|---|
| **P1a** | Transformer/Attention Encoder | Processes spatiotemporal sequences over the road graph; replaces RNN-style sequence models. |
| **P1b** | Learned Semantic Vectorization | Vectorizes road attributes (type, POIs) and topology as graph tokens — *learned*, not symbolic/rule-based. |
| **P2** | Latent-World-Model RL | An RL agent sequentially selects road segments inside a learned, compressed latent simulation of the network — not one-shot prediction. |
| **P3** | Label-Free Training | Self-supervised pretraining + physical-constraint rewards; no human-annotated or pre-matched ground truth. |

Empirical status: P1a/P1b/P3 delivered; P2 split — world-model representation + prediction work (decoder-light), RL-driven segment selection does not (yet) beat the supervised road-head readout.

## 4. Baseline Comparison Matrix (competitive landscape)

| Component | DeepMM | RouteKG | Ayara Symbolic BMW Lineage | This system |
|---|---|---|---|---|
| Input representation | Raster grid cells | Road segment sequences | Format-specific ontologies (NDS, HERE, HLMo) | Vectorized road-network graph + projected trajectory fixes |
| Sequence model | RNN (LSTM/GRU) | Hybrid (KG embeddings + sequence encoder) | None — rule-based spatial stream reasoning | Transformer encoder (TAT-Enc style) |
| Training signal | Supervised, synthetic augmented trajectories | Supervised, matched route sequences | None — hand-crafted OWL 2 RL rules + SPARQL | Label-free (self-supervised + RL reward) |
| Graph integration | None (grid) | Knowledge Graph (KG completion) | RDF triples + Datalog rules (OWL semantics) | Graph Attention Network (GAT) over road topology |
| Decision paradigm | One-shot seq2seq | One-shot seq2seq (top-K ranking) | Symbolic reasoning (SPARQL via RDFox) | Sequential decode over learned latent world model (hybrid Viterbi in batch mode) |
| Pillars covered | P3* (via augmentation only) | P1b | P1b (symbolic only) | **P1a + P1b + P2 + P3** |

## 5. Per-Baseline Notes

### DeepMM (Zhao et al., 2019 / TMC 2020)
Closest deep latent-space matching precedent — maps low-quality GPS to road sequences in a learned latent embedding.
- Raster grid loses topological constraints (connectivity, turn restrictions) that a vectorized graph preserves.
- RNN seq2seq can't capture long-range dependencies the way attention can.
- One-shot decoding only — no RL agent, no latent transition world model.
- Supervised on synthetic GraphHopper-derived trajectories with added noise, not raw unaligned fleet GPS.

### RouteKG (Tang et al., 2023 / T-ITS 2025)
State-of-the-art for KG-based road-network semantics in route prediction.
- Frames route prediction as KG completion + sequence encoder — no transformer over a vectorized graph.
- Requires matched route sequences (supervised); this system learns from unmatched raw GPS.
- One-shot seq2seq top-K ranking — no RL, no latent transition model.

### Ayara's Symbolic BMW Lineage (2019–2023; Ayara + Glimm, Univ. Ulm)
Five publications defining BMW's internal semantic-map representation and dynamic map-stream processing (forward/backward spatial windows for moving vehicles).
- Entirely symbolic: OWL 2 RL, RDF triples, Datalog rules, SPARQL via the RDFox reasoner — no neural model at all.
- Doesn't do map matching — checks map quality or does rule-based trip inference (grouping coordinates into trips), not snapping coordinates to segments.
- Relevance now: prior-art context and a potential integration target (their KEOD 2023 paper names ML+reasoning integration as their roadmap), not a gap to defend.

## 6. Adjacent-but-Distinct Systems — Do Not Conflate

- **RLOMM (2025), MIDIRL (2024)** — RL-based map matchers, but operate over explicit candidate states / hand-engineered feature MDPs and require ground-truth matched paths or demonstrations. Missing P3 (label-free) and the *latent* world-model property of P2.
- **Think2Drive (2024), Bench2Drive (2024)** — genuine latent-world-model RL, but confined to simulated driving control in CARLA, outputting steering/throttle rather than discrete road-segment selection from real fleet GPS. Missing P1b and the GPS/road-graph task framing entirely.

Keep tracking these as the competitive set: a system covering 3+ pillars is a direct competitor — surface it immediately.

## 7. Working Conventions

**Literature / competitor classification**
- Classify any new paper or system against P1a/P1b/P2/P3 using §3's definitions; slot real contenders into §4's matrix.
- Keep DeepMM, RouteKG, and the Ayara lineage as the three fixed comparison anchors unless explicitly told to add or change anchors.

**Design decisions**
- Any architectural decision should state which product metric it serves (accuracy / latency / OOD generalization / retraining cost) and which pillar(s) it touches.
- The thesis-era pre-registered-gate discipline is relaxed to cost/benefit: iteration is allowed when the expected product improvement justifies the compute, but every run still gets a written success bar BEFORE launch (that habit caught every bug so far).
- Log deviations from the four-pillar framing as explicit decisions with rationale — don't silently overwrite this file.
- Compute permission (user decision 2026-07-12, INVERTED from the old rule): **Kaggle launches need NO permission** — full go-ahead, **run them freely on T4x2 GPU ONLY**. Two conditions on any Kaggle run: (1) keep it resume/pause-able across sessions (Kaggle ~12h cap → use the trainer's `--max-hours`+`--resume`/`--ckpt-every` machinery so session-2+ can continue); (2) notify the user to save the session output, or save the output artifacts yourself. **Local runs (RTX 4060) now DO need permission — ask first.**

## 9. Constraint

Always pass offset and limit when reading files larger than 200 lines. Only widen the limit if the symbol you need is not in range
