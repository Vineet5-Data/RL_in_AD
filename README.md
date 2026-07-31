# AI-Driven Map Matching and Path Prediction on Semantically Enriched Road Networks

A label-free deep learning pipeline that matches noisy, raw fleet GPS to road segments and predicts future routes — trained with **zero human-annotated ground truth**, using a latent world model over a vectorized road-network graph.

> **Status: practical engineering project**, not a paper defense. The deliverable is a working, deployable matcher/predictor. Every number below is measured, every negative result is kept (not buried) — this is a running experimental log, not a highlight reel.

## What this is

Most fleet-GPS pipelines either (a) snap points to roads with classical geometry (Hidden Markov Models over road candidates), or (b) train a supervised sequence model on pre-matched routes. This project asks: **can a system learn to match GPS to roads, and predict where a vehicle goes next, from raw unmatched trajectories alone** — no human labels, no pre-matched training routes — while still beating the classical geometric baseline?

The answer, after ~3 months of experimentation (Porto taxi GPS, 1.66M trajectories, T-Drive/Beijing + Hannover held out for cross-city testing): **yes, but only when the learned signal is combined with geometry, not used to replace it.**

### Architecture — Four Pillars

| ID | Pillar | What it does |
|---|---|---|
| **P1a** | Transformer/Attention Encoder | Processes spatiotemporal GPS sequences over the road graph (replaces RNN-style sequence models) |
| **P1b** | Learned Semantic Vectorization | Vectorizes road attributes + topology as graph tokens — *learned*, not hand-coded rules |
| **P2** | Latent World-Model RL | An RL-style agent sequentially selects road segments inside a learned, compressed latent simulation of the road network |
| **P3** | Label-Free Training | Self-supervised pretraining + physical-constraint rewards — no human-annotated or pre-matched ground truth anywhere in the loop |

**Delivered:** P1a, P1b, P3. **Partially delivered:** P2 — the world-model *representation* (a DreamerV3-style recurrent state-space model, decoder-light) works and drives the results below; the RL-driven *sequential decision* variant does not beat a simple supervised readout head, and that negative result is treated as final (see "What didn't work").

## Current best system

| Capability | Method | Result |
|---|---|---|
| Offline (batch) map matching | Hybrid Viterbi — world-model road-head log-prob + classical Gaussian geometry emission, classical transitions | **0.868 tolerant Hit@1** (n=500, Porto held-out) |
| Online (streaming) map matching | World-model road head + geometry emission/transition | **0.829 match@1**, n=500 |
| Path prediction | World-model prior rollouts | Beats the classical HMM baseline at every prediction horizon |
| No-GPU / classical fallback | Pure geometric HMM Viterbi | 0.845 offline / 0.59 online — no training required |

Model: a 4.77M-parameter decoder-light recurrent state-space world model (32×32 categorical latent, DreamerV3-lineage), trained on ~1.66M Porto taxi trajectories (~82M GPS fixes) across chained Kaggle T4 GPU sessions.

**Key finding:** the learned world-model signal alone *loses* to pure geometric matching (0.77 vs 0.84 in early ablations) — but it is *complementary*. The hybrid combination is the first learned-signal result in this project that beats geometry outright, and that gain survives contact with the full production pipeline, not just an isolated ablation.

## How it compares

| | This system | DeepMM (Zhao et al., 2019) | RouteKG (Tang et al., 2023) | Symbolic reasoning (Ayara/BMW lineage) |
|---|---|---|---|---|
| Input | Vectorized road-network graph + raw GPS | Raster grid cells | Road segment sequences | Format-specific ontologies (OWL/RDF) |
| Sequence model | Transformer | RNN (LSTM/GRU) | KG embeddings + sequence encoder | None — rule-based |
| Training signal | Label-free (self-supervised + reward) | Supervised, synthetic trajectories | Supervised, matched routes | None — hand-crafted rules |
| Graph integration | Graph Attention Network over topology | None (grid) | Knowledge-graph completion | RDF triples + Datalog |
| Decision paradigm | Sequential decode over a learned latent world model | One-shot seq2seq | One-shot top-K ranking | Symbolic (SPARQL) |

## What didn't work (kept on the record, not hidden)

Negative results here were as decision-relevant as the positives — each one closed a research direction with evidence, not a guess:

- **Faithful reconstruction-based world model (DreamerV3 port): posterior collapse.** The latent representation "freeloaded" on a teacher-forced channel instead of encoding real signal, across 4 experiment rounds. Fix was architectural (decoder-light, remove the reconstruction decoder), not a hyperparameter tweak.
- **RL actor for sequential road selection: reopened, re-closed with stronger evidence.** The RL policy underperforms a simple supervised read-out head for matching accuracy (0.65 vs 0.77+). Root-caused to reward-model overoptimization (Goodhart's law against a frozen, imperfect learned reward) — an early-stopping probe showed *step 0 (no RL training at all) is already the optimum*; every gradient step afterward degrades monotonically. This matches the RLHF reward-overoptimization literature's degenerate boundary case, not a tuning failure.
- **Learned semantic road embeddings do not transfer out-of-distribution.** Porto-trained road semantics lose ~17pp of matching accuracy zero-shot on a different city (Beijing/T-Drive) — consistent across two model generations. The geometric baseline is far more robust to new cities. Practical consequence: hybrid mode is the trained-city configuration; pure geometry is the automatic fallback for unseen cities.
- **Per-city retraining on public third-party GPS traces failed twice** (naive and speed-filtered), and a further isolation experiment (unfreezing the pretrained encoder) made it *worse*, ruling out "frozen encoder mismatch" as the cause — the public traces themselves are too noisy/contaminated for this to work, not a fixable pipeline bug.

## Data-scaling and capacity findings

- Systematic data-scaling ablation (200k → 400k → 800k trajectories, epoch-matched): **+2pp match@1 accuracy per doubling of training data**, no flattening — refuting an earlier assumption that the model was data-saturated.
- Doubling latent capacity (16×16 → 32×32 categorical latent, matching DreamerV3's standard width) delivered a decisive accuracy jump (+2.5–3.3pp) even at 58% of its training schedule — capacity was a real, not cosmetic, bottleneck.
- A genuinely held-out cross-city test (Hannover, Germany — never included in any training mix) showed a multi-city-trained checkpoint generalizing with only a −7pp accuracy gap, versus −17pp for a single-city-trained model on a different held-out city — a promising but not yet fully controlled result (different cities, different training budgets — flagged as a signal, not a proven claim).

## Repository contents

This repository holds the **research record and documentation** for the project: the consolidated project log, the literature base used for competitive/technical grounding, and the standing engineering conventions. The active development codebase (world-model training, HMM baseline, matcher package, evaluation harnesses) lives in local git worktrees under active iteration and is not yet published here.

```
research/
  project_summary.md         the single active project log — current results, full run history, roadmap
  critique_and_next_steps.md adversarial internal review of an early architecture proposal
  archive/                   retired docs, kept verbatim as the historical experimental record
literature_papers/           structured summaries of the papers used for prior-art and technique grounding
```

## Roadmap

- Scripted per-city retraining recipe (elevated in priority after the OOD generalization findings above)
- Latency/footprint benchmark for production deployment
- Widen world-model heads to match capacity now proven to matter (pending final data-scaling readout)
- Multi-city training data mix, already underway, to close the remaining cross-city generalization gap

## License

MIT — see [LICENSE](LICENSE).
