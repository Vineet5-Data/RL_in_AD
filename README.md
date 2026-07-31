# RL/IL-Driven Map Matching and Path Prediction

## Overview
This project uses RL/IL to solve two core navigation problems for vehicle fleets (such as delivery trucks or taxis):
1. **Map Matching:** Converting noisy, imprecise raw GPS coordinates into the exact street segments a vehicle actually drove on.
2. **Path Prediction:** Predicting where the vehicle is heading next based on its current route.

### What Makes This Different?
Traditional map-matching systems usually rely strictly on geometry (snapping GPS points to the nearest road using Hidden Markov Models) or require millions of human-labeled routes to train an AI model.

This project trains an RL/IL to understand road networks and vehicle movement **without any human labels or pre-matched training data** (label-free self-supervised learning), while using a latent world model over a vectorized road-network graph.

> **Project Goal:** This is a **practical engineering log**, not an academic paper defense. It includes every real-world test result—including negative results—to document what actually works in production.

---

## Key Findings: RL/IL vs. Math

After testing on over 1.66 million Porto taxi routes (plus cross-city evaluation on Beijing and Hannover datasets), the core finding is clear:

* **RL alone loses to standard math:** A purely RL-driven model made more map-matching errors than standard geometry-based algorithms.
* **RL + Math wins:** Combining RL predictions with classic geometric algorithms produces a hybrid system that outperforms every single baseline.

### System Performance Breakdown

| Capability / Method | Algorithm Details | Accuracy / Result |
| :--- | :--- | :--- |
| **Offline (Batch) Map Matching** | **Hybrid Viterbi** (World-model road-head + Gaussian geometry emission/transition) | **86.8%** (Porto held-out) |
| **Online (Streaming) Map Matching** | **World-Model Road Head** + Geometry emission/transition | **82.9%** (0.829 match@1) |
| **Path Prediction** | World-model prior rollouts | **Beats HMM baseline** across all prediction horizons |
| **No-GPU / Fallback Mode** | Pure geometric HMM Viterbi | **84.5% offline / 59.0% online** (zero training required) |

*Model spec: 4.77M-parameter decoder-light recurrent state-space world model (32×32 categorical latent, DreamerV3-lineage) trained on ~1.66M Porto taxi trajectories (~82M GPS fixes).*

---

## Architecture: Four Pillars

| Pillar | Technology | Functional Description |
| :--- | :--- | :--- |
| **P1a: Sequence Encoder** | Transformer / Attention | Processes spatiotemporal GPS sequences directly over the road graph, replacing traditional RNNs. |
| **P1b: Semantic Vectorization** | Graph Attention Network (GAT) | Vectorizes road attributes and topology into graph tokens (learned automatically rather than using hand-coded rules). |
| **P2: Latent World Model** | Recurrent State-Space Model (RSSM) | Simulates vehicle movement inside a compressed latent representation of the road network. |
| **P3: Label-Free Training** | Self-Supervised + Physical Rewards | Learns patterns from scratch without needing any human annotations or pre-matched ground-truth routes. |

---

## Key Lessons & Negative Results (What Didn't Work)

To maintain an authentic engineering log, failed experiments are documented rather than hidden:

* **Faithful Reconstruction World Model (Posterior Collapse):** A standard DreamerV3 port suffered from posterior collapse where the latent representation bypassed learning actual spatial signals. **Fix:** Replaced with a *decoder-light* architecture.
* **RL Agent for Sequential Road Selection:** An RL policy underperformed a simple supervised readout head (65% vs 77%+ accuracy). Root cause was reward-model overoptimization (Goodhart's Law). Early-stopping probes showed step 0 (zero RL fine-tuning) was optimal.
* **Out-of-Distribution (Cross-City) Transfer:** A model trained exclusively on Porto lost ~17 percentage points in accuracy when tested zero-shot on Beijing. Geometric math remains far more robust to new environments.
  * **Production Strategy:** Use **Hybrid Mode** for trained cities and automatically fail back to **Pure Geometry** for unseen cities.
* **Retraining on Public Traces:** Per-city retraining on raw public GPS traces failed twice due to heavy noise and data contamination.

---

## Data-Scaling and Model Capacity Insights

* **Data Scaling (+2pp per 2x data):** Scaling training data (200k → 400k → 800k trajectories) yielded a consistent **+2 percentage point gain in accuracy per doubling** with no flattening observed.
* **Latent Capacity (16×16 → 32×32):** Doubling latent categorical dimensions delivered an immediate **+2.5 to +3.3pp accuracy boost**, proving capacity was a primary bottleneck.
* **Multi-City Generalization:** Multi-city pretraining reduced the cross-city transfer accuracy drop from **-17pp down to -7pp** on held-out test cities (e.g., Hannover).

---

## Repository Structure

```text
src/
  matcher.py, test_matcher.py       # Production matching/prediction API (offline + online + predict)
  models/                           # World model (decoder-light RSSM), GPS & road encoders
  training/                         # Training pipeline stages (stage 0-3)
  hmm_baseline/                     # Classical geometric HMM Viterbi fallback (CPU-only, no training)
  dataset/, preprocessing/, roadgraph/ # Data cleaning, road graph construction, candidate retrieval
  evals/                            # Evaluation harnesses and diagnostic probes
  tests/                            # Unit tests for data/graph processing
research/
  project_summary.md                # Active project log, full execution history, roadmap
  critique_and_next_steps.md        # Internal adversarial review of architecture proposals
  archive/                          # Historical experiment records
literature_papers/                  # Structured summaries of foundational research papers
```

---

## Future Roadmap

- [ ] Automate a per-city retraining script pipeline.
- [ ] Benchmark real-time latency and footprint metrics for edge/cloud deployment.
- [ ] Expand world-model head capacity following data-scaling findings.
- [ ] Train on a larger multi-city dataset mix to close remaining zero-shot generalization gaps.

---

## License

[MIT License](LICENSE
