# CLAUDE.md

> Persistent context for agentic sessions on map matching + path prediction. Read in full before classifying new literature, proposing architecture changes, or logging design decisions.

## 1. Project Identity

- **Working title:** AI-Driven Map Matching and Path Prediction on Semantically Enriched Road Networks
- Role: You are a senior post-doc assisting with the BMW Foresight thesis on AI-driven map matching.

## 2. Core Claim

A single architecture combining four components no prior work combines together: a transformer encoder over the road graph, learned semantic vectorization of road attributes, a latent-world-model RL agent that sequentially "drives" through a learned simulation to pick road segments, and label-free training directly on raw, unaligned fleet GPS.

## 3. Canonical Vocabulary — The Four Pillars

Fixed vocabulary for this project. Use these IDs verbatim in literature notes, design-decision logs, and commit messages — don't paraphrase or rename them.

| ID | Pillar | Definition |
|---|---|---|
| **P1a** | Transformer/Attention Encoder | Processes spatiotemporal sequences over the road graph; replaces RNN-style sequence models. |
| **P1b** | Learned Semantic Vectorization | Vectorizes road attributes (type, POIs) and topology as graph tokens — *learned*, not symbolic/rule-based. |
| **P2** | Latent-World-Model RL | An RL agent sequentially selects road segments inside a learned, compressed latent simulation of the network — not one-shot prediction. |
| **P3** | Label-Free Training | Self-supervised pretraining + physical-constraint rewards; no human-annotated or pre-matched ground truth. |

**The gap this thesis fills:** the unoccupied **P1a + P1b + P2 + P3** corner. No existing system — including the closest RL map matchers and the closest latent-world-model RL systems — covers all four at once (see §6).

## 4. Baseline Comparison Matrix

| Component | DeepMM | RouteKG | Ayara Symbolic BMW Lineage |
|---|---|---|---|---|
| Input representation | Raster grid cells | Road segment sequences | Format-specific ontologies (NDS, HERE, HLMo) | Vectorized road-network graph + projected trajectory fixes |
| Sequence model | RNN (LSTM/GRU) | Hybrid (KG embeddings + sequence encoder) | None — rule-based spatial stream reasoning | Transformer encoder (TAT-Enc style) |
| Training signal | Supervised, synthetic augmented trajectories | Supervised, matched route sequences | None — hand-crafted OWL 2 RL rules + SPARQL | Label-free (self-supervised + RL reward) |
| Graph integration | None (grid) | Knowledge Graph (KG completion) | RDF triples + Datalog rules (OWL semantics) | Graph Attention Network (GAT) over road topology |
| Decision paradigm | One-shot seq2seq | One-shot seq2seq (top-K ranking) | Symbolic reasoning (SPARQL via RDFox) | Sequential RL inside learned latent world model |
| Pillars covered | P3* (via augmentation only) | P1b | P1b (symbolic only) | **P1a + P1b + P2 + P3** |

## 5. Per-Baseline Gap Notes

### DeepMM (Zhao et al., 2019 / TMC 2020)
Closest deep latent-space matching precedent — maps low-quality GPS to road sequences in a learned latent embedding.
- Raster grid loses topological constraints (connectivity, turn restrictions) that a vectorized graph preserves.
- RNN seq2seq can't capture long-range dependencies the way attention can.
- One-shot decoding only — no RL agent, no latent transition world model.
- Supervised on synthetic GraphHopper-derived trajectories with added noise, not raw unaligned fleet GPS.

### RouteKG (Tang et al., 2023 / T-ITS 2025)
State-of-the-art for KG-based road-network semantics in route prediction.
- Frames route prediction as KG completion + sequence encoder — no transformer over a vectorized graph.
- Requires matched route sequences (supervised); this thesis learns from unmatched raw GPS.
- One-shot seq2seq top-K ranking — no RL, no latent transition model.

### Ayara's Symbolic BMW Lineage (2019–2023; Ayara + Glimm, Univ. Ulm)
Five publications defining BMW's internal semantic-map representation and dynamic map-stream processing (forward/backward spatial windows for moving vehicles).
- Entirely symbolic: OWL 2 RL, RDF triples, Datalog rules, SPARQL via the RDFox reasoner — no neural model at all.
- Doesn't do map matching — checks map quality or does rule-based trip inference (grouping coordinates into trips), not snapping coordinates to segments.
- **The bridge:** the most recent paper in the lineage (Qiu, Ayara & Mühlbauer, KEOD 2023) names integrating ML and reasoning as the explicit next step on their roadmap. This thesis is the learned, vectorized, transformer-based counterpart that fills that named gap directly — notable given Ayara is the industry supervisor.

## 6. Adjacent-but-Distinct Systems — Do Not Conflate

- **RLOMM (2025), MIDIRL (2024)** — RL-based map matchers, but operate over explicit candidate states / hand-engineered feature MDPs and require ground-truth matched paths or demonstrations. Missing P3 (label-free) and the *latent* world-model property of P2.
- **Think2Drive (2024), Bench2Drive (2024)** — genuine latent-world-model RL, but confined to simulated driving control in CARLA, outputting steering/throttle rather than discrete road-segment selection from real fleet GPS. Missing P1b and the GPS/road-graph task framing entirely.

This thesis's P2 specifically ports Think2Drive-style latent-world-model machinery to discrete road-graph navigation — that combination, not either piece alone, is the novelty.]

## 7. Working Conventions

**Literature classification**
- Classify any new paper against P1a/P1b/P2/P3 using §3's definitions; slot real baseline contenders into the matrix in §4.
- A paper claiming 3+ pillars together is a high-priority flag — surface it immediately rather than queuing it, since it could erode the gap claim in §6.
- Keep DeepMM, RouteKG, and the Ayara lineage as the three fixed comparison anchors unless explicitly told to add or change anchors.

**Design decisions**
- Any architectural decision should state which pillar(s) it serves and how it differs from the corresponding cell in §4's matrix.
- Log deviations from the four-pillar framing as explicit decisions with rationale — don't silently overwrite this file.

## 8. Source

Distilled from the "Thesis Framing Brief: Comparative Analysis and Research Gap" document. Treat this CLAUDE.md as the living, agent-facing version — update it as the literature review and architecture evolve rather than re-deriving from the original brief each time.

## 9. Constraint

Always pass offset and limit when reading files larger than 200 lines. Only widen the limit if the symbol you need is not in range