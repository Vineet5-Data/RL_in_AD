# Stage-2 (RSSM) Debugging Notes

## Main issue: road-membership head leak

**Where:** `training/stage2.py`, `rssm_losses()`, the per-timestep unroll loop.

**What happened:** at loop step `t`, the GRU transition used
`road_embed[:, t] = road_z[pseudo_seg[:, t]]` to build `h`, where
`pseudo_seg[:, t]` = nearest real candidate to the GPS fix **at time t**
(`argmin(d_perp)`). The road-membership head then decoded from that same `h`
and was trained to predict `pos_idx[:, t]` -- the **same** argmin over the
**same** candidate list at the **same** t. Input and target were two
representations of one identical measurement at one identical timestep ->
same-timestep tautology, not learned dynamics.

**Symptom:** `road` loss in the training log converged to exact `0.0000` by
~step 400 and stayed there for the rest of a 19,800-step run. Looked like a
"solved" metric; was actually meaningless.

**Fix:** shifted the GRU's road input to the *previous* timestep --
`road_embed[:, t - 1]` instead of `road_embed[:, t]`. `h` at step `t` now only
carries road info resolved *before* t, so predicting `pos_idx[:, t]` from it
is a genuine predict-from-history task.

**Verified:** re-ran real-data sanity post-fix -- `road` now shows a gradual
multi-step decay (`2.27 -> 1.33 -> 0.39 -> 0.21 -> 0.06`) instead of an instant
drop to zero.

**Open, not fully resolved:** on the full 200k-trajectory run, `road` still
converges close to ~0 within ~1000 steps, and its early-step values are
suspiciously close to the pre-fix run's numbers (step50: `0.1051` old vs
`0.1053` new). Working theory: a *second*, largely unavoidable channel --
the posterior `z_t ~ q(z|h, z1_stage1)` conditions on Stage-1's own embedding,
which was trained (92% Hit@1) to be maximally informative about "nearest
candidate" already. That's legitimate VAE behaviour (same channel GPS/speed
heads use), not a bug, but it means `road` may just be an intrinsically easy
sub-task now, independent of the structural leak that was fixed. Not proven
either way -- flagged, not chased further; `gps`/`speed`/`kl` are unaffected
and are the metrics that actually gate RSSM quality.

## Other issues hit while building this pipeline

| Where | Issue | Fix |
|---|---|---|
| `training/stage0.py`, `stage1.py` | Windows segfault: `torch` imported before `pyarrow` poisons arrow's native libs on parquet read | `import pyarrow.parquet` before `import torch` |
| `dataset/trajectories.py` | Per-fix `shapely` candidate query re-ran every epoch (Stage-1 stuck slow, ~5600 pts/s) | `CandidateIndex.query_batch()` -- one vectorized bulk STRtree query, cached to `.npz`, ~8.5x faster + runs once not per-epoch |
| `training/stage2.py` | No checkpoint resume, despite methodology Sec.5 saying "Resume from checkpoint mandatory" | ckpts now save `opt`/`sched`/`step`; `--resume <ckpt>` restores exact training state (verified: `lr` matched bit-for-bit across a kill/resume pair) |
| `models/world_model.py` | KL dipped to ~0.006 at step 50-150 of the first real run (near posterior collapse) before self-recovering | added DreamerV3 unimix (1% uniform floor on every categorical) as a structural guard |
