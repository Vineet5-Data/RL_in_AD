"""One-off eval of the 32x8-latent kl_act probe checkpoint (CPU-trained on Kaggle,
step 4805/6000, --max-hours 8.5 reached). Same shape total-dim as 16x16 (32*8=256)
but same GROUP COUNT as 32x32 -- isolates groups-vs-classes-per-group as the
variable in the kl_act capacity investigation (project_summary.md §3, Agent A's
per-group-density finding). Bypasses eval_research2.py's _checkpoint_table() (which
only scans research2/ckpt/, where the production P2 32x32 ckpt lives) to point
directly at the downloaded probe file without touching that directory.

Run (from src/evals):
  python eval_klact_32x8_cpu.py <path-to-stage2r_porto.pt>
"""
import sys
from pathlib import Path

import eval_research2 as ev

ckpt_path = Path(sys.argv[1])
road_z = ev._road_z()
stage1 = ev._stage1(road_z.size(1))
loader = ev._loader()
rssm, heads = ev._load_wm(ckpt_path, road_z.size(1))
print(f"[eval] 32x8 kl_act probe  ckpt={ckpt_path}  groups={rssm.groups} classes={rssm.classes}")
r = ev.eval_checkpoint(rssm, heads, heads.road, stage1, road_z, loader)
print(f"match@1 {r['match_hit1']:.4f}  cl_gps {r['cl_gps']:.4f}  kl {r['kl']:.3f}  ent {r['ent']:.2f}  "
      f"hit@1 {r['hit1']:.3f}  hit@5 {r['hit5']:.3f}  hit@15 {r['hit15']:.3f}")
