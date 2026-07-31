"""Eval the hybrid-cities (multi-city) session-1 checkpoint (porto+tdrive+geolife_car+
cabspotting+romataxi, 32x32 latent, step 13217/41500 -- --max-hours 8.5 cap reached
mid-run, not a completed ladder rung) against the written bar in project_summary.md:
does it beat the Porto-only P2 32x32 ceiling on Porto's own held-out split, and/or
meaningfully close the T-Drive OOD gap (zero-shot Porto-WM currently -17.0pp)?

Bypasses eval_research2.py's _checkpoint_table() (scans research2/ckpt/ only) to
point directly at the downloaded multi-city ckpt, same pattern as
eval_klact_32x8_cpu.py. Runs both the standard Porto in-domain eval and the T-Drive
OOD eval against the SAME checkpoint in one pass for direct comparison.

Run (from .worktrees/research2):
  python eval_multicity_step13217.py <path-to-stage2r_multi_....pt>
"""
import sys
from pathlib import Path

import eval_research2 as ev
import eval_ood_tdrive as ood

ckpt_path = Path(sys.argv[1])

print(f"[eval] multi-city ckpt={ckpt_path}  (porto+tdrive+geolife_car+cabspotting+romataxi, step 13217/41500)\n")

print("=== Porto in-domain (held-out split, same protocol as production P2 32x32) ===")
road_z_p = ev._road_z()
stage1_p = ev._stage1(road_z_p.size(1))
loader_p = ev._loader()
rssm, heads = ev._load_wm(ckpt_path, road_z_p.size(1))
print(f"[eval] groups={rssm.groups} classes={rssm.classes}")
r_p = ev.eval_checkpoint(rssm, heads, heads.road, stage1_p, road_z_p, loader_p)
print(f"match@1 {r_p['match_hit1']:.4f}  cl_gps {r_p['cl_gps']:.4f}  kl {r_p['kl']:.3f}  ent {r_p['ent']:.2f}  "
      f"hit@1 {r_p['hit1']:.3f}  hit@5 {r_p['hit5']:.3f}  hit@15 {r_p['hit15']:.3f}")
print("(compare: production P2 32x32 ceiling match@1 0.8371 in-domain @58% train, plateau band 0.8297-0.8433)\n")

print("=== T-Drive OOD (zero-shot road graph, same protocol as eval_ood_tdrive.py) ===")
road_z_t = ood._road_z_tdrive()
stage1_t = ev._stage1(road_z_t.size(1))
loader_t = ood._loader_tdrive(None)
rssm2, heads2 = ev._load_wm(ckpt_path, road_z_t.size(1))
r_t = ev.eval_checkpoint(rssm2, heads2, heads2.road, stage1_t, road_z_t, loader_t)
print(f"match@1 {r_t['match_hit1']:.4f}  cl_gps {r_t['cl_gps']:.4f}  kl {r_t['kl']:.3f}  ent {r_t['ent']:.2f}  "
      f"hit@1 {r_t['hit1']:.3f}  hit@5 {r_t['hit5']:.3f}  hit@15 {r_t['hit15']:.3f}")
print(f"(compare: Porto-only P2 32x32 zero-shot T-Drive OOD match@1 0.6567, in-domain 0.8371, -17.0pp gap)")
print(f"multi-city in-domain->OOD gap this ckpt: {r_p['match_hit1'] - r_t['match_hit1']:+.4f} ({(r_p['match_hit1'] - r_t['match_hit1'])*100:+.1f}pp)")
