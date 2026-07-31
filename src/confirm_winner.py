import copy
import torch
import tune_reward_weights as T

print("start", flush=True)
road_z = T.ev._road_z()
print("road_z OK", flush=True)
stage1 = T.ev._stage1(road_z.size(1))
print("stage1 OK", flush=True)
rssm, heads = T.load_world_model(str(T.WM_CKPT), road_z.size(1), T.DEVICE)
print("wm OK", flush=True)

winner = "head_only"
ckpt = torch.load(T.CKPT_DIR / f"tune_{winner}.pt", map_location=T.DEVICE, weights_only=True)
heads_final = copy.deepcopy(heads)
heads_final.reward.load_state_dict(ckpt["reward_head"])
actor_final = T.RoadScorer(h_dim=rssm.h_dim, road_dim=road_z.size(1))
actor_final.load_state_dict(ckpt["actor"])
actor_final.eval()
print("actor OK", flush=True)

T.ev.N_TRAJS = T.FINAL_EVAL_TRAJS
final_loader = T.ev._loader()
print("loader OK", flush=True)

r_final = T.ev.eval_checkpoint(rssm, heads_final, actor_final, stage1, road_z, final_loader)
print(f"FINAL {winner} weights={ckpt['weights']} match@1={r_final['match_hit1']:.4f} "
      f"hit1={r_final['hit1']:.4f} kl={r_final['kl']:.3f}", flush=True)

road_head_baseline = T.ev.eval_checkpoint(rssm, heads, heads.road, stage1, road_z, final_loader)
print(f"ROADHEAD match@1={road_head_baseline['match_hit1']:.4f}", flush=True)
