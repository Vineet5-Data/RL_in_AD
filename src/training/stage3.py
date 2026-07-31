"""research2 Stage-3: Dreamer-style actor-critic inside the frozen decoder-light
world model (methodology 2.4 + Stage 3; methodology-comparison Sec.4: this RL
formulation is retained UNCHANGED -- categorical actor over masked candidates,
lambda-return critic, imagined rollouts; PPO/DQN/SAC explicitly rejected there).

World model (RSSM2 + HeadsLight from training.stage2r) is FROZEN -- methodology
Stage 3: "No new real data -- all training inside world model". Real batches are
used only to (a) produce teacher-forced posterior start states and (b) provide
the REAL candidate sets at future timesteps ("which roads exist nearby" -- the
deployment-realistic input convention established by stage2_correction Change 5),
never d_perp/pos_idx labels.

Action space: the per-fix masked candidate set (radius 50m, K=10), NOT the pure
max-degree-8 topological-successor set of methodology 2.4. FLAGGED DEVIATION:
methodology 2.4 (hard successor mask) and 2.5 (r_topo REWARDS successor
membership -- constant 1 under a hard mask, i.e. contradictory) are mutually
inconsistent; resolved in favor of the candidate-set convention because (1) it
makes r_topo meaningful exactly as spec'd, (2) it is the convention the entire
existing eval/HMM-baseline comparison stack uses (tolerant Hit@1 over the same
candidate sets -- comparability requirement), (3) topological continuity is
still enforced, softly, through the lambda_topo=2.0 reward ("hardest physical
constraint" gets the largest weight, per 2.5).

Rewards inside imagination come from the frozen reward head (trained in stage2r
against the Sec.2.5 physics rewards) -- no real GPS is consulted during rollouts.

Actor warm start: methodology Stage 3 "Initialize actor from road membership
logits decoder head" -- the actor IS a RoadScorer, loaded from heads.road.

Run:   PYTHONPATH=../data-preprocess python -m training.stage3 --wm-ckpt ckpt/stage2r_porto.pt ...
Smoke: python -m training.stage3 --smoke
"""

from __future__ import annotations

# pyarrow must load before torch (Windows segfault guard, see training/stage0.py)
import pyarrow.parquet  # noqa: F401

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from models.world_model2 import (RSSM2, HeadsLight, RoadScorer, Critic,
                                      straight_through_sample, lambda_return, symexp)
    from training.stage2 import LengthCapped, _stage1_encoder
    from training.stage2r import extract_features_r
    from training.stage1 import _road_embeddings, build_loader
except ImportError:  # flat layout (Kaggle CODE_ROOT on sys.path)
    from world_model2 import (RSSM2, HeadsLight, RoadScorer, Critic,  # type: ignore
                               straight_through_sample, lambda_return, symexp)
    from stage2 import LengthCapped, _stage1_encoder  # type: ignore
    from stage2r import extract_features_r  # type: ignore
    from stage1 import _road_embeddings, build_loader  # type: ignore


def load_world_model(ckpt_path: str, road_dim: int, device) -> tuple[RSSM2, HeadsLight]:
    """Frozen decoder-light world model from a stage2r checkpoint."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    meta = sd.get("meta", {})
    norm = meta.get("norm", "batch")
    groups, classes = meta.get("stoch_groups", 16), meta.get("stoch_classes", 16)
    rssm = RSSM2(obs_dim=road_dim, road_dim=road_dim, norm=norm,
                 groups=groups, classes=classes).to(device)
    rssm.load_state_dict(sd["rssm"])
    heads = HeadsLight(h_dim=rssm.h_dim, stoch_dim=groups * classes, road_dim=road_dim).to(device)
    heads.load_state_dict(sd["heads"])
    for m in (rssm, heads):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return rssm, heads


def posterior_states(rssm: RSSM2, feats: dict, road_z: torch.Tensor):
    """Teacher-forced pass over the real sequence -> posterior states at every t.
    Conditioning: previous resolved (pseudo) segment, same shift as training.
    -> H_all[B,L,h_dim], Z_all[B,L,stoch_dim] (no grad; WM frozen)."""
    z1, road_embed, valid = feats["z1"], feats["road_embed"], feats["valid"]
    B, L = valid.shape
    device = z1.device
    h, z = rssm.initial(B, device)
    hs, zs = [], []
    with torch.no_grad():
        for t in range(L):
            if t > 0:
                h = rssm.step(h, z, road_embed[:, t - 1])
            post = rssm.posterior(h, z1[:, t])
            z = straight_through_sample(post).reshape(B, -1)
            hs.append(h)
            zs.append(z)
    return torch.stack(hs, 1), torch.stack(zs, 1)


def imagine_and_losses(rssm, heads, actor, critic, critic_tgt, feats, road_z,
                       horizon: int = 15, gamma: float = 0.99, lam: float = 0.95,
                       eta_entropy: float = 1e-3) -> dict:
    """Imagined rollouts from every eligible posterior start state (Dreamer-style
    all-starts flattening), REINFORCE actor loss on lambda-returns, critic MSE.

    Start t0 in [0, L-horizon-1]; imagined step tau scores the REAL candidate set
    at cur_t = t0+1+tau. Eligible = window fully valid (seq_mask is right-padding
    only, so valid[:, t0+horizon] implies the whole window) AND has candidates at
    every window step."""
    valid, has_cand = feats["valid"], feats["has_cand"]
    cand_seg, cand_head = feats["cand_segment_id"], feats["cand_heading_deg"]
    cand_oneway, cand_hw = feats["cand_oneway"], feats["cand_highway_id"]
    cand_mask = feats["cand_mask"]
    B, L = valid.shape
    S = L - horizon  # number of start offsets
    if S <= 0:
        raise ValueError(f"seq len {L} too short for horizon {horizon}")
    device = valid.device
    K = cand_seg.size(-1)

    H_all, Z_all = posterior_states(rssm, feats, road_z)
    h = H_all[:, :S].reshape(B * S, -1)
    z = Z_all[:, :S].reshape(B * S, -1)

    # eligibility: window fully valid + candidates exist at every imagined step
    start_ok = valid[:, horizon:horizon + S].clone()          # [B,S] right-pad property
    for tau in range(horizon):
        start_ok &= has_cand[:, 1 + tau: 1 + tau + S]
    mask = start_ok.reshape(-1)                                # [B*S]

    logps, entrs, rews, v_on, v_tg = [], [], [], [], []
    for tau in range(horizon):
        sl = slice(1 + tau, 1 + tau + S)
        cs = cand_seg[:, sl].reshape(B * S, K)
        ch = cand_head[:, sl].reshape(B * S, K)
        co = cand_oneway[:, sl].reshape(B * S, K)
        cw = cand_hw[:, sl].reshape(B * S, K)
        cm = cand_mask[:, sl].reshape(B * S, K)

        logits = actor(h, z, road_z, cs, ch, co, cw, cm)       # [B*S,K], grad->actor only
        dist = torch.distributions.Categorical(logits=logits)  # -1e4 mask: finite-safe
        a = dist.sample()
        logps.append(dist.log_prob(a))
        entrs.append(dist.entropy())

        with torch.no_grad():
            # reward head predicts symlog(reward); decode to true scale for returns
            r_hat = symexp(heads.reward(h, z, road_z, cs, ch, co, cw, cm))
            rews.append(r_hat.gather(1, a.unsqueeze(-1)).squeeze(-1))
            v_tg.append(critic_tgt(h, z))
        v_on.append(critic(h, z))

        seg = cs.gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            h = rssm.step(h, z, road_z[seg.clamp(min=0)])
            z = straight_through_sample(rssm.prior(h)).reshape(B * S, -1)

    with torch.no_grad():
        bootstrap = critic_tgt(h, z)                            # V(s_H)
    rews = torch.stack(rews)                                    # [H, N]
    v_tg = torch.stack(v_tg)
    v_on = torch.stack(v_on)
    logps = torch.stack(logps)
    entrs = torch.stack(entrs)

    returns = lambda_return(rews, v_tg, bootstrap, gamma=gamma, lam=lam)  # [H,N] no-grad
    adv = returns - v_tg
    adv = adv / adv[:, mask].std().clamp(min=1.0) if mask.any() else adv

    m = mask.unsqueeze(0).expand_as(logps)
    n = m.sum().clamp(min=1)
    actor_loss = -(adv.detach() * logps)[m].sum() / n - eta_entropy * entrs[m].sum() / n
    critic_loss = F.mse_loss(v_on[m], returns[m]) if mask.any() else v_on.sum() * 0.0

    return {"actor": actor_loss, "critic": critic_loss,
            "entropy": entrs[m].mean().detach() if mask.any() else entrs.mean().detach(),
            "ret_mean": returns[m].mean().detach() if mask.any() else returns.mean().detach(),
            "rew_mean": rews[m].mean().detach() if mask.any() else rews.mean().detach(),
            "n_starts": int(mask.sum()), "env_steps": int(mask.sum()) * horizon}


def train(processed_root, osm_root, stage0_ckpt, stage1_ckpt, wm_ckpt, city="porto",
          source="porto", limit_trajs=200_000, batch=16, steps=2000, seq_len=64,
          horizon=15, gamma=0.99, lam=0.95, eta_entropy=1e-3, lr=1e-4,
          ema_tau=0.02, grad_clip=100.0, weight_decay=1e-4, out_dir="ckpt",
          ckpt_every=250, log_every=25, device=None, seed=0, workers=0, resume=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    road_z = _road_embeddings(osm_root, city, stage0_ckpt, device)
    stage1 = _stage1_encoder(road_z.size(1), stage1_ckpt, device)
    rssm, heads = load_world_model(wm_ckpt, road_z.size(1), device)
    base_ds, _ = build_loader(processed_root, osm_root, city, source, limit_trajs, batch,
                               workers=workers, cache_dir=str(out / "cache"))

    from dataset.trajectories import collate_fn
    from torch.utils.data import DataLoader

    def loader():
        ds = LengthCapped(base_ds, seq_len)
        return iter(DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
                                num_workers=workers))

    actor = RoadScorer(h_dim=rssm.h_dim, road_dim=road_z.size(1)).to(device)
    actor.load_state_dict(heads.road.state_dict())  # methodology Stage 3 warm start
    critic = Critic(h_dim=rssm.h_dim).to(device)
    critic_tgt = copy.deepcopy(critic)
    for p in critic_tgt.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)

    start_step, total_env = 0, 0
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor"]); critic.load_state_dict(ckpt["critic"])
        critic_tgt.load_state_dict(ckpt["critic_tgt"])
        opt_a.load_state_dict(ckpt["opt_a"]); opt_c.load_state_dict(ckpt["opt_c"])
        start_step = ckpt["step"] + 1; total_env = ckpt.get("env_steps", 0)
        print(f"[stage3] resumed from {resume} at step {start_step}")

    print(f"[stage3] device={device} steps={steps} horizon={horizon} gamma={gamma} "
          f"lam={lam} eta={eta_entropy} lr={lr:.1e} wm={wm_ckpt}")
    it = loader()
    for step in range(start_step, steps):
        try:
            b = next(it)
        except StopIteration:
            it = loader(); b = next(it)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        feats = extract_features_r(stage1, road_z, b)

        out_l = imagine_and_losses(rssm, heads, actor, critic, critic_tgt, feats, road_z,
                                    horizon=horizon, gamma=gamma, lam=lam,
                                    eta_entropy=eta_entropy)
        opt_a.zero_grad(); opt_c.zero_grad()
        (out_l["actor"] + out_l["critic"]).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        opt_a.step(); opt_c.step()
        with torch.no_grad():  # EMA target critic (DreamerV3-style slow critic)
            for pt, po in zip(critic_tgt.parameters(), critic.parameters()):
                pt.lerp_(po, ema_tau)

        total_env += out_l["env_steps"]
        if step % log_every == 0:
            print(f"step {step:06d}  actor {float(out_l['actor']):.4f}  "
                  f"critic {float(out_l['critic']):.4f}  entropy {out_l['entropy']:.3f}  "
                  f"ret {out_l['ret_mean']:.3f}  rew {out_l['rew_mean']:.3f}  "
                  f"starts {out_l['n_starts']}  env_steps {total_env:,}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_ckpt(out / f"stage3_{city}_step{step}.pt", actor, critic, critic_tgt,
                       opt_a, opt_c, step, total_env, wm_ckpt)

    _save_ckpt(out / f"stage3_{city}.pt", actor, critic, critic_tgt, opt_a, opt_c,
               steps - 1, total_env, wm_ckpt)
    print(f"[stage3] saved -> {out / f'stage3_{city}.pt'}  (imagined env steps: {total_env:,})")


def _save_ckpt(path, actor, critic, critic_tgt, opt_a, opt_c, step, env_steps, wm_ckpt):
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "critic_tgt": critic_tgt.state_dict(),
                "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                "step": step, "env_steps": env_steps, "wm_ckpt": str(wm_ckpt)}, path)


def _smoke():
    """Actor-critic machinery on a random frozen WM + synthetic features:
    critic regresses toward lambda-returns (loss drops) and the actor moves
    probability mass toward higher predicted-reward candidates."""
    try:
        from training.stage2r import _synthetic_feats
    except ImportError:
        from stage2r import _synthetic_feats  # type: ignore
    torch.manual_seed(0)
    feats, road_z, _ = _synthetic_feats(B=4, L=24)
    rssm = RSSM2(obs_dim=256, road_dim=256, norm="batch")
    heads = HeadsLight(h_dim=rssm.h_dim, road_dim=256)
    for m in (rssm, heads):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    actor = RoadScorer(h_dim=rssm.h_dim, road_dim=256)
    actor.load_state_dict(heads.road.state_dict())
    critic = Critic(h_dim=rssm.h_dim)
    critic_tgt = copy.deepcopy(critic)
    for p in critic_tgt.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    first_c = last_c = first_r = last_r = None
    for i in range(80):
        out_l = imagine_and_losses(rssm, heads, actor, critic, critic_tgt, feats, road_z,
                                    horizon=8)
        opt_a.zero_grad(); opt_c.zero_grad()
        (out_l["actor"] + out_l["critic"]).backward()
        opt_a.step(); opt_c.step()
        with torch.no_grad():
            for pt, po in zip(critic_tgt.parameters(), critic.parameters()):
                pt.lerp_(po, 0.05)
        if first_c is None:
            first_c, first_r = float(out_l["critic"]), float(out_l["rew_mean"])
        last_c, last_r = float(out_l["critic"]), float(out_l["rew_mean"])
    print(f"[smoke] critic {first_c:.4f} -> {last_c:.4f}  rew_mean {first_r:.4f} -> {last_r:.4f}  "
          f"entropy {out_l['entropy']:.3f}  starts {out_l['n_starts']}")
    assert last_c < first_c, f"critic did not learn: {first_c:.4f} -> {last_c:.4f}"
    assert last_r > first_r, f"actor did not move toward predicted reward: {first_r:.4f} -> {last_r:.4f}"
    assert float(out_l["entropy"]) > 0.0
    print("[smoke] PASS")


def main():
    p = argparse.ArgumentParser(description="research2 Stage-3 actor-critic in frozen world model")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--osm-root", default="data/osm")
    p.add_argument("--stage0-ckpt", default="ckpt/stage0_porto.pt")
    p.add_argument("--stage1-ckpt", default="ckpt/stage1_porto.pt")
    p.add_argument("--wm-ckpt", default="ckpt/stage2r_porto.pt")
    p.add_argument("--city", default="porto")
    p.add_argument("--source", default="porto")
    p.add_argument("--limit-trajs", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eta-entropy", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-tau", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--out-dir", default="ckpt")
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        _smoke(); return
    train(a.processed_root, a.osm_root, a.stage0_ckpt, a.stage1_ckpt, a.wm_ckpt, a.city,
          a.source, a.limit_trajs, a.batch, a.steps, a.seq_len, a.horizon, a.gamma, a.lam,
          a.eta_entropy, a.lr, a.ema_tau, a.grad_clip,
          out_dir=a.out_dir, ckpt_every=a.ckpt_every, log_every=a.log_every,
          device=a.device, workers=a.workers, resume=a.resume)


if __name__ == "__main__":
    main()
