import torch
from regime_governed.envs import SimConfig, sample_macro_regimes
from regime_governed.generator import ConditionalNeuralJumpSDE
from regime_governed.control import (
    RegimeGovernedMoE,
    MonolithicPolicy,
    ExtremaCVaRLoss,
    rollout_hedging_episode,
)
from importlib import import_module

# Import the exact same causal rollout from 02_train_moe
train_moe_mod = import_module("02_train_moe")
rollout_hedging_episode = train_moe_mod.rollout_hedging_episode


def evaluate_by_regime(
    policy, generator, cfg, X_eval, regime_ids, is_moe=True, sparse_top1=False
):
    with torch.no_grad():
        L_T, _, _ = rollout_hedging_episode(
            policy,
            generator,
            X_eval,
            cfg,
            is_moe=is_moe,
            sparse_top1=sparse_top1,
        )
        metrics = {}
        for r, name in enumerate(["Regime 0", "Regime 1", "Regime 2"]):
            mask = regime_ids == r
            if mask.sum() == 0:
                continue
            L_T_r = L_T[mask]
            sorted_L, _ = torch.sort(L_T_r)
            idx_99 = max(0, int(0.99 * len(sorted_L)) - 1)
            cvar99 = sorted_L[idx_99:].mean().item()
            metrics[name] = cvar99
        return metrics


def main():
    torch.manual_seed(42)
    cfg = SimConfig()
    print("[*] Starting Stage 3: Causal Baseline Evaluation & Table 2 Generation")

    generator = ConditionalNeuralJumpSDE(cfg.macro_dim, cfg.hidden_dim).to(cfg.device)
    generator.load_state_dict(torch.load("generator_v1.pt", map_location=cfg.device))
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    moe_policy = RegimeGovernedMoE(5, 5).to(cfg.device)
    moe_policy.load_state_dict(torch.load("moe_policy_v1.pt", map_location=cfg.device))
    moe_policy.eval()

    # Train Monolithic Baseline under the EXACT same causal rollout & CVaR loss
    print("[*] Training Monolithic Baseline (Causal PnL Loop + CVaR99 Objective)...")
    mono_policy = MonolithicPolicy(5, 5).to(cfg.device)
    mono_cvar_criterion = ExtremaCVaRLoss(lambda_reg=0.0).to(cfg.device)

    optimizer = torch.optim.AdamW(
        list(mono_policy.parameters()) + [mono_cvar_criterion.v_offset],
        lr=2e-3,
        weight_decay=1e-4,
    )

    for epoch in range(1, 151):
        optimizer.zero_grad()
        X_batch, _ = sample_macro_regimes(cfg.batch_size, cfg.n_steps, cfg.device)
        L_T, lam_paths, _ = rollout_hedging_episode(
            mono_policy, generator, X_batch, cfg, is_moe=False
        )
        loss, raw_cvar = mono_cvar_criterion(L_T, lam_paths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mono_policy.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"    [Mono] Epoch {epoch:03d} | CVaR99 (bps): {raw_cvar.item():6.2f}")

    mono_policy.eval()

    # Out-of-sample Evaluation on 10,000 fresh trajectories
    X_eval, reg_eval = sample_macro_regimes(10000, cfg.n_steps, cfg.device)

    mono_metrics = evaluate_by_regime(
        mono_policy, generator, cfg, X_eval, reg_eval, is_moe=False
    )
    moe_dense_metrics = evaluate_by_regime(
        moe_policy, generator, cfg, X_eval, reg_eval, is_moe=True, sparse_top1=False
    )
    moe_sparse_metrics = evaluate_by_regime(
        moe_policy, generator, cfg, X_eval, reg_eval, is_moe=True, sparse_top1=True
    )

    print("\n=== PAPER TABLE 2: OUT-OF-SAMPLE CVaR_99 HEDGING LOSS (BPS) ===")
    print(
        f"{'Regime':<12} | {'Monolithic MLP':<16} | {'MoE (Dense)':<14} | {'MoE (Top-1)':<14} | {'MoE Gain':<10}"
    )
    print("-" * 78)
    for r in ["Regime 0", "Regime 1", "Regime 2"]:
        mono_val = mono_metrics[r]
        dense_val = moe_dense_metrics[r]
        sparse_val = moe_sparse_metrics[r]
        imp = ((mono_val - dense_val) / mono_val) * 100.0
        print(
            f"{r:<12} | {mono_val:>16.2f} | {dense_val:>14.2f} | {sparse_val:>14.2f} | {imp:>8.1f}%"
        )


if __name__ == "__main__":
    main()