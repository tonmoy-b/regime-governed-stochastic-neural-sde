import torch
import torch.nn.functional as F
from regime_governed.envs import SimConfig, sample_macro_regimes
from regime_governed.generator import ConditionalNeuralJumpSDE
from regime_governed.control import (
    RegimeGovernedMoE,
    ExtremaCVaRLoss,
    moe_gating_loss,
    calculate_duration,
    rollout_hedging_episode,
)




def main():
    torch.manual_seed(42)
    cfg = SimConfig()
    print(f"[*] Starting Stage 2: Causal MoE Control Optimization on {cfg.device}")

    generator = ConditionalNeuralJumpSDE(cfg.macro_dim, cfg.hidden_dim).to(cfg.device)
    generator.load_state_dict(torch.load("generator_v1.pt", map_location=cfg.device))
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False

    I_dim, Z_dim = 5, 5
    moe_policy = RegimeGovernedMoE(I_dim, Z_dim).to(cfg.device)
    cvar_criterion = ExtremaCVaRLoss().to(cfg.device)

    optimizer = torch.optim.AdamW(
        list(moe_policy.parameters()) + [cvar_criterion.v_offset],
        lr=2e-3,
        weight_decay=1e-4,
    )

    for epoch in range(1, 151):
        optimizer.zero_grad()
        X_batch, _ = sample_macro_regimes(cfg.batch_size, cfg.n_steps, cfg.device)

        L_T, lam_paths, g_weights = rollout_hedging_episode(
            moe_policy, generator, X_batch, cfg, is_moe=True
        )

        cvar_extrema_loss, raw_cvar = cvar_criterion(L_T, lam_paths)
        gate_loss = moe_gating_loss(g_weights)

        total_loss = cvar_extrema_loss + gate_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_policy.parameters(), 1.0)
        optimizer.step()

        if epoch % 25 == 0:
            g_bar = g_weights.mean(dim=(0, 1))
            print(
                f"Epoch {epoch:03d} | CVaR99 (bps): {raw_cvar.item():6.2f} | "
                f"Gate Utils: [{g_bar[0]:.2f}, {g_bar[1]:.2f}, {g_bar[2]:.2f}]"
            )

    torch.save(moe_policy.state_dict(), "moe_policy_v1.pt")
    print("[+] Saved moe_policy_v1.pt")


if __name__ == "__main__":
    main()