import torch
import torch.nn.functional as F
from regime_governed.envs import SimConfig, sample_macro_regimes
from regime_governed.generator import ConditionalNeuralJumpSDE
from regime_governed.control import RegimeGovernedMoE, ExtremaCVaRLoss, moe_gating_loss

def calculate_duration(lam_t, t_norm):
    """Approximates CS01 risky duration."""
    time_to_mat = 5.0 * (1.0 - t_norm)
    return time_to_mat * torch.exp(-lam_t * time_to_mat)

def main():
    torch.manual_seed(42)
    cfg = SimConfig()
    print(f"[*] Starting Stage 2: MoE Control Optimization on {cfg.device}")

    # Load frozen Stage 1 Generator
    generator = ConditionalNeuralJumpSDE(cfg.macro_dim, cfg.hidden_dim).to(cfg.device)
    try:
        generator.load_state_dict(torch.load("generator_v1.pt", map_location=cfg.device))
        generator.eval()
        for param in generator.parameters():
            param.requires_grad = False
    except FileNotFoundError:
        print("Error: generator_v1.pt not found. Run 01_train_generator.py first.")
        return

    # Initialize Stage 2 Controller
    # I_dim = lam_t (1) + S_t (1) + Duration (1) + u_prev (1) + T-t (1) = 5
    I_dim = 5
    # Z_dim = Macro X_t (3) + nu_phi (1) + CVaR_tail (1) = 5
    Z_dim = 5
    moe_policy = RegimeGovernedMoE(I_dim, Z_dim).to(cfg.device)
    cvar_criterion = ExtremaCVaRLoss().to(cfg.device)
    
    optimizer = torch.optim.AdamW(
        list(moe_policy.parameters()) + [cvar_criterion.v], 
        lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(1, 151):
        optimizer.zero_grad()
        X_batch, regime_ids = sample_macro_regimes(cfg.batch_size, cfg.n_steps, cfg.device)
        
        # Generator rollout
        lam_paths, telemetry = generator.simulate_paths(X_batch, cfg.lambda_0, cfg.dt)
        nu_paths = telemetry["nu_paths"]
        
        B, T = X_batch.shape[0], cfg.n_steps
        u_t = torch.zeros(B, 1, device=cfg.device)
        total_pnl = torch.zeros(B, 1, device=cfg.device)
        switching_costs = torch.zeros(B, 1, device=cfg.device)
        g_weights = []

        S_t = torch.ones(B, 1, device=cfg.device) # Index Price
        
        # Unhedged book liability Delta V_T (short risk)
        terminal_lam = lam_paths[:, -1:]
        delta_v_t = -100.0 * (terminal_lam - cfg.lambda_0) 

        for t in range(T):
            cur_lam = lam_paths[:, t:t+1]
            next_lam = lam_paths[:, t+1:t+2]
            cur_nu = nu_paths[:, t:t+1]
            x_t = X_batch[:, t, :]
            
            t_norm = torch.full((B, 1), float(t)/float(T), device=cfg.device)
            dur_t = calculate_duration(cur_lam, t_norm)
            
            # Form Local Plant State I_t
            I_t = torch.cat([cur_lam, S_t, dur_t, u_t, 1.0 - t_norm], dim=-1)
            
            # Form Routing Context Z_t (Includes generator telemetry)
            # Pseudo-tail extrema feature (normalized jump diff)
            tail_extrema = F.relu(next_lam - cur_lam) / cfg.dt
            Z_t = torch.cat([x_t, cur_nu, tail_extrema], dim=-1)
            
            u_next, g_t = moe_policy(I_t, Z_t, u_t)
            g_weights.append(g_t)
            
            # Stepwise PnL & Transaction Costs
            dS = -dur_t * (next_lam - cur_lam) * S_t
            pnl_step = u_next * dS
            
            # Lambda_sw varies by regime
            spread_z = x_t[:, 0:1]
            c_t = 0.0001 * (1.0 + F.softplus(spread_z))
            cost_step = c_t * torch.abs(u_next - u_t)
            
            total_pnl += pnl_step
            switching_costs += cost_step
            S_t = S_t + dS
            u_t = u_next
            
        g_weights = torch.stack(g_weights, dim=1) # (B, T, 3)
        L_T = -total_pnl + switching_costs - delta_v_t
        
        cvar_extrema_loss, raw_cvar = cvar_criterion(L_T.squeeze(), lam_paths)
        gate_loss = moe_gating_loss(g_weights)
        
        total_loss = cvar_extrema_loss + gate_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_policy.parameters(), 1.0)
        optimizer.step()

        if epoch % 25 == 0:
            g_bar = g_weights.mean(dim=(0, 1))
            print(f"Epoch {epoch:03d} | CVaR99: {raw_cvar.item():.4f} | "
                  f"Gate Utils: [{g_bar[0]:.2f}, {g_bar[1]:.2f}, {g_bar[2]:.2f}]")

    torch.save(moe_policy.state_dict(), "moe_policy_v1.pt")
    print("[+] Saved moe_policy_v1.pt")

if __name__ == "__main__":
    main()
