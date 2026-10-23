import torch
import torch.nn as nn
import torch.nn.functional as F
from regime_governed.envs import SimConfig, sample_macro_regimes
from regime_governed.generator import ConditionalNeuralJumpSDE
from regime_governed.control import RegimeGovernedMoE

def calculate_duration(lam_t, t_norm):
    """Approximates CS01 risky duration."""
    time_to_mat = 5.0 * (1.0 - t_norm)
    return time_to_mat * torch.exp(-lam_t * time_to_mat)

class MonolithicPolicy(nn.Module):
    """Classical Monolithic Deep Hedging Network (Baseline)."""
    def __init__(self, I_dim: int, Z_dim: int):
        super().__init__()
        # Concatenates local state (I_t) and regime context (Z_t) into one MLP
        self.net = nn.Sequential(
            nn.Linear(I_dim + Z_dim, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, I_t: torch.Tensor, Z_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([I_t, Z_t], dim=-1))
        return u_prev + delta

def evaluate_policy(policy, generator, cfg, X_batch, regime_ids, is_moe=False):
    """Evaluates a policy on fixed paths and returns regime-specific CVaR99 metrics."""
    with torch.no_grad():
        lam_paths, telemetry = generator.simulate_paths(X_batch, cfg.lambda_0, cfg.dt)
        nu_paths = telemetry["nu_paths"]
        
        B, T = X_batch.shape[0], cfg.n_steps
        u_t = torch.zeros(B, 1, device=cfg.device)
        total_pnl = torch.zeros(B, 1, device=cfg.device)
        switching_costs = torch.zeros(B, 1, device=cfg.device)
        S_t = torch.ones(B, 1, device=cfg.device)
        
        terminal_lam = lam_paths[:, -1:]
        delta_v_t = -100.0 * (terminal_lam - cfg.lambda_0) 

        for t in range(T):
            cur_lam = lam_paths[:, t:t+1]
            next_lam = lam_paths[:, t+1:t+2]
            cur_nu = nu_paths[:, t:t+1]
            x_t = X_batch[:, t, :]
            
            t_norm = torch.full((B, 1), float(t)/float(T), device=cfg.device)
            dur_t = calculate_duration(cur_lam, t_norm)
            
            I_t = torch.cat([cur_lam, S_t, dur_t, u_t, 1.0 - t_norm], dim=-1)
            tail_extrema = F.relu(next_lam - cur_lam) / cfg.dt
            Z_t = torch.cat([x_t, cur_nu, tail_extrema], dim=-1)
            
            if is_moe:
                u_next, _ = policy(I_t, Z_t, u_t)
            else:
                u_next = policy(I_t, Z_t, u_t)
                
            dS = -dur_t * (next_lam - cur_lam) * S_t
            pnl_step = u_next * dS
            
            c_t = 0.0001 * (1.0 + F.softplus(x_t[:, 0:1]))
            cost_step = c_t * torch.abs(u_next - u_t)
            
            total_pnl += pnl_step
            switching_costs += cost_step
            S_t = S_t + dS
            u_t = u_next
            
        L_T = (-total_pnl + switching_costs - delta_v_t).squeeze()
        
        metrics = {}
        for r, name in enumerate(["Regime 0", "Regime 1", "Regime 2"]):
            mask = (regime_ids == r)
            if mask.sum() == 0: continue
            
            L_T_r = L_T[mask]
            sorted_L, _ = torch.sort(L_T_r)
            cvar99 = sorted_L[int(0.99 * len(sorted_L)):].mean().item()
            metrics[name] = cvar99
            
        return metrics

def main():
    torch.manual_seed(42)
    cfg = SimConfig()
    print(f"[*] Starting Stage 3: Baseline Evaluation & Table 2 Generation")

    generator = ConditionalNeuralJumpSDE(cfg.macro_dim, cfg.hidden_dim).to(cfg.device)
    generator.load_state_dict(torch.load("generator_v1.pt", map_location=cfg.device))
    generator.eval()

    moe_policy = RegimeGovernedMoE(5, 5).to(cfg.device)
    moe_policy.load_state_dict(torch.load("moe_policy_v1.pt", map_location=cfg.device))
    moe_policy.eval()

    # Train Monolithic Baseline with proper PnL sequence AND CVaR Loss
    print("[*] Training Monolithic Baseline (Sequential PnL Loop)...")
    mono_policy = MonolithicPolicy(5, 5).to(cfg.device)
    
    from regime_governed.control import ExtremaCVaRLoss # Import the CVaR loss
    mono_cvar_criterion = ExtremaCVaRLoss(lambda_reg=0.0).to(cfg.device) # Turn off extrema weight for pure CVaR
    
    optimizer = torch.optim.AdamW(
        list(mono_policy.parameters()) + [mono_cvar_criterion.v], 
        lr=1e-3, weight_decay=1e-4
    )
    
    for epoch in range(1, 151):
        optimizer.zero_grad()
        X_batch, _ = sample_macro_regimes(cfg.batch_size, cfg.n_steps, cfg.device)
        lam_paths, telemetry = generator.simulate_paths(X_batch, cfg.lambda_0, cfg.dt)
        nu_paths = telemetry["nu_paths"]
        
        B, T = X_batch.shape[0], cfg.n_steps
        u_t = torch.zeros(B, 1, device=cfg.device)
        total_pnl = torch.zeros(B, 1, device=cfg.device)
        switching_costs = torch.zeros(B, 1, device=cfg.device)
        S_t = torch.ones(B, 1, device=cfg.device)
        
        terminal_lam = lam_paths[:, -1:]
        delta_v_t = -100.0 * (terminal_lam - cfg.lambda_0) 

        for t in range(T):
            cur_lam = lam_paths[:, t:t+1]
            next_lam = lam_paths[:, t+1:t+2]
            cur_nu = nu_paths[:, t:t+1]
            x_t = X_batch[:, t, :]
            
            t_norm = torch.full((B, 1), float(t)/float(T), device=cfg.device)
            dur_t = calculate_duration(cur_lam, t_norm)
            
            I_t = torch.cat([cur_lam, S_t, dur_t, u_t, 1.0 - t_norm], dim=-1)
            tail_extrema = F.relu(next_lam - cur_lam) / cfg.dt
            Z_t = torch.cat([x_t, cur_nu, tail_extrema], dim=-1)
            
            # Bound the monolithic output to prevent early explosion before CVaR stabilizes
            delta_u = mono_policy.net(torch.cat([I_t, Z_t], dim=-1))
            u_next = u_t + torch.clamp(delta_u, -1.0, 1.0) 
            
            dS = -dur_t * (next_lam - cur_lam) * S_t
            pnl_step = u_next * dS
            c_t = 0.0001 * (1.0 + F.softplus(x_t[:, 0:1]))
            cost_step = c_t * torch.abs(u_next - u_t)
            
            total_pnl += pnl_step
            switching_costs += cost_step
            S_t = S_t + dS
            u_t = u_next
            
        L_T = (-total_pnl + switching_costs - delta_v_t).squeeze()
        
        # Use the robust CVaR loss instead of explosive Mean-Variance
        loss, _ = mono_cvar_criterion(L_T, lam_paths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mono_policy.parameters(), 1.0)
        optimizer.step()

    # Out-of-sample Evaluation
    X_eval, reg_eval = sample_macro_regimes(10000, cfg.n_steps, cfg.device)
    
    print("\n=== PAPER TABLE 2: OUT-OF-SAMPLE CVaR_99 HEDGING LOSS ===")
    moe_metrics = evaluate_policy(moe_policy, generator, cfg, X_eval, reg_eval, is_moe=True)
    mono_metrics = evaluate_policy(mono_policy, generator, cfg, X_eval, reg_eval, is_moe=False)
    
    print(f"{'Regime':<12} | {'Monolithic Baseline':<20} | {'Regime-Governed MoE':<20} | {'Improvement':<12}")
    print("-" * 75)
    for r in ["Regime 0", "Regime 1", "Regime 2"]:
        mono_val = mono_metrics.get(r, 0)
        moe_val = moe_metrics.get(r, 0)
        imp = ((mono_val - moe_val) / mono_val) * 100 if mono_val > 0 else 0
        print(f"{r:<12} | {mono_val:>18.2f} | {moe_val:>18.2f} | {imp:>11.1f}%")

if __name__ == "__main__":
    main()