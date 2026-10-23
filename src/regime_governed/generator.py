import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DuffiePanSingletonBaseline:
    """Classical Affine Jump-Diffusion Ground Truth & Baseline."""
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def simulate_ground_truth(self, X_path: torch.Tensor) -> torch.Tensor:
        B, T, _ = X_path.shape
        dt, sqrt_dt = self.cfg.dt, math.sqrt(self.cfg.dt)
        lam = torch.full((B, T + 1), self.cfg.lambda_0, device=X_path.device)

        for t in range(T):
            x_t = X_path[:, t, :]
            spread_z, vix, slope = x_t[:, 0], x_t[:, 1], x_t[:, 2]
            
            theta_t = 0.02 + 0.025 * F.relu(spread_z) + 0.015 * F.relu(-slope)
            sigma_t = 0.12 + 0.10 * vix
            nu_t = 0.5 + 3.5 * F.softplus(spread_z - 0.5) + 15.0 * lam[:, t]
            jump_mean_t = 0.008 + 0.018 * vix

            cur_lam = torch.clamp(lam[:, t], min=1e-5)
            dW = torch.randn(B, device=X_path.device) * sqrt_dt
            dN = torch.bernoulli(1.0 - torch.exp(-nu_t * dt))
            J_dN = torch.distributions.Exponential(1.0 / jump_mean_t).sample() * dN

            next_lam = cur_lam + 2.5 * (theta_t - cur_lam) * dt + sigma_t * torch.sqrt(cur_lam) * dW + J_dN
            lam[:, t + 1] = torch.clamp(next_lam, min=1e-4)
        return lam

class ConditionalNeuralJumpSDE(nn.Module):
    """Exogenous-Conditioned Neural Jump-Diffusion (Sections 1.2 & 2.1)."""
    def __init__(self, macro_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        in_dim = 1 + macro_dim + 1
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.drift_head = nn.Linear(hidden_dim, 1)
        self.diff_head = nn.Linear(hidden_dim, 1)
        self.intensity_head = nn.Linear(hidden_dim, 1)
        self.jump_size_head = nn.Linear(hidden_dim, 2)

    def forward_step_params(self, lam_t, x_t, t_norm):
        h = self.trunk(torch.cat([lam_t, x_t, t_norm], dim=-1))
        mu = self.drift_head(h) - 1.5 * (lam_t - 0.03)  # k_0 structural prior
        sigma = F.softplus(self.diff_head(h)) * torch.sqrt(torch.clamp(lam_t, min=1e-5))
        nu = F.softplus(self.intensity_head(h)) + 1e-3
        j_params = self.jump_size_head(h)
        return mu, sigma, nu, F.softplus(j_params[:, 0:1]) * 0.05 + 1e-4, F.softplus(j_params[:, 1:2]) * 0.02 + 1e-4

    def simulate_paths(self, X_path: torch.Tensor, lambda_0: float, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, _ = X_path.shape
        lam_list = [torch.full((B, 1), lambda_0, device=X_path.device)]
        nu_list = []

        for t in range(T):
            cur_lam = lam_list[-1]
            t_norm = torch.full((B, 1), float(t) / float(T), device=X_path.device)
            mu, sigma, nu, j_mean, j_vol = self.forward_step_params(cur_lam, X_path[:, t, :], t_norm)

            dW = torch.randn(B, 1, device=X_path.device) * math.sqrt(dt)
            
            # Straight-Through Poisson Estimator (Section 2.2)
            jump_prob = 1.0 - torch.exp(-nu * dt)
            dN_hard = (torch.rand_like(jump_prob) < jump_prob).float()
            dN = dN_hard - jump_prob.detach() + jump_prob

            J_mag = j_mean + j_vol * F.softplus(torch.randn(B, 1, device=X_path.device))
            next_lam = 1e-4 + F.softplus((cur_lam + mu * dt + sigma * dW + J_mag * dN) - 1e-4, beta=100.0)
            
            lam_list.append(next_lam)
            nu_list.append(nu)

        return torch.cat(lam_list, dim=1), {"nu_paths": torch.cat(nu_list, dim=1)}

def multi_horizon_wasserstein_tail_loss(sim_paths, target_paths, regime_ids, horizons=(15, 30, 45, 63)):
    """Section 2.3: W1 + Tail Quantile Calibration Loss."""
    total_loss = 0.0
    for r in range(3):
        mask = (regime_ids == r)
        if mask.sum() < 16: continue
        s_r, t_r = sim_paths[mask], target_paths[mask]

        for h in horizons:
            s_sort, _ = torch.sort(s_r[:, h])
            t_sort, _ = torch.sort(t_r[:, h])
            w1_loss = torch.mean(torch.abs(s_sort - t_sort))
            
            idx_95 = int(0.95 * len(s_sort))
            tail_loss = torch.mean(torch.abs(s_sort[idx_95:] - t_sort[idx_95:]))
            total_loss += w1_loss + 2.5 * tail_loss

        # Jump variation pathwise matching
        ds_sort, _ = torch.sort((s_r[:, 1:] - s_r[:, :-1]).reshape(-1))
        dt_sort, _ = torch.sort((t_r[:, 1:] - t_r[:, :-1]).reshape(-1))
        idx_99 = int(0.99 * len(ds_sort))
        total_loss += 1.5 * torch.mean(torch.abs(ds_sort - dt_sort)) + 4.0 * torch.mean(torch.abs(ds_sort[idx_99:] - dt_sort[idx_99:]))
    return total_loss