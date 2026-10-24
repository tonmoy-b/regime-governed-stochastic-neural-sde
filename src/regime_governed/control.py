from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .envs import SimConfig


def calculate_duration(lam_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
    """Approximates CS01 risky duration of a 5-year credit portfolio."""
    time_to_mat = 5.0 * (1.0 - 0.5 * t_norm)
    return time_to_mat * torch.exp(-lam_t * time_to_mat)


class ExtremaCVaRLoss(nn.Module):
    """
    Section 3: Rockafellar-Uryasev CVaR with Generator-Governed Extrema Regularization.
    """
    def __init__(
        self,
        alpha: float = 0.99,
        tau: float = 1.0,
        beta: float = 2.5,
        eta: float = 2.5,
        lambda_reg: float = 0.35,
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.eta = eta
        self.lambda_reg = lambda_reg
        # Trainable residual offset around the empirical alpha-quantile
        self.v_offset = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, terminal_loss: torch.Tensor, lambda_paths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Pathwise quadratic jump variation Q_T^{(i)}
        d_lam = lambda_paths[:, 1:] - lambda_paths[:, :-1]
        Q_T = torch.sum(d_lam ** 2, dim=1)
        Q_T_norm = (Q_T - Q_T.mean()) / (Q_T.std() + 1e-6)

        # 2. Rockafellar-Uryasev threshold anchored at empirical quantile q_alpha
        q_alpha = torch.quantile(terminal_loss.detach(), self.alpha)
        v = q_alpha + self.v_offset

        excess = terminal_loss - v
        smoothed_hinge = F.softplus(excess, beta=self.tau)
        cvar_term = v + smoothed_hinge.mean() / (1.0 - self.alpha)

        # 3. Softmax Extrema Weights (focuses gradients on high-loss + high-jump paths)
        loss_std = terminal_loss.detach().std() + 1e-6
        ex_norm = F.relu(excess) / loss_std
        logits = torch.clamp(self.beta * ex_norm + self.eta * F.relu(Q_T_norm), max=15.0)
        w_i = F.softmax(logits, dim=0)

        # 4. Stop-Gradient Extrema Regularizer
        extrema_reg = torch.sum(w_i.detach() * terminal_loss)

        # Normalize total loss by loss_std so its gradient scale (~O(1)) matches moe_gating_loss
        raw_objective = cvar_term + self.lambda_reg * extrema_reg
        scaled_loss = raw_objective / loss_std
        return scaled_loss, cvar_term.detach()


def moe_gating_loss(
    gate_weights: torch.Tensor,
    lambda_bal: float = 0.25,
    lambda_ent: float = 0.08,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Section 4.3: Mutual Information & Load-Balancing Regularizer.
    gate_weights shape: (B, M, E)
    """
    B, M, E = gate_weights.shape
    g_bar = gate_weights.mean(dim=(0, 1))

    # Term 1: Marginal Load-Balancing
    bal_loss = lambda_bal * E * torch.sum((g_bar - (1.0 / E)) ** 2)

    # Term 2: Conditional Per-Sample Entropy
    ent_loss = (
        lambda_ent
        * (-torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)).mean()
    )

    return bal_loss + ent_loss


class SoftShrinkExpert(nn.Module):
    """Expert 1: Quiescent / Switching-Cost Dead-Zone Policy."""
    def __init__(self, in_dim: int, eps1: float = 0.08, u_max: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh()
        )
        self.eps1 = eps1
        self.u_max = u_max

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        delta = 0.25 * self.net(I_t)
        shrink = torch.sign(delta) * F.relu(torch.abs(delta) - self.eps1)
        return torch.clamp(u_prev + shrink, 0.0, self.u_max)


class DiffusiveTrackingExpert(nn.Module):
    """Expert 2: High-Diffusion Continuous Tracking Policy."""
    def __init__(self, in_dim: int, u_max: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh()
        )
        self.u_max = u_max

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        delta = 0.50 * self.net(I_t)
        return torch.clamp(u_prev + delta, 0.0, self.u_max)


class JumpShockExpert(nn.Module):
    """Expert 3: Discontinuous Jump-Shock Impulse Policy (Unanchored Reset)."""
    def __init__(self, in_dim: int, u_max: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.u_max = u_max

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        return self.u_max * self.net(I_t)


class RegimeGovernedMoE(nn.Module):
    """Section 4: Macro-Gated Mixture of Experts Control Policy."""
    def __init__(self, I_dim: int, Z_dim: int, u_max: float = 1.5):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(Z_dim, 32), nn.SiLU(), nn.Linear(32, 3)
        )
        self.expert1 = SoftShrinkExpert(I_dim, u_max=u_max)
        self.expert2 = DiffusiveTrackingExpert(I_dim, u_max=u_max)
        self.expert3 = JumpShockExpert(I_dim, u_max=u_max)
        self.tau_g = 0.7

    def forward(
        self,
        I_t: torch.Tensor,
        Z_t: torch.Tensor,
        u_prev: torch.Tensor,
        sparse_top1: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gating(Z_t)
        g_t = F.softmax(logits / self.tau_g, dim=-1)

        u1 = self.expert1(I_t, u_prev)
        u2 = self.expert2(I_t, u_prev)
        u3 = self.expert3(I_t, u_prev)

        if sparse_top1:
            u_stack = torch.cat([u1, u2, u3], dim=-1)
            k_star = torch.argmax(g_t, dim=-1, keepdim=True)
            u_out = torch.gather(u_stack, 1, k_star)
            return u_out, g_t

        u_dense = g_t[:, 0:1] * u1 + g_t[:, 1:2] * u2 + g_t[:, 2:3] * u3
        return u_dense, g_t


class MonolithicPolicy(nn.Module):
    """
    Classical Monolithic Deep Hedging Network (Baseline).
    Uses the exact same state inputs (I_t, Z_t) and actuation bounds [0, u_max] as the MoE.
    """
    def __init__(self, I_dim: int, Z_dim: int, u_max: float = 1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(I_dim + Z_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.u_max = u_max

    def forward(
        self, I_t: torch.Tensor, Z_t: torch.Tensor, u_prev: torch.Tensor
    ) -> torch.Tensor:
        delta = self.u_max * self.net(torch.cat([I_t, Z_t], dim=-1))
        return torch.clamp(u_prev + delta, 0.0, self.u_max)


def rollout_hedging_episode(
    policy: nn.Module,
    generator: nn.Module,
    X_batch: torch.Tensor,
    cfg: SimConfig,
    is_moe: bool = True,
    sparse_top1: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Strictly G_t-adapted sequential hedging rollout.
    """
    with torch.no_grad():
        lam_paths, telemetry = generator.simulate_paths(X_batch, cfg.lambda_0, cfg.dt)
        nu_paths = telemetry["nu_paths"]

    B, T = X_batch.shape[0], cfg.n_steps
    u_t = torch.ones((B, 1), device=cfg.device)  # Start at 1.0 par duration hedge
    S_t = torch.ones(B, 1, device=cfg.device)
    cum_loss = torch.zeros(B, 1, device=cfg.device)
    g_weights = []

    for t in range(T):
        cur_lam = lam_paths[:, t : t + 1]
        prev_lam = lam_paths[:, t - 1 : t] if t > 0 else cur_lam
        cur_nu = nu_paths[:, t : t + 1]
        x_t = X_batch[:, t, :]

        t_norm = torch.full((B, 1), float(t) / float(T), device=cfg.device)
        dur_t = calculate_duration(cur_lam, t_norm)

        # 1. Local Plant State I_t (Strictly F_t-measurable)
        I_t = torch.cat([cur_lam, S_t, dur_t, u_t, 1.0 - t_norm], dim=-1)

        # 2. Routing Context Z_t (Strictly F_t-measurable)
        lagged_jump_rate = F.relu(cur_lam - prev_lam) / cfg.dt
        Z_t = torch.cat([x_t, cur_nu, lagged_jump_rate], dim=-1)

        # 3. Policy Actuation Decision at time t
        if is_moe:
            u_next, g_t = policy(I_t, Z_t, u_t, sparse_top1=sparse_top1)
            g_weights.append(g_t)
        else:
            u_next = policy(I_t, Z_t, u_t)

        # 4. Environment Step [t -> t+1]: Realized shock revealed AFTER action u_next
        next_lam = lam_paths[:, t + 1 : t + 2]
        d_lam = next_lam - cur_lam
        is_jump = (d_lam > 0.006).float()

        # Liability move in bps: diffusive duration + nonlinear jump-to-default convexity
        spread_z = x_t[:, 0:1]
        target_ratio = (
            1.0
            + 0.12 * torch.tanh(spread_z)
            + 0.35 * torch.clamp(cur_nu / 2.5, 0.0, 1.0)
        )
        move_bps = dur_t * torch.abs(d_lam) * 1e4 * (1.0 + 1.5 * is_jump)

        # Asymmetric tail tracking penalty (under-hedging during a jump is penalized 2.5x)
        hedge_err = target_ratio - u_next
        underhedge_mult = 1.0 + 1.5 * is_jump * (hedge_err > 0).float()
        tracking_loss = (
            underhedge_mult * torch.abs(hedge_err) * move_bps
            + 0.5 * (hedge_err ** 2) * move_bps
        )

        # Regime-dependent switching cost (high in calm carry to penalize chattering)
        c_t_bps = 3.5 * (1.0 + F.softplus(-spread_z))
        switching_cost_bps = c_t_bps * torch.abs(u_next - u_t)

        cum_loss = cum_loss + tracking_loss + switching_cost_bps
        S_t = S_t - dur_t * d_lam * S_t
        u_t = u_next

    L_T = cum_loss.squeeze(-1)
    g_tensor = torch.stack(g_weights, dim=1) if is_moe else None
    return L_T, lam_paths, g_tensor