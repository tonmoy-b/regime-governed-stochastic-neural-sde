import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ExtremaCVaRLoss(nn.Module):
    """
    Section 3: Rockafellar-Uryasev CVaR with Generator-Governed Extrema Regularization.
    """
    def __init__(self, alpha: float = 0.99, tau: float = 20.0, beta: float = 5.0, eta: float = 10.0, lambda_reg: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.eta = eta
        self.lambda_reg = lambda_reg
        # Trainable Rockafellar-Uryasev VaR threshold v
        self.v = nn.Parameter(torch.tensor(0.02))

    def forward(self, terminal_loss: torch.Tensor, lambda_paths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Pathwise quadratic jump variation Q_T^{(i)} (from Generator)
        d_lam = lambda_paths[:, 1:] - lambda_paths[:, :-1]
        Q_T = torch.sum(d_lam ** 2, dim=1)
        Q_T_norm = (Q_T - Q_T.mean()) / (Q_T.std() + 1e-6)

        # 2. Smoothed Rockafellar-Uryasev CVaR_alpha
        excess = terminal_loss - self.v
        smoothed_hinge = F.softplus(excess, beta=self.tau)
        cvar_term = self.v + smoothed_hinge.mean() / (1.0 - self.alpha)

        # 3. Softmax Extrema Weights
        logits = self.beta * F.relu(excess) + self.eta * F.relu(Q_T_norm)
        w_i = F.softmax(logits, dim=0)

        # 4. Stop-Gradient Extrema Regularizer
        extrema_reg = torch.sum(w_i.detach() * terminal_loss)

        total_loss = cvar_term + self.lambda_reg * extrema_reg
        return total_loss, cvar_term.detach()

def moe_gating_loss(gate_weights: torch.Tensor, lambda_bal: float = 0.20, lambda_ent: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Section 4.3: Mutual Information & Load-Balancing Regularizer.
    gate_weights shape: (B, M, E)
    """
    B, M, E = gate_weights.shape
    g_bar = gate_weights.mean(dim=(0, 1))
    
    # Term 1: Marginal Load-Balancing
    bal_loss = lambda_bal * E * torch.sum((g_bar - (1.0 / E)) ** 2)
    
    # Term 2: Conditional Per-Sample Entropy
    ent_loss = -lambda_ent * torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1).mean()
    
    return bal_loss + ent_loss

class SoftShrinkExpert(nn.Module):
    """Expert 1: Quiescent / Switching-Cost Dead-Zone Policy."""
    def __init__(self, in_dim: int, eps1: float = 0.04):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.eps1 = eps1

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        delta = self.net(I_t)
        # Differentiable SoftShrink
        shrink = torch.sign(delta) * F.relu(torch.abs(delta) - self.eps1)
        return u_prev + shrink

class DiffusiveTrackingExpert(nn.Module):
    """Expert 2: High-Diffusion Continuous Tracking Policy."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        return u_prev + self.net(I_t)

class JumpShockExpert(nn.Module):
    """Expert 3: Discontinuous Jump-Shock Impulse Policy."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, I_t: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        # Bypasses u_prev anchor for rapid state resetting
        return self.net(I_t)

class RegimeGovernedMoE(nn.Module):
    """Section 4: Macro-Gated Mixture of Experts Control Policy."""
    def __init__(self, I_dim: int, Z_dim: int):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(Z_dim, 32), nn.SiLU(),
            nn.Linear(32, 3)
        )
        self.expert1 = SoftShrinkExpert(I_dim)
        self.expert2 = DiffusiveTrackingExpert(I_dim)
        self.expert3 = JumpShockExpert(I_dim)
        self.tau_g = 1.0

    def forward(self, I_t: torch.Tensor, Z_t: torch.Tensor, u_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gating(Z_t)
        g_t = F.softmax(logits / self.tau_g, dim=-1)
        
        u1 = self.expert1(I_t, u_prev)
        u2 = self.expert2(I_t, u_prev)
        u3 = self.expert3(I_t, u_prev)
        
        u_dense = g_t[:, 0:1] * u1 + g_t[:, 1:2] * u2 + g_t[:, 2:3] * u3
        return u_dense, g_t