import torch
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimConfig:
    dt: float = 1.0 / 252.0        # Daily steps
    n_steps: int = 63              # 1 Quarter horizon
    batch_size: int = 2048
    lambda_0: float = 0.0300       # 300 bps initial hazard
    macro_dim: int = 3             # [CDX_Spread_Z, VIX_Norm, Yield_Slope]
    hidden_dim: int = 64
    lr: float = 2e-3
    epochs: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def sample_macro_regimes(batch_size: int, n_steps: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates exogenous macro indicator trajectories X_t.
    Regime 0: Calm Carry
    Regime 1: Liquidity Stress
    Regime 2: Jump-to-Default Crisis
    """
    regime_ids = torch.randint(0, 3, (batch_size,), device=device)
    centroids = torch.tensor([
        [-0.5, 0.15,  0.50],
        [ 0.8, 0.45,  0.00],
        [ 2.2, 0.85, -0.60],
    ], device=device)

    base_x = centroids[regime_ids].unsqueeze(1).expand(batch_size, n_steps, 3)
    noise = 0.08 * torch.randn(batch_size, n_steps, 3, device=device)
    return base_x + noise, regime_ids

def compute_tail_metrics(paths: torch.Tensor) -> dict:
    """Computes terminal statistics for Table 1 evaluation."""
    terminal_bps = paths[:, -1] * 1e4
    sorted_bps, _ = torch.sort(terminal_bps)
    n = len(sorted_bps)
    return {
        "Mean": terminal_bps.mean().item(),
        "Std": terminal_bps.std().item(),
        "VaR_95": sorted_bps[int(0.95 * n)].item(),
        "CVaR_99": sorted_bps[int(0.99 * n):].mean().item(),
    }