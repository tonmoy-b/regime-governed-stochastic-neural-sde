import torch
from regime_governed.envs import SimConfig, sample_macro_regimes, compute_tail_metrics
from regime_governed.generator import DuffiePanSingletonBaseline, ConditionalNeuralJumpSDE, multi_horizon_wasserstein_tail_loss

def main():
    torch.manual_seed(42)
    cfg = SimConfig()
    print(f"[*] Starting Stage 1: Generator Calibration on {cfg.device}")

    dps_engine = DuffiePanSingletonBaseline(cfg)
    model = ConditionalNeuralJumpSDE(cfg.macro_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        X_batch, regime_ids = sample_macro_regimes(cfg.batch_size, cfg.n_steps, cfg.device)
        target_paths = dps_engine.simulate_ground_truth(X_batch)
        
        sim_paths, _ = model.simulate_paths(X_batch, cfg.lambda_0, cfg.dt)
        loss = multi_horizon_wasserstein_tail_loss(sim_paths, target_paths, regime_ids)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | W1 + Tail Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "generator_v1.pt")
    print("[+] Saved generator_v1.pt")

    print("\n=== PAPER TABLE 1: TERMINAL HAZARD RATE METRICS (BPS) ===")
    with torch.no_grad():
        X_eval, reg_eval = sample_macro_regimes(6000, cfg.n_steps, cfg.device)
        true_eval = dps_engine.simulate_ground_truth(X_eval)
        neural_eval, _ = model.simulate_paths(X_eval, cfg.lambda_0, cfg.dt)

        for r, name in enumerate(["Regime 0 (Carry)", "Regime 1 (Roll)", "Regime 2 (Crisis)"]):
            mask = (reg_eval == r)
            m_true = compute_tail_metrics(true_eval[mask])
            m_neur = compute_tail_metrics(neural_eval[mask])
            print(f"{name:18} | True CVaR99: {m_true['CVaR_99']:6.1f} | Neural CVaR99: {m_neur['CVaR_99']:6.1f}")

if __name__ == "__main__":
    main()