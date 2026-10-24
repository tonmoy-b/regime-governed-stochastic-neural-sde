# Regime-Governed Stochastic Control: Exogenous Neural Jump-Diffusions and Extrema-Regularized Mixture-of-Experts Policies

[![SSRN DOI](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.7661600-blue.svg)](https://doi.org/10.2139/ssrn.7661600)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%252B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%25+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **Regime-Governed Stochastic Control**, establishing an endogenous governance loop between an Exogenous-Conditioned Neural Jump-Diffusion SDE generator and a downstream Macro-Gated Mixture-of-Experts (MoE) control policy for risk-sensitive tail mitigation.

> **Citation / Pre-print:**
> Bhattacharya, T. (2026). *Regime-Governed Stochastic Control: Exogenous Neural Jump-Diffusions and Extrema-Regularized Mixture-of-Experts Policies*. SSRN Electronic Journal. 
> DOI: [10.2139/ssrn.7661600](https://doi.org/10.2139/ssrn.7661600)

---

## 🏗️ Repository Architecture

```text
regime-governed-stochastic-neural-sde/
├── src/
│   └── regime_governed/
│       ├── __init__.py
│       ├── envs.py         # Multi-regime macroeconomic environment & simulation config
│       ├── generator.py    # Conditional Neural Jump-SDE with Straight-Through Estimator (STE)
│       └── control.py      # Macro-Gated MoE, Monolithic Baseline, & Extrema-CVaR Loss
├── experiments/
│   ├── 01_train_generator.py # Stage 1: Generator calibration via Multi-Horizon Wasserstein-1
│   ├── 02_train_moe.py       # Stage 2: Causal MoE control optimization under CVaR_99
│   └── 03_evaluate_baselines.py # Stage 3: Out-of-sample evaluation & Table 2 generation
├── requirements.txt
└── README.md
```

## Key Mathematical Components
1. Conditional Neural Jump-Diffusion SDE: Formulates a positivity-preserving hazard rate process $\lambda_t$ driven by neural drift $\mu_\theta$, square-root diffusion $\sigma_\theta$, and an inhomogeneous Poisson jump process $N_t$ calibrated via a Straight-Through Estimator (STE).
2. Extrema-Regularized Rockafellar–Uryasev $\mathrm{CVaR}_\alpha$ Objective: Combines variational dual $\mathrm{CVaR}_{0.99}$ minimization with a Softmax Extrema Regularizer weighted by pathwise quadratic jump variation $Q_T = \sum_m \|\Delta Y_{t_m}\|_2^2$ to eliminate tail-gradient starvation.
3. Macro-Gated Mixture of Experts (MoE): Partitions state information into local plant coordinates ($\mathcal{I}_t$) and regime routing context ($\mathcal{Z}_t = X_t \oplus \mathcal{M}_t$). Features mutual-information load balancing ($\mathcal{L}_{\text{gate}}$) and expert-specific structural inductive biases ($\mathrm{SoftShrink}$ dead-zone, diffusive tracker, unanchored jump reset).

## Installation & Setup

1. Clone the repository:
```bash
git clone [https://github.com/tonmoy-b/regime-governed-stochastic-neural-sde.git](https://github.com/tonmoy-b/regime-governed-stochastic-neural-sde.git)
cd regime-governed-stochastic-neural-sde
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Experimental Pipeline
Execute the three-stage pipeline to calibrate the generator, optimize the MoE controller, and generate out-of-sample benchmarking tables.

1. Calibrate the Neural Jump-SDE Generator
```bash
python experiments/01_train_generator.py
```
> To train generator_v1.pt on multi-regime macroeconomic trajectories using multi-horizon Wasserstein-1 tail objectives

2. Train the Macro-Gated MoE Controller
```bash
python experiments/02_train_moe.py
```
> To optimize expert weights $\Psi$, while gating network $\omega$, and auxiliary VaR threshold $v_{offset}$ under causal rollout conditions, and saving moe_policy_v1.pt

3. Evaluate Baselines & Generate Table 2
```bash 
python experiments/03_evaluate_baselines.py
```
> This compares the Monolithic MLP Baseline against Dense MoE and Top-1 Sparse MoE out-of-sample across 10,000 fresh trajectories

## Out-of-Sample Results
Out-of-sample $\mathrm{CVaR}_{0.99}$ Net Hedging Loss (BPS) under strict $\mathcal{G}_t$-adapted causal rollouts:

| Macro Regime | Monolithic MLP | MoE (Dense) | MoE (Top-1 Sparse) | MoE Gain (Dense) |
| --- | --- | --- | --- | --- |
| **Regime 0 (Calm Carry)** | 1007.84 | **313.36** | 1941.52 | 68.9% |
| **Regime 1 (Roll Stress)** | 3242.10 | **433.00** | 8901.41 | 86.6% |
| **Regime 2 (Crisis Jump)** | 5525.82 | **813.18** | 18412.67 | 85.3% |

## License & Attribution

Distributed under the MIT License. 

Disclosure of Generative AI Assistance: During the preparation of the accompanying manuscript, generative AI (Google Gemini) was utilized for language editing, mathematical notation standardization, and boilerplate PyTorch simulation scaffolding. All core theoretical formulations, system architectures, empirical benchmarks, and validations were independently executed and verified by the author.