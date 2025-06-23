import numpy as np, itertools, pathlib, math, matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

"""e_subnet_sensitivity.py

This script recomputes the economic evaluation pipeline (Stage-1 + Stage-2)
for a range of subnet emissions E_SUBNET and visualises how the minimum user
cost D_min depends on audit-rate α, slash fraction f, and malicious score
penalty γ.

The implementation is a light refactor of figures/parameter_tuning.py and
figures/user_cost_tuning.py, with E_SUBNET exposed as a sweep parameter.
Results and plots are written to the folder  sim_data2/.
"""

# Global constants that do NOT change across the sweep 
T_STEPS      = 30                   # diffusion steps per prompt
TAO_PER_STEP = 6.2e-6              # TAO cost of one diffusion step (A100 @ 3.3 s)
COST_STEP    = TAO_PER_STEP        # same symbol as original scripts
GAS_FEE      = 0.0002              # TAO gas per on-chain settle
REWARD_SHARE = 0.41                # miner's share of subnet reward
VAL_SHARE    = 1 - REWARD_SHARE    # validator share
ETA          = 0.01                # MD-VQS learning rate
BETA         = 0.95                # discount factor
K_CUT        = 60                  # tail epochs simulated after detection window
N_MINERS     = 10
N_VAL        = 5
TAO_USD      = 436.0               # market price for USD conversion (May-2025)

#Parameter grids for the economic sweep 
GAMMAS = [0.0, 0.5, 0.8, 1.0]                       # malicious weight penalty
ALPHAS = np.linspace(0.10, 0.60, 11)                # audit-rate α ∈ [0.1, 0.6]
SLASHES = np.linspace(0.00, 0.60, 13)               # slash fraction f ∈ [0,0.6]

E_SUBNET_SWEEP = [0.003, 0.005, 0.007, 0.010]       # TAO rewarded to subnet per block

# Helper utilities (unchanged logic, just vectorised) 

def row_norm(A: np.ndarray) -> np.ndarray:
    """Normalise rows to probability simplex."""
    rs = A.sum(1, keepdims=True)
    rs[rs == 0] = 1
    return A / rs

def kappa_clip(W: np.ndarray, S: np.ndarray, k: float = 0.5) -> np.ndarray:
    """κ-clipping from the original scripts."""
    V, N = W.shape; tot = S.sum(); out = W.copy()
    for j in range(N):
        idx = np.argsort(-W[:, j]); cum = np.cumsum(S[idx])
        thr = W[idx[np.searchsorted(cum, k * tot)], j]
        out[:, j] = np.minimum(W[:, j], thr)
    return out

def p_detect(T: int, m: int, k: int) -> float:
    from math import comb
    return 1.0 if k > T - m else 1 - comb(T - m, k) / comb(T, k)

# Stage-1: expected value of cheating vs honesty 

def ev_pair(alpha: float, f_slash: float, gamma: float, *, E_SUBNET: float,
            T: int = T_STEPS, epochs_tail: int = K_CUT, n_val: int = N_VAL,
            n_min: int = N_MINERS, κ: float = 0.5) -> tuple[float, float]:
    """Compute discounted EV of (cheating, honest) miner for a given parameter
    triple (alpha, f_slash, gamma) and subnet emission E_SUBNET.
    Returns (EV_cheat, EV_honest).
    """
    rng = np.random.default_rng(0)
    S_val = rng.uniform(1, 2, n_val)
    W0 = row_norm(rng.random((n_val, n_min)))
    bonus = 0.5 / n_min + 1.0 / n_min
    k_spot = max(1, int(round(alpha * T)))

    worst = -np.inf
    for m in range(1, T + 1):        # m = tampered steps
        pd = p_detect(T, m, k_spot)   # detection probability
        C_comp = COST_STEP * (T - m)

        W = W0.copy(); stake = np.ones(n_min)
        ev_disc = np.zeros(n_min); disc = 1.0
        for _ in range(epochs_tail):
            Wc = kappa_clip(W, S_val, κ)
            rank = (S_val[:, None] * Wc).sum(0)
            share = np.full(n_min, 1 / n_min) if rank.sum() == 0 else rank / rank.sum()
            reward = REWARD_SHARE * E_SUBNET * share
            ev_epoch = reward - C_comp - GAS_FEE - pd * (reward + f_slash * stake)
            ev_disc += disc * ev_epoch; disc *= BETA

            stake += reward - pd * f_slash * stake
            caught = rng.random(n_min) < pd
            W[:, caught] *= (1 - gamma)
            W[:, ~caught] = (1 - ETA) * W[:, ~caught] + ETA * bonus
            W = row_norm(W)

        # geometric tail after K_CUT epochs (stationary approx)
        tail = disc / (1 - BETA) * ((1 - pd) * reward.mean() - GAS_FEE - pd * f_slash * stake.mean())
        worst = max(worst, ev_disc.sum() + tail)
        if worst >= 0:
            break

    # honest miner EV (m = 0 -> pd = 0)
    R = REWARD_SHARE * E_SUBNET / n_min
    C_h = T * COST_STEP
    ev_h = (R - C_h - GAS_FEE) / (1 - BETA)
    return worst, ev_h

#  Stage-2: minimum user cost calculation 

def honest_reward_share(gamma: float, eta: float = ETA, K: int = 200) -> float:
    """Simulate MD-VQS drift without slashes to obtain share of an honest miner."""
    rng = np.random.default_rng(0)
    W = rng.random((N_VAL, N_MINERS)); W /= W.sum(axis=1, keepdims=True)
    bonus = 0.5 / N_MINERS + 1.0 / N_MINERS
    for _ in range(K):
        honest = np.zeros(N_MINERS, bool); honest[0] = True
        W[:, honest] = (1 - eta) * W[:, honest] + eta * bonus
        W /= W.sum(axis=1, keepdims=True)
    rank0 = (W[:, 0]).sum(); total = W.sum()
    return rank0 / total

R_GAMMA = np.array([honest_reward_share(g) for g in GAMMAS])

# Main sweep loop 

pathlib.Path("sim_data2").mkdir(exist_ok=True)

for E_SUBNET in E_SUBNET_SWEEP:
    print(f"\n Recomputing economics for E_SUBNET = {E_SUBNET:.4f} TAO ")

    # 1. Stage-1 parallel sweep
    grid = list(itertools.product(range(len(GAMMAS)), range(len(ALPHAS)), range(len(SLASHES))))
    params = [(ALPHAS[a], SLASHES[f], GAMMAS[g]) for g, a, f in grid]

    EV_cheat = np.empty((len(GAMMAS), len(ALPHAS), len(SLASHES)))
    EV_honest = np.empty_like(EV_cheat)

    with tqdm_joblib(tqdm(total=len(params), desc="EV pairs")):
        results = Parallel(n_jobs=-1)(delayed(ev_pair)(a, f, g, E_SUBNET=E_SUBNET) for a, f, g in params)

    for (gi, ai, fi), (c, h) in zip(grid, results):
        EV_cheat[gi, ai, fi] = c
        EV_honest[gi, ai, fi] = h

    Delta = EV_honest - EV_cheat

    # 2. Stage-2 user-cost bounds
    shape = EV_cheat.shape
    Dmin_tao = np.full(shape, np.nan)

    for gi, gamma in enumerate(GAMMAS):
        R_h = R_GAMMA[gi]
        miner_term = (T_STEPS * COST_STEP + GAS_FEE - REWARD_SHARE * E_SUBNET * R_h) / 0.9  # split_miner fixed at 0.9
        for ai, alpha in enumerate(ALPHAS):
            k = max(1, int(round(alpha * T_STEPS)))
            C_val = k * COST_STEP
            for fi, f in enumerate(SLASHES):
                if EV_cheat[gi, ai, fi] >= 0:
                    continue  # no need for fee if cheating not profitable
                val_need = max(0.0, (C_val - VAL_SHARE * E_SUBNET / N_VAL) / 0.1)  # split_val 0.1
                Dmin_tao[gi, ai, fi] = max(miner_term, val_need)

    Dmin_usd = Dmin_tao * TAO_USD

    # 3. Persist numpy archive
    np.savez(f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz",
             gammas=GAMMAS, alphas=ALPHAS, slashes=SLASHES,
             Delta=Delta, Dmin_usd=Dmin_usd)

    print("Saved sim_data2/sensitivity_E{:.4f}.npz".format(E_SUBNET))

print("\nFinished recomputation for all E_SUBNET values.")
print("To generate visualisations, run:  python figures/e_subnet_plots.py") 
