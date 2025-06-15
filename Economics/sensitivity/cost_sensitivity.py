import numpy as np, itertools, pathlib, math
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# compute economics for different cost steps
# looks at how user fees change with compute costs

# fixed params
T_STEPS   = 30
E_SUBNET  = 0.005            # tao per block
REWARD_SHARE = 0.41
VAL_SHARE    = 1 - REWARD_SHARE
GAS_FEE      = 0.0002
TAO_USD      = 436.0
ETA          = 0.01
BETA         = 0.95
K_CUT        = 60
N_MINERS     = 10
N_VAL        = 5

# cost per step sweep
COST_STEP_SWEEP = [4.0e-6, 5.0e-6, 6.2e-6, 8.0e-6, 1.0e-5]

# param grids
GAMMAS  = [0.0, 0.5, 0.8, 1.0]
ALPHAS  = np.linspace(0.10, 0.60, 11)
SLASHES = np.linspace(0.00, 0.60, 13)

# helper funcs
def row_norm(A):
    rs = A.sum(1, keepdims=True)
    rs[rs == 0] = 1
    return A / rs

def kappa_clip(W, S, k=0.5):
    V, N = W.shape; tot = S.sum(); out = W.copy()
    for j in range(N):
        idx = np.argsort(-W[:, j]); cum = np.cumsum(S[idx])
        thr = W[idx[np.searchsorted(cum, k * tot)], j]
        out[:, j] = np.minimum(W[:, j], thr)
    return out

def p_detect(T, m, k):
    from math import comb
    return 1.0 if k > T - m else 1 - comb(T - m, k) / comb(T, k)

# stage 1 ev calc
def ev_pair(alpha, f_slash, gamma, COST_STEP,
            T=T_STEPS, epochs_tail=K_CUT,
            n_val=N_VAL, n_min=N_MINERS, κ=0.5):
    rng = np.random.default_rng(0)
    S_val = rng.uniform(1, 2, n_val)
    W0 = row_norm(rng.random((n_val, n_min)))
    bonus = 0.5 / n_min + 1.0 / n_min
    k_spot = max(1, int(round(alpha * T)))

    worst = -np.inf
    for m in range(1, T + 1):
        pd = p_detect(T, m, k_spot)
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
        tail = disc / (1 - BETA) * ((1 - pd) * reward.mean() - GAS_FEE - pd * f_slash * stake.mean())
        worst = max(worst, ev_disc.sum() + tail)
        if worst >= 0:
            break

    # honest miner ev
    R = REWARD_SHARE * E_SUBNET / n_min
    C_h = T * COST_STEP
    ev_h = (R - C_h - GAS_FEE) / (1 - BETA)
    return worst, ev_h

# honest drift share
def honest_reward_share(gamma, eta=ETA, K=200):
    rng = np.random.default_rng(0)
    W = rng.random((N_VAL, N_MINERS)); W /= W.sum(axis=1, keepdims=True)
    bonus = 0.5 / N_MINERS + 1.0 / N_MINERS
    for _ in range(K):
        honest = np.zeros(N_MINERS, bool); honest[0] = True
        W[:, honest] = (1 - eta) * W[:, honest] + eta * bonus
        W /= W.sum(axis=1, keepdims=True)
    rank0 = W[:, 0].sum(); total = W.sum()
    return rank0 / total

R_GAMMA = np.array([honest_reward_share(g) for g in GAMMAS])

# main loop
pathlib.Path("sim_data2").mkdir(exist_ok=True)

grid_index = list(itertools.product(range(len(GAMMAS)), range(len(ALPHAS)), range(len(SLASHES))))

for COST_STEP in COST_STEP_SWEEP:
    print(f"\nRecomputing for COST_STEP = {COST_STEP:.2e} TAO")

    params = [(ALPHAS[a], SLASHES[f], GAMMAS[g]) for g, a, f in grid_index]

    EV_cheat = np.empty((len(GAMMAS), len(ALPHAS), len(SLASHES)))
    EV_honest = np.empty_like(EV_cheat)

    with tqdm_joblib(tqdm(total=len(params), desc="EV pairs")):
        results = Parallel(n_jobs=-1)(delayed(ev_pair)(alpha, f_slash, gamma, COST_STEP) for alpha, f_slash, gamma in params)

    for (gi, ai, fi), (c, h) in zip(grid_index, results):
        EV_cheat[gi, ai, fi] = c
        EV_honest[gi, ai, fi] = h
    Delta = EV_honest - EV_cheat

    # stage 2 fee floor
    Dmin_tao = np.full_like(EV_cheat, np.nan)
    for gi, gamma in enumerate(GAMMAS):
        R_h = R_GAMMA[gi]
        miner_term_base = (T_STEPS * COST_STEP + GAS_FEE - REWARD_SHARE * E_SUBNET * R_h) / 0.9
        for ai, alpha in enumerate(ALPHAS):
            k = max(1, int(round(alpha * T_STEPS)))
            C_val = k * COST_STEP
            for fi, f in enumerate(SLASHES):
                if EV_cheat[gi, ai, fi] >= 0:
                    continue
                val_need = max(0.0, (C_val - VAL_SHARE * E_SUBNET / N_VAL) / 0.1)
                Dmin_tao[gi, ai, fi] = max(miner_term_base, val_need)

    Dmin_usd = Dmin_tao * TAO_USD

    # save results
    np.savez(f"sim_data2/sensitivity_C{COST_STEP:.2e}.npz",
             gammas=GAMMAS, alphas=ALPHAS, slashes=SLASHES,
             Delta=Delta, Dmin_usd=Dmin_usd)
    print(f"Saved sim_data2/sensitivity_C{COST_STEP:.2e}.npz")

print("\nFinished COST_STEP sensitivity sweep.") 