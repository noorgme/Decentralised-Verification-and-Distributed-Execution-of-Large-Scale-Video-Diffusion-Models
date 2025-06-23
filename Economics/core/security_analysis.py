#───────── stage1_security.py  (fast + Δ-EV) ─────────#
import numpy as np, itertools, pathlib, math
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ---- economic constants (realistic May-2025) ----------
T_STEPS      = 30
TAO_PER_STEP = 6.2e-6          # 3.3 s A100 step
COST_STEP    = TAO_PER_STEP
GAS_FEE      = 0.0002
E_SUBNET     = 0.005           # 0.5 % of global emission
REWARD_SHARE = 0.41
ETA          = 0.01
BETA         = 0.95
K_CUT        = 60

# ---- helpers ------------------------------------------
def row_norm(A): rs=A.sum(1,keepdims=True); rs[rs==0]=1; return A/rs
def kappa_clip(W,S,k=0.5):
    V,N=W.shape; tot=S.sum(); out=W.copy()
    for j in range(N):
        idx=np.argsort(-W[:,j]); cum=np.cumsum(S[idx])
        thr=W[idx[np.searchsorted(cum,k*tot)],j]
        out[:,j]=np.minimum(W[:,j],thr)
    return out
def p_detect(T,m,k):
    from math import comb
    return 1.0 if k>T-m else 1-comb(T-m,k)/comb(T,k)

# ---------- EV routine (cheat & honest) ----------------
def ev_pair(alpha, f_slash, gamma,
            T=T_STEPS, epochs_tail=K_CUT,
            n_val=5, n_min=10, κ=0.5):
    rng=np.random.default_rng(0)
    S_val=rng.uniform(1,2,n_val)
    W0=row_norm(rng.random((n_val,n_min)))
    bonus=0.5/n_min+1.0/n_min
    k_spot=max(1,int(round(alpha*T)))

    worst=-np.inf
    for m in range(1,T+1):
        pd=p_detect(T,m,k_spot)
        C_comp=COST_STEP*(T-m)
        W=W0.copy(); stake=np.ones(n_min)
        ev_disc=np.zeros(n_min); disc=1
        for _ in range(epochs_tail):
            Wc=kappa_clip(W,S_val,κ)
            rank=(S_val[:,None]*Wc).sum(0)
            share = np.full(n_min,1/n_min) if rank.sum()==0 else rank/rank.sum()
            reward=REWARD_SHARE*E_SUBNET*share
            ev_epoch=reward - C_comp - GAS_FEE - pd*(reward+f_slash*stake)
            ev_disc+=disc*ev_epoch; disc*=BETA
            stake+=reward-pd*f_slash*stake
            caught=rng.random(n_min)<pd
            W[:,caught]*=(1-gamma)
            W[:,~caught]=(1-ETA)*W[:,~caught]+ETA*bonus
            W=row_norm(W)
        tail=disc/(1-BETA)*((1-pd)*reward.mean()-GAS_FEE-pd*f_slash*stake.mean())
        worst=max(worst, ev_disc.sum()+tail)
        if worst>=0: break

    # ----- honest miner EV (m=0, pd=0) ------------------
    R=REWARD_SHARE*E_SUBNET/n_min
    C_h=T*COST_STEP
    ev_h=(R - C_h - GAS_FEE)/(1-BETA)
    return worst, ev_h

# ---------- grid definition -----------------------------
gammas =[0.0,0.5,0.8,1.0]
alphas =np.linspace(0.10,0.60,11)   # audit 10-60 %
slashes=np.linspace(0.00,0.60,13)   # slash 0-60 %

grid=list(itertools.product(range(len(gammas)),
                            range(len(alphas)),
                            range(len(slashes))))
params=[(gammas[g], alphas[a], slashes[f]) for g,a,f in grid]

# ---------- parallel sweep ------------------------------
EV_cheat = np.empty((len(gammas),len(alphas),len(slashes)))
EV_honest= np.empty_like(EV_cheat)

print("⏳ Stage-1 sweep ...")
with tqdm_joblib(tqdm(total=len(params), desc="EV pairs")):
    results = Parallel(n_jobs=-1)(
        delayed(ev_pair)(a,f,g) for g,a,f in params)

for (gi,ai,fi), (c,h) in zip(grid, results):
    EV_cheat[gi,ai,fi]=c
    EV_honest[gi,ai,fi]=h

Delta = EV_honest - EV_cheat

# ---------- save ----------------------------------------
pathlib.Path("sim_data2").mkdir(exist_ok=True)
np.savez("sim_data2/results_stage1.npz",
         gammas=gammas, alphas=alphas, slashes=slashes,
         EV_cheat=EV_cheat, EV_honest=EV_honest, Delta=Delta)
print("Stage-1 data written to sim_data2/results_stage1.npz")
