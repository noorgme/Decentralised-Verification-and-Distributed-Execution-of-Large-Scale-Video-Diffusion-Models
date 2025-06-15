import numpy as np, itertools, pathlib, math
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt

# Economic parameters (May 2025 estimates)
TAO_PRICE = 0.5  # USD
COST_PER_STEP = 5.0e-6  # TAO
REWARD_PER_STEP = 1.0e-5  # TAO
SLASH_FRACTION = 0.3  # 30% slash

# Simulation parameters
GAMMAS = np.linspace(0.1, 0.9, 9)  # Discount factors
ALPHAS = np.linspace(0.1, 0.9, 9)  # Audit rates
SLASHES = np.linspace(0.1, 0.9, 9)  # Slash fractions

def compute_ev_cheat(gamma, alpha, slash):
    # Expected value for cheating miners
    p_detected = alpha
    reward = REWARD_PER_STEP
    cost = COST_PER_STEP
    slash_amount = reward * slash
    
    ev_cheat = (1 - p_detected) * reward - cost - p_detected * slash_amount
    return ev_cheat / (1 - gamma)

def compute_ev_honest(gamma, alpha, slash):
    # Expected value for honest miners
    reward = REWARD_PER_STEP
    cost = COST_PER_STEP
    ev_honest = reward - cost
    return ev_honest / (1 - gamma)

def compute_delta(gamma, alpha, slash):
    # Difference between honest and cheating EV
    ev_honest = compute_ev_honest(gamma, alpha, slash)
    ev_cheat = compute_ev_cheat(gamma, alpha, slash)
    return ev_honest - ev_cheat

# Run simulations in parallel
with Parallel(n_jobs=-1) as parallel:
    # Compute EV for cheating miners
    ev_cheat = parallel(
        delayed(compute_ev_cheat)(g, a, s)
        for g, a, s in itertools.product(GAMMAS, ALPHAS, SLASHES)
    )
    ev_cheat = np.array(ev_cheat).reshape(len(GAMMAS), len(ALPHAS), len(SLASHES))
    
    # Compute EV for honest miners
    ev_honest = parallel(
        delayed(compute_ev_honest)(g, a, s)
        for g, a, s in itertools.product(GAMMAS, ALPHAS, SLASHES)
    )
    ev_honest = np.array(ev_honest).reshape(len(GAMMAS), len(ALPHAS), len(SLASHES))
    
    # Compute Delta (honest - cheat)
    delta = parallel(
        delayed(compute_delta)(g, a, s)
        for g, a, s in itertools.product(GAMMAS, ALPHAS, SLASHES)
    )
    delta = np.array(delta).reshape(len(GAMMAS), len(ALPHAS), len(SLASHES))

# Save results
out = pathlib.Path("sim_data2")
out.mkdir(exist_ok=True)
np.savez(out / "results_stage1.npz",
         gammas=GAMMAS, alphas=ALPHAS, slashes=SLASHES,
         EV_cheat=ev_cheat, EV_honest=ev_honest, Delta=delta)

# Parameters to tune
PARAMS = {
    'alpha': np.linspace(0.1, 0.9, 9),
    'beta': np.linspace(0.1, 0.9, 9),
    'gamma': np.linspace(0.1, 0.9, 9)
}

def evaluate_params(alpha, beta, gamma):
    return alpha * beta * gamma

# Calculate results
results = np.zeros((len(PARAMS['alpha']), len(PARAMS['beta']), len(PARAMS['gamma'])))
for i, a in enumerate(PARAMS['alpha']):
    for j, b in enumerate(PARAMS['beta']):
        for k, g in enumerate(PARAMS['gamma']):
            results[i,j,k] = evaluate_params(a, b, g)

# Plot heatmap for alpha vs beta (averaged over gamma)
plt.figure(figsize=(8, 6))
mean_results = np.mean(results, axis=2)
plt.imshow(mean_results, origin='lower', aspect='auto',
           extent=[PARAMS['beta'][0], PARAMS['beta'][-1],
                  PARAMS['alpha'][0], PARAMS['alpha'][-1]])
plt.colorbar(label='Performance Score')
plt.title('Parameter Tuning Analysis')
plt.xlabel('Beta')
plt.ylabel('Alpha')
plt.tight_layout()
plt.savefig('parameter_tuning.png', dpi=300)
plt.close()
