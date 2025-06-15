import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Load simulation results
d = np.load("sim_data2/results_stage1.npz")
gammas, alphas, slashes = d["gammas"], d["alphas"], d["slashes"]
EVc, EVh, Delta = d["EV_cheat"], d["EV_honest"], d["Delta"]

# Create output directory
out = pathlib.Path("sim_data2")
out.mkdir(exist_ok=True)

def plot_heatmap(Z, title, file, cmap="coolwarm", levels=20):
    # Handle missing values
    Zm = np.ma.masked_invalid(Z)
    plt.figure(figsize=(5,4))
    plt.title(title)
    
    # Choose plot type based on data
    if Zm.min() == Zm.max():
        plt.imshow(Zm, origin='lower', aspect='auto',
                   extent=[alphas[0], alphas[-1], slashes[0], slashes[-1]],
                   cmap=cmap)
    else:
        plt.contourf(alphas, slashes, Zm, levels, cmap=cmap)
        plt.colorbar()
    
    plt.xlabel("Audit-rate α = k/T")
    plt.ylabel("Slash fraction f")
    plt.tight_layout()
    plt.savefig(out / file, dpi=300)
    plt.close()

# Plot user cost tuning for each gamma
for gi, g in enumerate(gammas):
    # Plot cost sensitivity
    title = f"Cost Sensitivity γ={g}"
    cost_sens = np.gradient(EVc[gi], axis=0)
    plot_heatmap(cost_sens.T, title, f"cost_sensitivity_gamma{g}.png", cmap="RdYlBu_r")
    
    # Plot cost threshold
    title = f"Cost Threshold γ={g}"
    cost_thresh = np.where(EVc[gi] > 0, 1, 0)
    plot_heatmap(cost_thresh.T, title, f"cost_threshold_gamma{g}.png", cmap="binary") 