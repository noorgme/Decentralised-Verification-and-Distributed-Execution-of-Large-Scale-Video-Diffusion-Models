import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Load simulation data
d = np.load("sim_data2/results_stage1.npz")
gammas, alphas, slashes = d["gammas"], d["alphas"], d["slashes"]
EVc, EVh, Delta = d["EV_cheat"], d["EV_honest"], d["Delta"]

# Setup output directory
out = pathlib.Path("sim_data2")
out.mkdir(exist_ok=True)

def plot_heatmap(Z, title, file, cmap="coolwarm", levels=20):
    # Handle invalid values
    Zm = np.ma.masked_invalid(Z)
    plt.figure(figsize=(5,4))
    plt.title(title)
    
    # Plot either contour or image based on data range
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

# Plot cost per step for each gamma
for gi, g in enumerate(gammas):
    title = f"Cost per Step γ={g}"
    plot_heatmap(EVc[gi].T, title, f"cost_step_gamma{g}.png", cmap="YlOrRd")

# Plot cost efficiency for each gamma
for gi, g in enumerate(gammas):
    title = f"Cost Efficiency γ={g}"
    plot_heatmap(Delta[gi].T, title, f"cost_efficiency_gamma{g}.png", cmap="RdYlBu_r")

