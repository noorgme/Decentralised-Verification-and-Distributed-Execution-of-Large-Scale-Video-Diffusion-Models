# plot stage 1 cubes
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math

d = np.load("sim_data2/results_stage1.npz")
gammas, alphas, slashes = d["gammas"], d["alphas"], d["slashes"]
EVc, EVh, Delta = d["EV_cheat"], d["EV_honest"], d["Delta"]

out = pathlib.Path("sim_data2")
out.mkdir(exist_ok=True)

# helper for heatmaps
def safe_heat(Z, title, file, cmap="coolwarm", levels=20):
    Zm = np.ma.masked_invalid(Z)
    plt.figure(figsize=(5,4))
    plt.title(title)
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

# plots for each gamma
for gi, g in enumerate(gammas):
    title = f"Worst-case cheat EV∞ γ={g}"
    safe_heat(EVc[gi].T, title, f"heat_cheat_gamma{g}.png")

for gi, g in enumerate(gammas):
    title = f"Δ-EV (Honest − Cheat) γ={g}"
    safe_heat(Delta[gi].T, title, f"heat_delta_gamma{g}.png", cmap="PuBuGn")

# honest miner EV plots
for gi, g in enumerate(gammas):
    title = f"Honest Miner EV∞ γ={g}"
    safe_heat(EVh[gi].T, title, f"heat_honest_gamma{g}.png", cmap="viridis")

print("Stage-1 figures written to sim_data2/")

