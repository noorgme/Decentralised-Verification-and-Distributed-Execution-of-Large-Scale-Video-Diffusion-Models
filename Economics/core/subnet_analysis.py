import numpy as np, matplotlib.pyplot as plt, pathlib
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager

# use times new roman if available
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "figure.dpi": 150,
})


pathlib.Path("sim_data2").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# config
E_SUBNET_SWEEP = [0, 0.001, 0.003, 0.005, 0.007, 0.010]

gamma_pick  = 0.8   # fixed gamma
f_pick      = 0.3   # fixed f
alpha_pick  = 0.3   # fixed alpha

# check which data files exist
valid_E = []
for E_SUBNET in E_SUBNET_SWEEP:
    path = f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz"
    if pathlib.Path(path).exists():
        valid_E.append(E_SUBNET)
    else:
        print(f"Missing data file for E_SUBNET={E_SUBNET:.4f}; skipping in plots.")

if not valid_E:
    raise RuntimeError("No sensitivity data files found for any E_SUBNET in the sweep. Run e_subnet_sensitivity.py first.")

# load params from first valid file
sample = np.load(f"sim_data2/sensitivity_E{valid_E[0]:.4f}.npz")
alphas  = sample["alphas"]
slashes = sample["slashes"]
gammas  = sample["gammas"]

# get indices for chosen params
gi = np.argmin(np.abs(gammas  - gamma_pick))
fi = np.argmin(np.abs(slashes - f_pick))
ai = np.argmin(np.abs(alphas  - alpha_pick))

colors = sns.color_palette("husl", len(valid_E))

# plot D_min vs alpha
plt.figure(figsize=(8, 5))
for idx, (E_SUBNET, col) in enumerate(zip(valid_E, colors)):
    d = np.load(f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz")
    D_curve = d["Dmin_usd"][gi, :, fi]
    plt.plot(alphas, D_curve, "-o", color=col, lw=2,
             label=f"E={E_SUBNET:.3f} TAO", markersize=6,
             markerfacecolor="white", markeredgewidth=1.5)
plt.title(r"$D_{min}$ vs audit-rate $\alpha$ (f=0.3, γ=0.8)")
plt.xlabel(r"audit-rate $\alpha = k/T$")
plt.ylabel("Minimum user cost (USD)")
plt.legend(); plt.tight_layout()
plt.savefig("sim_data2/Fig1_Dmin_vs_alpha.png", dpi=300)
plt.close()

# plot heatmaps
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.ravel()
vmin, vmax = None, None

# get global color limits
for E_SUBNET in valid_E:
    d = np.load(f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz")
    Z = d["Dmin_usd"][gi]
    finite = np.isfinite(Z)
    if vmin is None:
        vmin, vmax = Z[finite].min(), Z[finite].max()
    else:
        vmin, vmax = min(vmin, Z[finite].min()), max(vmax, Z[finite].max())

for ax, E_SUBNET in zip(axes, valid_E):
    d = np.load(f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz")
    Z = d["Dmin_usd"][gi]
    im = ax.imshow(Z.T, origin="lower", aspect="auto",
                   extent=[alphas[0], alphas[-1], slashes[0], slashes[-1]],
                   cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(f"E = {E_SUBNET:.3f} TAO")
    ax.set_xlabel(r"audit-rate α")
    if ax is axes[0] or ax is axes[2]:
        ax.set_ylabel("slash fraction f")

cbar = fig.colorbar(im, ax=axes[:len(valid_E)], orientation="vertical", shrink=0.83)
cbar.set_label("Minimum user cost (USD)")
fig.suptitle(r"Heat-map of $D_{min}$ across (α, f)  at γ=0.8", y=0.94, fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig("sim_data2/Fig2_Dmin_heatmaps.png", dpi=300)
plt.close(fig)

print("Fig2_Dmin_heatmaps.png")

# plot D_min vs E_SUBNET
plt.figure(figsize=(8, 5))
E_vals, D_vals = [], []
for E_SUBNET in valid_E:
    d = np.load(f"sim_data2/sensitivity_E{E_SUBNET:.4f}.npz")
    D_vals.append(d["Dmin_usd"][gi, ai, fi])
    E_vals.append(E_SUBNET)
plt.plot(E_vals, D_vals, "o-", color="#e41a1c", lw=2, markersize=6,
         markerfacecolor="white", markeredgewidth=1.5)
plt.title(r"Sensitivity of $D_{min}$ to $E_{subnet}$ (α=0.3, f=0.3, γ=0.8)")
plt.xlabel(r"Subnet emission $E_{subnet}$ (TAO per block)")
plt.ylabel("Minimum user cost (USD)")
plt.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("sim_data2/Fig3_Dmin_vs_Esubnet.png", dpi=300)
plt.close()


d = np.load("sim_data2/results_stage1.npz")
gammas, alphas, slashes = d["gammas"], d["alphas"], d["slashes"]
EVc, EVh, Delta = d["EV_cheat"], d["EV_honest"], d["Delta"]

out = pathlib.Path("sim_data2")
out.mkdir(exist_ok=True)

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

for gi, g in enumerate(gammas):
    title = f"Subnet Economics γ={g}"
    safe_heat(EVc[gi].T, title, f"subnet_economics_gamma{g}.png", cmap="YlOrRd")

for gi, g in enumerate(gammas):
    title = f"Network Efficiency γ={g}"
    safe_heat(Delta[gi].T, title, f"network_efficiency_gamma{g}.png", cmap="RdYlBu_r") 