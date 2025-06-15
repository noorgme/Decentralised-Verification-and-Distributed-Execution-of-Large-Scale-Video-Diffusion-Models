import pandas as pd
import numpy as np

# one_gpu = np.load('fsdp_1gpu_results.npy', allow_pickle=True)
# three_gpu = np.load('fsdp_3gpu_results.npy', allow_pickle=True)
# four_gpu = np.load('fsdp_4gpu_results.npy', allow_pickle=True)

# df_one = pd.DataFrame(list(one_gpu))
# df_three = pd.DataFrame(list(three_gpu))
# df_four = pd.DataFrame(list(four_gpu))

# df = pd.concat([df_one, df_three, df_four], ignore_index=True)

files_to_view = ["fsdp_1gpuB.npy", "fsdp_3gpu.npy", "fsdp_4gpu.npy", "fsdp_5gpu.npy", "fsdp_6gpu.npy"]

parent = "zeroscopeXL_FSDP"
for idx, file in enumerate(files_to_view):
    files_to_view[idx] = f"{parent}/{file}"


all_dfs = []

for file in files_to_view:
    data = np.load(file, allow_pickle=True)
    df = pd.DataFrame(list(data))
    df.to_csv(file.replace('.npy', '.csv'), index=False)
    print(df)
    all_dfs.append(df)


df = pd.concat(all_dfs, ignore_index=True)

import matplotlib.pyplot as plt


plt.style.use('default')


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))


ax1.plot(df['world_size'], df['latency_s'], marker='o', linestyle='-', color='b', linewidth=2)
ax1.set_title('Latency vs World Size', fontsize=12, pad=15)
ax1.set_xlabel('World Size', fontsize=10)
ax1.set_ylabel('Latency (s)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=9)


vram_cols = [col for col in df.columns if 'vram' in col.lower()]


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  
for idx, col in enumerate(vram_cols):
    ax2.plot(df['world_size'], df[col], marker='o', linestyle='-', 
             label=col.replace('_', ' ').title(), linewidth=2, color=colors[idx % len(colors)])

ax2.set_title('VRAM Usage Across GPUs', fontsize=12, pad=15)
ax2.set_xlabel('World Size', fontsize=10)
ax2.set_ylabel('VRAM Usage (GB)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.tick_params(axis='both', which='major', labelsize=9)


plt.tight_layout()

plt.savefig(f"{parent}/performance_analysis.png", bbox_inches='tight', dpi=300)
plt.show()

# Additional VRAM Analysis Plot
plt.figure(figsize=(12, 6))

# Calculate average VRAM usage per GPU
avg_vram = df[vram_cols].mean(axis=1)
std_vram = df[vram_cols].std(axis=1)

plt.errorbar(df['world_size'], avg_vram, yerr=std_vram, 
             fmt='o-', capsize=5, capthick=2, elinewidth=2,
             label='Average VRAM Usage', color='purple')

plt.fill_between(df['world_size'], 
                 avg_vram - std_vram, 
                 avg_vram + std_vram, 
                 alpha=0.2, color='purple')

plt.title('Average VRAM Usage with Standard Deviation', fontsize=12, pad=15)
plt.xlabel('World Size', fontsize=10)
plt.ylabel('VRAM Usage (GB)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the additional analysis plot
plt.savefig(f"{parent}/vram_analysis.png", bbox_inches='tight', dpi=300)
plt.show()