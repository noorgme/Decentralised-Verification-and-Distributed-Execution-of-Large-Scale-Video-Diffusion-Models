import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Simulation settings
TRIALS = 1000
TAMPER_RATES = np.linspace(0, 1, 20)
DETECTION_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

def run_detection_sim(tamper_rate, threshold):
    # Simulate detection for given tamper rate and threshold
    detections = 0
    for _ in range(TRIALS):
        if np.random.random() < tamper_rate:
            if np.random.random() < threshold:
                detections += 1
    return detections / TRIALS

# Run simulations for each threshold
results = {}
for threshold in DETECTION_THRESHOLDS:
    rates = [run_detection_sim(rate, threshold) for rate in TAMPER_RATES]
    results[threshold] = rates

# Plot results
plt.figure(figsize=(8, 5))
for threshold, rates in results.items():
    plt.plot(TAMPER_RATES, rates, label=f'Threshold {threshold}')
plt.title("Tamper Detection Analysis")
plt.xlabel("Actual Tamper Rate")
plt.ylabel("Detection Rate")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("tamper_rate_detection.png", dpi=300)
plt.close()
