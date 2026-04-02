import numpy as np
import json
import matplotlib.pyplot as plt

# Load JSON file
with open("results/mwpm/results_threshold.json", "r") as f:
    data = json.load(f)

plt.figure()

# Plot curves
for distance, entries in data.items():
    physical = [p["physical_error_rate"] for p in entries]
    logical = [p["logical_error_rate"] for p in entries]

    physical, logical = zip(*sorted(zip(physical, logical)))
    plt.plot(physical, logical, marker='o', label=f"d={distance}")

# ---- Compute threshold (log-spread method) ----
pers = sorted({p["physical_error_rate"] for d in data.values() for p in d})
threshold_estimates = []

for per in pers:
    log_vals = []

    for entries in data.values():
        for point in entries:
            if point["physical_error_rate"] == per:
                val = point["logical_error_rate"]
                if val > 0:
                    log_vals.append(np.log10(val))

    if len(log_vals) > 1:
        spread = max(log_vals) - min(log_vals)
        threshold_estimates.append((per, spread))

threshold = min(threshold_estimates, key=lambda x: x[1])[0]

# ---- Draw vertical dashed line ----
plt.axvline(threshold, linestyle='--', linewidth=2,
            label=f"Threshold ≈ {threshold:.4g}")

# Log scales
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Physical Error Rate")
plt.ylabel("Logical Error Rate")
plt.title("Threshold Plot (Surface Code)")
plt.legend()
plt.grid(True)

plt.show()