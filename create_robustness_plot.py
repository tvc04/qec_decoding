import numpy as np
import json
import matplotlib.pyplot as plt

# Load JSON file
with open("results/mwpm/results_robustness.json", "r") as f:
    data = json.load(f)

plt.figure()

# Plot each noise model
for model_name, entries in data.items():
    physical = [p["physical_error_rate"] for p in entries]
    conditional = [p["conditional_error_rate"] for p in entries]

    # Sort for clean curves
    physical, conditional = zip(*sorted(zip(physical, conditional)))

    plt.plot(physical, conditional, marker='o', label=model_name)

# Log-log scale (same as threshold plots)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Physical Error Rate")
plt.ylabel("Conditional Error Rate")
plt.title("Robustness Plot (Noise Model Comparison)")
plt.legend()
plt.grid(True)

plt.show()