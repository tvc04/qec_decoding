import numpy as np
import json
import matplotlib.pyplot as plt

# Load JSON file
with open("results/mwpm/results_correctness.json", "r") as f:
    data = json.load(f)

# Extract values
physical = [point["physical_error_rate"] for point in data]
conditional = [point["conditional_error_rate"] for point in data]

# Sort (important for clean plotting)
physical, conditional = zip(*sorted(zip(physical, conditional)))

# Plot
plt.figure()
plt.plot(physical, conditional, marker='o')

# Log-log scale (important!)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Physical Error Rate")
plt.ylabel("Conditional Error Rate")
plt.title("Decoder Correctness vs Physical Error Rate")
plt.grid(True)

plt.show()