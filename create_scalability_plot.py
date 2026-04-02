import numpy as np
import json
import matplotlib.pyplot as plt

# Load JSON file
with open("results/mwpm/results_scalability.json", "r") as f:
    data = json.load(f)

plt.figure()

# Plot each code distance
for distance, entries in data.items():
    physical = [p["physical_error_rate"] for p in entries]
    latency = [p["average_decoding_latency"] for p in entries]

    # Sort for clean curves
    physical, latency = zip(*sorted(zip(physical, latency)))

    plt.plot(physical, latency, marker='o', label=f"d={distance}")

# Log scale on x (like threshold plots)
plt.xscale("log")

# Y-scale: usually linear (latency changes are small)
# Uncomment if needed:
# plt.yscale("log")

plt.xlabel("Physical Error Rate")
plt.ylabel("Average Decoding Latency (seconds)")
plt.title("Scalability Plot (Latency vs Physical Error Rate)")
plt.legend()
plt.grid(True)

plt.show()