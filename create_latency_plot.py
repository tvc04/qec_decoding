import numpy as np
import json
import matplotlib.pyplot as plt

# Load JSON file
with open("results/mwpm/results_latency.json", "r") as f:
    data = json.load(f)

# Extract values
physical = [point["physical_error_rate"] for point in data]
latency = [point["average_decoding_latency"] for point in data]

# Sort for clean plotting
physical, latency = zip(*sorted(zip(physical, latency)))

# Plot
plt.figure()
plt.plot(physical, latency, marker='o')

# Log scale on x (y can be linear or log depending on variation)
plt.xscale("log")

plt.xlabel("Physical Error Rate")
plt.ylabel("Average Decoding Latency (seconds)")
plt.title("Decoder Latency vs Physical Error Rate")
plt.grid(True)

plt.show()