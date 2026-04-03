import sys
import numpy as np
import json
import matplotlib.pyplot as plt


def generate_plot(decoder_type):
    decoder = ""
    if (decoder_type == 1):
        decoder = "mwpm"
    if (decoder_type == 2):
        decoder = "union"
    if (decoder_type == 3):
        decoder = "tensor"
    if (decoder_type == 4):
        decoder = "neural"

    # Load JSON file
    with open(f"results/{decoder}/results_threshold.json", "r") as f:
        data = json.load(f)

    plt.figure()

    for distance, entries in data.items():
        physical = [p["physical_error_rate"] for p in entries]
        logical = [p["logical_error_rate"] for p in entries]

        physical, logical = zip(*sorted(zip(physical, logical)))
        plt.plot(physical, logical, marker='o', label=f"d={distance}")

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

    plt.axvline(threshold, linestyle='--', linewidth=2,
                label=f"Threshold ≈ {threshold:.4g}")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Logical Error Rate")
    plt.title("Threshold Plot (Surface Code)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2  or (int(sys.argv[1]) not in range(1,5)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN)")
    else:
        decoder_type = int(sys.argv[1])
        generate_plot(decoder_type)