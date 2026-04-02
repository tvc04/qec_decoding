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
        decoder = "nueral"

    # Load JSON file
    with open(f"results/{decoder}/results_robustness.json", "r") as f:
        data = json.load(f)

    plt.figure()

    for model_name, entries in data.items():
        physical = [p["physical_error_rate"] for p in entries]
        conditional = [p["conditional_error_rate"] for p in entries]

        physical, conditional = zip(*sorted(zip(physical, conditional)))

        plt.plot(physical, conditional, marker='o', label=model_name)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Conditional Error Rate")
    plt.title("Robustness Plot (Noise Model Comparison)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    generate_plot(1)