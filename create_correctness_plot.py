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
    with open(f"results/{decoder}/results_correctness.json", "r") as f:
        data = json.load(f)

    physical = [point["physical_error_rate"] for point in data]
    conditional = [point["conditional_error_rate"] for point in data]

    physical, conditional = zip(*sorted(zip(physical, conditional)))

    plt.figure()
    plt.plot(physical, conditional, marker='o')

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Conditional Error Rate")
    plt.title("Decoder Correctness vs Physical Error Rate")
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    generate_plot(1)