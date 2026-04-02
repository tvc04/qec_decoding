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
    with open(f"results/{decoder}/results_scalability.json", "r") as f:
        data = json.load(f)

    plt.figure()

    for distance, entries in data.items():
        physical = [p["physical_error_rate"] for p in entries]
        latency = [p["average_decoding_latency"] for p in entries]

        physical, latency = zip(*sorted(zip(physical, latency)))

        plt.plot(physical, latency, marker='o', label=f"d={distance}")

    plt.xscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Average Decoding Latency (seconds)")
    plt.title("Scalability Plot (Latency vs Physical Error Rate)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    generate_plot(1)