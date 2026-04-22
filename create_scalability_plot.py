import sys
import numpy as np
import json
import matplotlib.pyplot as plt


def generate_plot(decoder_type):
    decoder = ""
    title = ""
    if (decoder_type == 1):
        decoder = "mwpm"
        title = "MWPM"
    if (decoder_type == 2):
        decoder = "union"
        title = "Union-Find"
    if (decoder_type == 3):
        decoder = "tensor"
        title = "Tensor Network"
    if (decoder_type == 4):
        decoder = "neural"
        title = "Neural Network"

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
    plt.title(f"Scalability Plot -- {title}")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2  or (int(sys.argv[1]) not in range(1,5)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN)")
    else:
        decoder_type = int(sys.argv[1])
        generate_plot(decoder_type)