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
    with open(f"results/{decoder}/results_latency.json", "r") as f:
        data = json.load(f)

    physical = [point["physical_error_rate"] for point in data]
    latency = [point["average_decoding_latency"] for point in data]

    physical, latency = zip(*sorted(zip(physical, latency)))

    plt.figure()
    plt.plot(physical, latency, marker='o')

    plt.xscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Average Decoding Latency (seconds)")
    plt.title("Decoder Latency vs Physical Error Rate")
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2  or (int(sys.argv[1]) not in range(1,5)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN)")
    else:
        decoder_type = int(sys.argv[1])
        generate_plot(decoder_type)