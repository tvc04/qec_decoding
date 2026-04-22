import sys
import numpy as np
import json
import matplotlib.pyplot as plt


def generate_plot(decoder_type):
    decoder = ""
    color = ""
    marker = ''
    title = ""
    if (decoder_type == 1):
        decoder = "mwpm"
        color = "blue"
        marker = 'o'
        title = "MWPM"
    if (decoder_type == 2):
        decoder = "union"
        color = "red"
        marker = 's'
        title = "Union-Find"
    if (decoder_type == 3):
        decoder = "tensor"
        color = "green"
        marker = '^'
        title = "Tensor Network"
    if (decoder_type == 4):
        decoder = "neural"
        color = "orange"
        marker = '*'
        title = "Neural Network"

    # Load JSON file
    with open(f"results/{decoder}/results_correctness.json", "r") as f:
        data = json.load(f)

    physical = [point["physical_error_rate"] for point in data]
    conditional = [point["conditional_error_rate"] for point in data]

    physical, conditional = zip(*sorted(zip(physical, conditional)))

    plt.figure()
    plt.plot(physical, conditional, marker=marker, color=color)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical Error Rate")
    plt.ylabel("Conditional Error Rate")
    plt.title(f"Conditional Correctness -- {title}")
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2  or (int(sys.argv[1]) not in range(1,5)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN)")
    else:
        decoder_type = int(sys.argv[1])
        generate_plot(decoder_type)