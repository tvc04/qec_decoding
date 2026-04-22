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
    plt.title(f"Robustness Plot -- {title}")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 2  or (int(sys.argv[1]) not in range(1,5)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN)")
    else:
        decoder_type = int(sys.argv[1])
        generate_plot(decoder_type)