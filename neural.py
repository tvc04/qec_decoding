import stim
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
import time


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 1000000
error_rates = [5*i/100000 for i in range(1,41)] # 0.00005 - 0.002
nn_dir = "nn_models"


# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------

class DecoderNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def generate_data(distance, shots, p):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=synd_rounds,
        after_clifford_depolarization=p
    )

    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots, append_observables=True)

    num_detectors = circuit.num_detectors
    syndromes = samples[:, :num_detectors]
    observables = samples[:, num_detectors:]

    return syndromes.astype(np.float32), observables.astype(np.float32)

def load_model(distance, input_size, d=0, m=0, r=0):
    model = DecoderNN(input_size)
    path = os.path.join(nn_dir, f"decoder_{d}{m}{r}_d{distance}.pt")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def nn_decode(model, syndromes):
    with torch.no_grad():
        preds = model(torch.tensor(syndromes)).squeeze()
        return (preds > 0.5).cpu().numpy()



# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

def correctness(depolarization = 0, measure = 0, reset = 0):
    with open(os.path.join(nn_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    results = []

    input_size = metadata[f"{depolarization}{measure}{reset}"][f"{dist}"]["input_size"]
    model = load_model(dist, input_size, depolarization, measure, reset)

    for per in error_rates:

        x, y = generate_data(dist, shots, per)

        preds = nn_decode(model, x)

        mask = np.any(x != 0, axis=1)

        filtered_preds = preds[mask]
        filtered_obs = y.squeeze()[mask]

        cond_error_rate = np.mean(filtered_preds != filtered_obs)

        results.append({
            "physical_error_rate": per,
            "conditional_error_rate": cond_error_rate
        })

        print()
        print(f"Physical Error Rate: {per}")
        print(f"Conditional Error Rate: {cond_error_rate:.8f}")
        print()

    return results

def latency(distance = dist):
    with open(os.path.join(nn_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    results = []

    input_size = metadata["000"][f"{distance}"]["input_size"]
    model = load_model(distance, input_size)

    for per in error_rates:

        x, _ = generate_data(distance, shots, per)

        X = torch.tensor(x)

        start = time.perf_counter()
        model(X)
        end = time.perf_counter()

        avg_latency = (end - start) / shots

        results.append({
            "physical_error_rate": per,
            "average_decoding_latency": avg_latency
        })

        print()
        print(f"Distance: {distance}, Physical Error Rate: {per}")
        print(f"Average Decoding Latency: {avg_latency:.10f}")
        print()

    return results

def threshold():
    with open(os.path.join(nn_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    results = {}

    for dist in range(3,10,2):

        input_size = metadata["000"][f"{dist}"]["input_size"]
        model = load_model(dist, input_size)

        results[f'{dist}'] = []
        dist_results = []

        for per in error_rates:

            x, y = generate_data(dist, shots, per)

            preds = nn_decode(model, x)

            log_error_rate = np.mean(preds != y.squeeze())

            dist_results.append({
                "physical_error_rate": per,
                "logical_error_rate": log_error_rate
            })

            print()
            print(f"Distance: {dist}, Physical Error Rate: {per}")
            print(f"Logical Error Rate: {log_error_rate:.8f}")
            print()

        results[f'{dist}'].extend(dist_results)

    return results

def robustness():
    results = {}
    print("\n--------- Control ---------\n")
    results['control'] = []
    control_results = correctness()
    results['control'].extend(control_results)
    print("\n--------- Depolarization ---------\n")
    results['depolarization'] = []
    depolarization_results = correctness(depolarization=1)
    results['depolarization'].extend(depolarization_results)
    print("\n--------- Measure ---------\n")
    results['measure'] = []
    measure_results = correctness(measure=1)
    results['measure'].extend(measure_results)
    print("\n--------- Reset ---------\n")
    results['reset'] = []
    reset_results = correctness(reset=1)
    results['reset'].extend(reset_results)
    print("\n--------- All Errors ---------\n")
    results['all'] = []
    all_results = correctness(depolarization=1, measure=1, reset=1)
    results['all'].extend(all_results)

    return results

def scalability():
    results = {}
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = latency(dist)
        results[f'{dist}'].extend(dist_results)
    
    return results



def neural_test(test_num):
    results = []
    output_file = ""
    if test_num == 1:
        results = correctness()
        output_file = "results/neural/results_correctness.json"
    if test_num == 2:
        results = latency()
        output_file = "results/neural/results_latency.json"
    if test_num == 3:
        results = threshold()
        output_file = "results/neural/results_threshold.json"
    if test_num == 4:
        results = robustness()
        output_file = "results/neural/results_robustness.json"
    if test_num == 5:
        results = scalability()
        output_file = "results/neural/results_scalability.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    

if __name__ == '__main__':
    if len(sys.argv) != 2 or int(sys.argv[1]) not in range(1,6):
        print("Specify Test Type (1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
    else:
        test_type = int(sys.argv[1])
        neural_test(test_type)