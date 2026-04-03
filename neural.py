import sys
import stim
import pymatching
import numpy as np
import time
import json


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 100000


# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------


# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

# Plot decoder's conditional correctness as per increases
# Conditional correctness: prediction correctness in cases with errors
def correctness(depolarization = 0, measure = 0, reset = 0):
    results = []
    return results

# Plot decoding time as per increases later (distance is a part of scalability)
def latency(distance = dist):
    results = []
    return results

# Plot logical error rate as distnace increases
def threshold():
    results = {}
    return results

# Include different error models and test correctness
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

# Track qubit counts and decoding latency -> space time measurements
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
    if len(sys.argv) > 2 or int(sys.argv[1]) not in range(1,6):
        print("Specify Test Type (1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
    else:
        test_type = int(sys.argv[1])
        neural_test(test_type)