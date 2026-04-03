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

# Creates surface code circuit
def surface_code(distance, rounds, phys_error_rate, depolarization = 0, measure = 0, reset = 0):
    sc = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=phys_error_rate,
        before_round_data_depolarization=phys_error_rate*depolarization,
        before_measure_flip_probability=phys_error_rate*measure,
        after_reset_flip_probability=phys_error_rate*reset
    )
    return sc

# Creates decoding graph for pymatching
def decoder(surface_code):
    error_model = surface_code.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(error_model)
    return matching

# Generates sample error syndrome
def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=True)


# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

# Plot decoder's conditional correctness as per increases
# Conditional correctness: prediction correctness in cases with errors
def correctness(depolarization = 0, measure = 0, reset = 0):
    results = []
    for i in range(1,21): # 0.0005 - 0.01
        per = 5*i/10000
        code = surface_code(dist, synd_rounds, per, depolarization, measure, reset)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        predictions = dc.decode_batch(detections)
        errors = np.any(detections != 0, axis=1)

        fails = 0
        total = 0
        for i in range(shots):
            if errors[i]:
                total += 1
                if not np.array_equal(predictions[i], observed_flips[i]):
                    fails += 1
        
        cond_error_rate = 0
        if total != 0:
            cond_error_rate = fails / total

        results.append({
            "physical_error_rate": per,
            "conditional_error_rate": cond_error_rate
        })

        print()
        print(f"Physical Error Rate: {per}")
        print(f"Conditional Error Rate: {cond_error_rate:.8f}")
        print()

    return results

# Plot decoding time as per increases later (distance is a part of scalability)
def latency(distance = dist):
    results = []
    for i in range(1,21): # 0.0005 - 0.01
        per = 5*i/10000
        code = surface_code(distance, synd_rounds, per)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        latencies = []
        
        for i in range(len(detections)):
            start = time.perf_counter()
            dc.decode(detections[i])
            end = time.perf_counter()
            latencies.append(end-start)
        
        avg_latency = np.mean(latencies)

        results.append({
            "physical_error_rate": per,
            "average_decoding_latency": avg_latency
        })

        print()
        print(f"Distance: {distance}, Physical Error Rate: {per}")
        print(f"Average Decoding Latency: {avg_latency:.10f}")
        print()
    
    return results

# Plot logical error rate as distnace increases
def threshold():
    results = {}
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = []
        for i in range(1,41): # 0.0005 - 0.02
            per = 5*i/10000
            code = surface_code(dist, synd_rounds, per)
            dc = decoder(code)
            detections, observed_flips = run_simulations(code, shots)

            predictions = dc.decode_batch(detections)

            fails = 0
            for i in range(shots):
                if not np.array_equal(predictions[i], observed_flips[i]):
                    fails += 1

            log_error_rate = fails / shots

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



def mwpm_test(test_num):
    results = []
    output_file = ""
    if test_num == 1:
        results = correctness()
        output_file = "results/mwpm/results_correctness.json"
    if test_num == 2:
        results = latency()
        output_file = "results/mwpm/results_latency.json"
    if test_num == 3:
        results = threshold()
        output_file = "results/mwpm/results_threshold.json"
    if test_num == 4:
        results = robustness()
        output_file = "results/mwpm/results_robustness.json"
    if test_num == 5:
        results = scalability()
        output_file = "results/mwpm/results_scalability.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    

if __name__ == '__main__':
    if len(sys.argv) > 2 or int(sys.argv[1]) not in range(1,6):
        print("Specify Test Type (1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
    else:
        test_type = int(sys.argv[1])
        mwpm_test(test_type)