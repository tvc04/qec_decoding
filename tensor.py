import cudaq
import cudaq_qec as qec
import numpy as np
import time
import json
import stim
from beliefmatching.belief_matching import detector_error_model_to_check_matrices


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 1
shots = 1000000


# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------

# Create surface code circuit
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

# Makes error model usable for cudaq decoder
def parse_detector_error_model(detector_error_model):
    matrices = detector_error_model_to_check_matrices(detector_error_model)

    out_H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=out_H)
    out_L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=out_L)

    return out_H, out_L, [float(p) for p in matrices.priors]

# Creates cudaq tensor network decoder
def decoder(surface_code):
    detector_error_model = surface_code.detector_error_model(decompose_errors=True)

    H, logicals, noise_model = parse_detector_error_model(detector_error_model)

    decoder = qec.get_decoder(
        "tensor_network_decoder",
        H,
        logical_obs=logicals,
        noise_model=noise_model,
        contract_noise_model=True,
    )

    return decoder

# Generates sample error syndrome measurments
def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots, separate_observables=True)


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

        res = dc.decode_batch(detections)

        preds = [r.result[0] > 0.5 for r in res]
        actuals = [bool(o[0]) for o in observed_flips]

        fails = 0
        total = 0
        for p, a in zip(preds, actuals):
            if a:  # condition on actual logical flip
                total += 1
                if p != a:
                    fails += 1

        cond_error_rate = fails / total if total > 0 else 0

        results.append({
            "physical_error_rate": per,
            "conditional_error_rate": cond_error_rate
        })

        print()
        print(f"Distance: {dist}, Physical Error Rate: {per}")
        print(f"Conditional error rate: {cond_error_rate:.8f}")
        print()

    return results

# Plot decoding time as per increases later (distance is a part of scalability)
def latency(distance = dist):
    results = []
    shots = 1000
    for i in range(1,21): # 0.0005 - 0.01
        per = 5*i/10000
        code = surface_code(distance, synd_rounds, per)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        start = time.perf_counter()

        predictions = []
        for i in range(shots):
            det = detections[i].astype(float)
            pred = dc.decode(det)
            predictions.append(pred)

        end = time.perf_counter()

        avg_latency = (end - start)/shots

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
    shots = 100000
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = []
        for i in range(30,61): # 0.015 - 0.03
            per = 5*i/10000
            code = surface_code(dist, synd_rounds, per)
            dc = decoder(code)
            detections, observed_flips = run_simulations(code, shots)

            res = dc.decode_batch(detections)

            preds = [r.result[0] > 0.5 for r in res]
            actuals = [bool(o[0]) for o in observed_flips]

            fails = sum(p != a for p, a in zip(preds, actuals))
            logical_error_rate = fails / len(preds)

            dist_results.append({
                "physical_error_rate": per,
                "logical_error_rate": logical_error_rate
            })

            print()
            print(f"Distance: {dist}, Physical Error Rate: {per}")
            print(f"Logical error rate: {logical_error_rate:.8f}")
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



def tensor_test(test_num):
    results = []
    output_file = ""
    if test_num == 1:
        results = correctness()
        output_file = "results/tensor/results_correctness.json"
    if test_num == 2:
        results = latency()
        output_file = "results/tensor/results_latency.json"
    if test_num == 3:
        results = threshold()
        output_file = "results/tensor/results_threshold.json"
    if test_num == 4:
        results = robustness()
        output_file = "results/tensor/results_robustness.json"
    if test_num == 5:
        results = scalability()
        output_file = "results/tensor/results_scalability.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    

if __name__ == '__main__':
    tensor_test(3)