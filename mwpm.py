import stim
import pymatching
import numpy as np
import time


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 1000000


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

# Samples possible error outputs and decodes them
def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=True)


# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

# Plot decoder's conditional correctness as per increases
# Conditional correctness: prediction correctness in cases with errors
def correctness(depolarization = 0, measure = 0, reset = 0):
    for i in range(1,11): # 0.0005 - 0.005
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

        print()
        print(f"Physical Error Rate: {per}")
        print(f"Conditional Error Rate: {cond_error_rate:.8f}")
        print()

    return None

# Plot decoding time as per increases later (distance is a part of scalability)
def latency():
    for i in range(1,11): # 0.0005 - 0.005
        per = 5*i/10000
        code = surface_code(dist, synd_rounds, per)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        start = time.perf_counter()
        dc.decode_batch(detections)
        end = time.perf_counter()
        
        avg_latency = (end - start)/shots

        print()
        print(f"Distance: {dist}, Physical Error Rate: {per}")
        print(f"Average Decoding Latency: {avg_latency:.10f}")
        print()

    return None

# Plot logical error rate as distnace increases
def threshold():
    for dist in range(3,10,2):
        for i in range(1,21): # 0.0005 - 0.01
            per = 5*i/10000
            code = surface_code(dist, synd_rounds, per)
            dc = decoder(code)
            detections, observed_flips = run_simulations(code, shots)

            start = time.perf_counter()
            predictions = dc.decode_batch(detections)
            end = time.perf_counter()

            #fails = np.sum(np,any(predictions[:,0] != observed_flips))
            fails = 0
            for i in range(shots):
                if not np.array_equal(predictions[i], observed_flips[i]):
                    fails += 1

            log_error_rate = fails / shots
            avg_latency = (end - start)/shots

            print()
            print(f"Distance: {dist}, Physical Error Rate: {per}")
            print(f"Logical Error Rate: {log_error_rate:.8f}")
            print(f"Average Decoding Latency: {avg_latency:.10f}")
            print()
    
    return None

# Include different error models and test correctness
def robustness():
    print("\n--------- Control ---------\n")
    control_plot = correctness()
    print("\n--------- Depolarization ---------\n")
    depolarization_plot = correctness(depolarization=1)
    print("\n--------- Measure ---------\n")
    measure_plot = correctness(measure=1)
    print("\n--------- Reset ---------\n")
    reset_plot = correctness(reset=1)
    return None

# Track qubit counts and decoding latency -> space time measurements
def scalability():
    return None



def mwpm_test(test_num):
    output_plot = None

    if test_num == 1:
        output_plot = correctness()
    if test_num == 2:
        output_plot = latency()
    if test_num == 3:
        output_plot = threshold()
    if test_num == 4:
        output_plot = robustness()
    if test_num == 5:
        output_plot = scalability()
    
    # Display plot?