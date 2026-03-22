import stim
import pymatching
import numpy as np
import time


# PARAMETERS

dist = 3
per = 0.001
synd_rounds = dist
shots = 100000


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

def decoder(surface_code):
    error_model = surface_code.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(error_model)
    return matching

def run_simulations(surface_code, decoder, shots):
    samples = surface_code.compile_detector_sampler()
    detections, observed_flips = samples.sample(shots=shots, separate_observables=True)

    latencies = []
    fails = 0

    for i in range(shots):
        syndrome = detections[i]

        start = time.perf_counter()
        prediction = decoder.decode(syndrome)
        end = time.perf_counter()

        latencies.append(end-start)
        if not np.array_equal(prediction, observed_flips[i]):
            fails += 1
    
    log_error_rate = fails / shots

    return {
        "logical_error_rate": log_error_rate,
        "avg_latency": np.mean(latencies),
        "failures": fails
    }


def main():
    print("Constructing Surface Code...")
    code = surface_code(dist, synd_rounds, per)

    print("Constructing MWPM Decoder...")
    dc = decoder(code)

    print("Running Simulations...")
    results = run_simulations(code, dc, shots)

    print("\nResults:\n")
    print(f"Distance: {dist}, Physical Error Rate: {per}")
    print(f"Logical Error Rate: {results['logical_error_rate']}")
    print(f"Average Decoding Latency: {results['avg_latency']}")
    print(f"Number of Failures: {results['failures']}")


if __name__ == "__main__":
    main()