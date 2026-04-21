import numpy as np
import stim
import json
import time
import sys

from ldpc.union_find_decoder import UnionFindDecoder as _UFDecoder


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 50000


class UnionFindDecoder:
    def __init__(self, circuit: stim.Circuit, logical: int = 0):
        dem = circuit.detector_error_model(decompose_errors=True)
        self.num_det = dem.num_detectors
        self.num_obs = dem.num_observables
        self.logical = logical

        error_list = [
            inst for inst in dem if inst.type == "error"
        ]
        num_errors = len(error_list)

        H        = np.zeros((self.num_det, num_errors), dtype=np.uint8)
        logicals = np.zeros(num_errors, dtype=np.uint8)

        for e, inst in enumerate(error_list):
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    H[t.val, e] = 1
                elif t.is_logical_observable_id() and t.val == logical:
                    logicals[e] = 1

        self._decoder  = _UFDecoder(H, uf_method="global")
        self._logicals = logicals

    def decode(self, dets_row: np.ndarray) -> np.ndarray:
        dets_row   = np.asarray(dets_row, dtype=np.uint8)
        correction = self._decoder.decode(dets_row)
        pred       = np.zeros(self.num_obs, dtype=np.uint8)
        pred[self.logical] = int(np.dot(correction, self._logicals) % 2)
        return pred

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections, dtype=np.uint8)
        out = np.zeros((detections.shape[0], self.num_obs), dtype=np.uint8)
        for s in range(detections.shape[0]):
            out[s] = self.decode(detections[s])
        return out



# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------

def surface_code(distance, rounds, phys_error_rate, depolarization = 0, measure = 0, reset = 0):
    sc = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=phys_error_rate*depolarization,
        before_round_data_depolarization=phys_error_rate,
        before_measure_flip_probability=phys_error_rate*measure,
        after_reset_flip_probability=phys_error_rate*reset
    )
    return sc

def decoder(surface_code):
    return UnionFindDecoder(surface_code)

def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=True)



# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

def correctness(depolarization = 0, measure = 0, reset = 0):
    results = []
    for i in range(50,201,5): # 0.0005 - 0.01
        per = 5*i/10000
        code = surface_code(dist, dist//2, per, depolarization, measure, reset)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        predictions = dc.decode_batch(detections)
        detections = np.asarray(detections)
        predictions = np.asarray(predictions)
        observed_flips = np.asarray(observed_flips)

        # shots where syndrome is nontrivial
        mask = np.any(observed_flips != 0, axis=1)

        if np.sum(mask) == 0:
            return 0  # no errors occurred

        correct = np.all(predictions[mask] != observed_flips[mask], axis=1)

        cond_error_rate = np.mean(correct)

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
    results = []
    for i in range(50,201,5): # 0.0005 - 0.01
        per = 5*i/10000
        code = surface_code(dist, dist//2, per)
        dc = decoder(code)
        detections, observed_flips = run_simulations(code, shots)

        start = time.perf_counter()
        dc.decode_batch(detections)
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
    results = {}
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = []
        for r in range(50,201,5): # 0.025 - 0.1
            per = 5*r/10000
            code = surface_code(dist, dist//2, per)
            dc = decoder(code)
            detections, observed_flips = run_simulations(code, shots)

            predictions = dc.decode_batch(detections)

            fails = np.count_nonzero(
                np.any(predictions != observed_flips, axis=1)
            )

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




def union_find_test(test_num):
    results = []
    output_file = ""
    if test_num == 1:
        results = correctness()
        output_file = "results/union/results_correctness.json"
    if test_num == 2:
        results = latency()
        output_file = "results/union/results_latency.json"
    if test_num == 3:
        results = threshold()
        output_file = "results/union/results_threshold.json"
    if test_num == 4:
        results = robustness()
        output_file = "results/union/results_robustness.json"
    if test_num == 5:
        results = scalability()
        output_file = "results/union/results_scalability.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    

if __name__ == '__main__':
    if len(sys.argv) > 2 or int(sys.argv[1]) not in range(1,6):
        print("Specify Test Type (1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
    else:
        test_type = int(sys.argv[1])
        union_find_test(test_type)