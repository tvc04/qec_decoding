import sys
import numpy as np
import time
import json
import stim
import pymatching


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 100000


# ----------------------------------------
#       CUSTOM UNION-FIND DECODER
# ----------------------------------------

class UnionFindDecoder:
    def __init__(self, dem):
        """
        dem: stim.DetectorErrorModel
        """
        self.dem = dem
        self.num_detectors = dem.num_detectors

        # Build graph from DEM
        self.edges = self._extract_edges(dem)

    def _extract_edges(self, dem):
        """
        Extract edges: (detector1, detector2) or boundary edges
        """
        edges = []
        for inst in dem:
            if inst.type == "error":
                dets = [t.val for t in inst.targets if t.is_relative_detector_id()]
                
                if len(dets) == 2:
                    edges.append((dets[0], dets[1]))
                elif len(dets) == 1:
                    # boundary edge
                    edges.append((dets[0], None))
        return edges

    def decode(self, syndrome):
        """
        syndrome: binary array of detector outcomes
        """
        syndrome = np.array(syndrome, dtype=np.uint8)

        # Initialize clusters
        parent = np.arange(self.num_detectors)
        size = np.ones(self.num_detectors)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        # Step 1: initialize clusters on syndrome nodes
        active = np.where(syndrome == 1)[0]

        # Step 2: grow clusters (VERY simplified version)
        for u, v in self.edges:
            if v is None:
                continue
            if syndrome[u] and syndrome[v]:
                union(u, v)

        # Step 3: identify clusters (placeholder)
        clusters = {}
        for i in active:
            root = find(i)
            clusters.setdefault(root, []).append(i)

        # Step 4: return correction (stub for now)
        return self._clusters_to_correction(clusters)

    def _clusters_to_correction(self, clusters):
        """
        Convert clusters into corrections.
        (placeholder – will implement peeling next)
        """
        return clusters


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

def latency():
    pass

def threshold():
    pass

def robustness():
    pass

def scalability():
    pass

def union_find_test(test_num):
    if test_num == 1:
        correctness()
    if test_num == 2:
        latency()
    if test_num == 3:
        threshold()
    if test_num == 4:
        robustness()
    if test_num == 5:
        scalability()


if __name__ == '__main__':
    if len(sys.argv) > 2 or int(sys.argv[1]) not in range(1,6):
        print("Specify Test Type (1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
    else:
        test_type = int(sys.argv[1])
        union_find_test(test_type)