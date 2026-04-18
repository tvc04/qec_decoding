import numpy as np
import stim
import json
from collections import defaultdict, deque
import pymatching

# ======================================================
#                UNION-FIND DECODER
# ======================================================

class UnionFindDecoder:
    def __init__(self, dem):
        self.dem = dem
        self.num_detectors = dem.num_detectors

        self.edges = self._extract_edges(dem)
        self.adj = self._build_adj(self.edges)

    def _extract_edges(self, dem):
        edges = []
        for inst in dem:
            if inst.type != "error":
                continue

            dets = [t.val for t in inst.targets_copy()
                    if t.is_relative_detector_id()]

            if len(dets) == 2:
                edges.append((dets[0], dets[1]))

        return edges

    def _build_adj(self, edges):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # --------------------------------------------------
    # Union-Find core
    # --------------------------------------------------

    def _find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return ra

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    def decode(self, syndrome):
        syndrome = np.array(syndrome, dtype=np.uint8)

        self.parent = np.arange(self.num_detectors)
        self.size = np.ones(self.num_detectors)

        frontier = defaultdict(set)
        grown = set()

        # ----------------------------
        # INITIAL ACTIVE NODES
        # ----------------------------
        active = set()
        for i in np.where(syndrome == 1)[0]:
            r = self._find(i)
            active.add(r)

            for nb in self.adj[i]:
                frontier[r].add(tuple(sorted((i, nb))))

        # ============================
        # GROW UNTIL FIXED POINT
        # ============================
        changed = True

        while changed:
            changed = False
            new_frontier = defaultdict(set)

            for r in list(active):
                r = self._find(r)

                for edge in frontier[r]:
                    if edge in grown:
                        continue

                    grown.add(edge)
                    u, v = edge

                    ru, rv = self._find(u), self._find(v)

                    if ru != rv:
                        new_r = self._union(ru, rv)
                        changed = True

                        active.discard(ru)
                        active.discard(rv)
                        active.add(new_r)

                        new_frontier[new_r] |= frontier[ru]
                        new_frontier[new_r] |= frontier[rv]

            frontier = new_frontier

        # ============================
        # FINAL CLUSTERS ONLY
        # ============================
        clusters = defaultdict(list)
        for i in np.where(syndrome == 1)[0]:
            clusters[self._find(i)].append(i)

        return clusters


# ======================================================
#         LOGICAL EVALUATION (CORRECT FORMULATION)
# ======================================================

def logical_failure_from_clusters(clusters):
    """
    Correct UF-style logical failure proxy:
    If any cluster remains "odd", it implies unresolved logical structure.
    """

    failures = 0

    for nodes in clusters.values():
        if len(nodes) % 2 == 1:
            failures += 1

    return failures > 0



# ======================================================
#                SURFACE CODE GENERATION
# ======================================================

def surface_code(distance, rounds, phys_error_rate,
                 depolarization=0, measure=0, reset=0):

    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=phys_error_rate,
        before_round_data_depolarization=phys_error_rate * depolarization,
        before_measure_flip_probability=phys_error_rate * measure,
        after_reset_flip_probability=phys_error_rate * reset
    )


def run_simulations(circuit, shots):
    sampler = circuit.compile_detector_sampler()
    return sampler.sample(shots, separate_observables=True)


def decoder(circuit):
    dem = circuit.detector_error_model(decompose_errors=True)
    return UnionFindDecoder(dem)


# ======================================================
#                THRESHOLD EXPERIMENT
# ======================================================

def threshold():

    shots = 1000
    synd_rounds = 5

    for dist in range(3, 10, 2):

        for i in range(1, 21):
            per = 5 * i / 10000

            circuit = surface_code(dist, synd_rounds, per)
            dc = decoder(circuit)

            detections, observed_flips = run_simulations(circuit, shots)

            uf_fails = 0
            pm_fails = 0

            for shot in detections:
                clusters = dc.decode(shot)

                if logical_failure_from_clusters(clusters):
                    uf_fails += 1

            error_model = circuit.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(error_model)

            predictions = matching.decode_batch(detections)

            for i in range(shots):
                if not np.array_equal(predictions[i], observed_flips[i]):
                    pm_fails += 1

            pm_ler = pm_fails / shots

            uf_ler = uf_fails / shots

            print("\nUNION FIND")
            print(f"Distance {dist}, p={per:.5f}, LER={uf_ler:.5f}")
            print("\nMWPM")
            print(f"Distance {dist}, p={per:.5f}, LER={uf_ler:.5f}")




class UFD:
    def __init__(self, dem):
        self.num_detectors = dem.num_detectors

        self.adj = defaultdict(list)

        for inst in dem:
            if inst.type != "error":
                continue

            dets = [t.val for t in inst.targets_copy()
                    if t.is_relative_detector_id()]

            if len(dets) == 2:
                u, v = dets
                self.adj[u].append(v)
                self.adj[v].append(u)

    # -------------------------
    def _find(self, parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, parent, size, a, b):
        ra = self._find(parent, a)
        rb = self._find(parent, b)

        if ra == rb:
            return ra, False

        if size[ra] < size[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        size[ra] += size[rb]
        return ra, True

    # -------------------------
    def decode(self, syndrome):
        syndrome = np.array(syndrome, dtype=np.uint8)

        parent = np.arange(self.num_detectors)
        size = np.ones(self.num_detectors)

        active = np.where(syndrome == 1)[0]

        correction_edges = []

        # union-find growth
        for i in active:
            for j in self.adj[i]:
                _, merged = self._union(parent, size, i, j)
                if merged:
                    correction_edges.append((i, j))

        # build final clusters
        clusters = defaultdict(list)
        for i in active:
            clusters[self._find(parent, i)].append(i)

        return clusters, correction_edges


def logical_error_occurred(clusters):
    """
    UF proxy for logical failure without boundary assumptions:
    failure occurs if any cluster has odd syndrome parity.
    """

    for nodes in clusters.values():
        if len(nodes) % 2 == 1:
            return True

    return False

def run_threshold_experiment():

    results = {}
    shots = 100000
    rounds = 5

    for distance in range(3, 10, 2):
        results[f'{distance}'] = []
        dist_results = []

        for k in range(1, 41):
            p = 5 * k / 10000000

            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=p,
            )

            sampler = circuit.compile_detector_sampler()
            detections = sampler.sample(shots)

            dem = circuit.detector_error_model(decompose_errors=True)
            decoder = UFD(dem)

            failures = 0

            # ==================================================
            # CORRECT DECODING EVALUATION
            # ==================================================
            for shot in detections:
                clusters, corr = decoder.decode(shot)

                if logical_error_occurred(clusters):
                    failures += 1

            ler = failures / shots

            dist_results.append({
                "physical_error_rate": p,
                "logical_error_rate": ler
            })

            print(f"distance={distance}, p={p:.5f}, LER={ler:.5f}")

        results[f'{distance}'].extend(dist_results)

    output_file = "results/union/results_threshold.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

run_threshold_experiment()