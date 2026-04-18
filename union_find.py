import sys
import numpy as np
import time
import json
import math
import stim
from collections import defaultdict, deque


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 1000


# ----------------------------------------
#       CUSTOM UNION-FIND DECODER
# ----------------------------------------

class UnionFindDecoder:
    def __init__(self, dem, debug=False):
        self.dem = dem
        self.num_detectors = dem.num_detectors
        self.debug = debug

        # Graph
        self.edges = self._extract_edges(dem)
        self.adj = self._build_adj_list(self.edges)
        self.observable_map = self._extract_observables(dem)

    # -------------------------
    # DEM parsing
    # -------------------------
    def _extract_edges(self, dem):
        edges = []

        for inst in dem:
            if inst.type != "error":
                continue

            p = inst.args_copy()[0]  # error probability

            targets = inst.targets_copy()
            dets = [t.val for t in targets if t.is_relative_detector_id()]

            if len(dets) == 2:
                edges.append((dets[0], dets[1], p))
            elif len(dets) == 1:
                edges.append((dets[0], None, p))

        return edges

    def _build_adj_list(self, edges):
        adj = defaultdict(list)

        for u, v, p in edges:
            w = -math.log(p) if p > 0 else 100  # avoid log(0)

            if v is not None:
                adj[u].append((v, w))
                adj[v].append((u, w))

        return adj

    def _extract_observables(self, dem):
        obs_map = []

        for inst in dem:
            if inst.type != "error":
                continue

            targets = inst.targets_copy()

            dets = [t.val for t in targets if t.is_relative_detector_id()]
            obs = [t.val for t in targets if t.is_logical_observable_id()]

            if len(obs) > 0:
                obs_map.append((tuple(dets), tuple(obs)))

        return obs_map

    def predict_logicals(self, correction_edges):
        # number of logical observables
        num_obs = 0
        for _, obs in self.observable_map:
            for o in obs:
                num_obs = max(num_obs, o + 1)

        pred = np.zeros(num_obs, dtype=np.uint8)

        # normalize correction edges
        correction_set = {tuple(sorted(e)) for e in correction_edges}

        # ----------------------------------------------------
        # CORRECT INTERPRETATION:
        # use DEM mapping: (detectors → observables)
        # ----------------------------------------------------
        for dets, obs in self.observable_map:

            # Stim DEM entries may have 1 or more detectors
            affected = False

            if len(dets) == 0:
                continue

            elif len(dets) == 1:
                # single-detector events still matter
                affected = True

            else:
                edge = tuple(sorted(dets))
                if edge in correction_set:
                    affected = True

            if affected:
                for o in obs:
                    pred[o] ^= 1  # parity accumulation

        return pred

    # -------------------------
    # Union-Find
    # -------------------------
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

        self.parity[ra] ^= self.parity[rb]
        self.boundary[ra] |= self.boundary[rb]

        return ra

    # -------------------------
    # Decode
    # -------------------------
    def _log(self, msg):
        if self.debug:
            print(msg)
    
    def _edge_weight(self, edge):
        u, v = edge
        for nb, w in self.adj[u]:
            if nb == v:
                return w
        return 0

    def decode(self, syndrome):
        syndrome = np.array(syndrome, dtype=np.uint8)

        # --- initialize UF ---
        self.parent = np.arange(self.num_detectors)
        self.size = np.ones(self.num_detectors)
        self.parity = syndrome.copy()
        self.boundary = np.zeros(self.num_detectors, dtype=bool)

        # boundary nodes
        for u, v, _ in self.edges:
            if v is None:
                self.boundary[u] = True

        # ------------------------------------
        # UF CLUSTER DATA STRUCTURES
        # ------------------------------------
        frontier = {}
        cluster_edges = defaultdict(set)   # ✅ IMPORTANT: UF-grown edges per cluster
        active_roots = set()
        grown_edges = set()

        # initialize clusters
        for i in np.where(syndrome == 1)[0]:
            r = self._find(i)
            active_roots.add(r)

            frontier[r] = set()
            for nb, _ in self.adj[i]:
                e = tuple(sorted((i, nb)))
                frontier[r].add(e)

        max_iters = 10000
        iters = 0

        # ------------------------------------
        # UF GROWTH (LAYERED)
        # ------------------------------------
        while active_roots:
            iters += 1
            if iters > max_iters:
                raise RuntimeError("UF growth did not terminate")

            def is_active(r):
                r = self._find(r)
                return self.parity[r] == 1 and not self.boundary[r]

            if not any(is_active(r) for r in active_roots):
                break

            current_frontier = frontier
            next_frontier = {}
            merges_this_round = 0

            for r in list(active_roots):
                r = self._find(r)

                if self.parity[r] == 0:
                    continue

                for edge in current_frontier.get(r, set()):
                    if edge in grown_edges:
                        continue

                    grown_edges.add(edge)

                    u, v = edge
                    ru, rv = self._find(u), self._find(v)

                    # ------------------------------------
                    # UNION
                    # ------------------------------------
                    if ru != rv:
                        new_root = self._union(ru, rv)
                        merges_this_round += 1

                        # ------------------------------------
                        # MERGE CLUSTER EDGES (CRITICAL FIX)
                        # ------------------------------------
                        cluster_edges[new_root] |= cluster_edges[ru]
                        cluster_edges[new_root] |= cluster_edges[rv]
                        cluster_edges[new_root].add(edge)

                        # merge frontiers
                        merged = current_frontier.get(ru, set()) | current_frontier.get(rv, set())
                        next_frontier[new_root] = merged

                        active_roots.discard(ru)
                        active_roots.discard(rv)

                        if self.parity[new_root] == 1:
                            active_roots.add(new_root)

                    # ------------------------------------
                    # EXPAND (NEXT LAYER ONLY)
                    # ------------------------------------
                    for node in (u, v):
                        rnode = self._find(node)
                        for nb, _ in self.adj[node]:
                            new_edge = tuple(sorted((node, nb)))
                            if new_edge not in grown_edges:
                                next_frontier.setdefault(rnode, set()).add(new_edge)

                                # IMPORTANT: track UF-grown structure
                                cluster_edges[rnode].add(new_edge)

            frontier = next_frontier

            active_roots = {
                self._find(r)
                for r in active_roots
                if self.parity[self._find(r)] == 1 and not self.boundary[self._find(r)]
            }

        # ------------------------------------
        # BUILD CLUSTERS
        # ------------------------------------
        clusters = defaultdict(list)
        for i in np.where(syndrome == 1)[0]:
            root = self._find(i)
            clusters[root].append(i)

        # attach cluster edges for peeling
        self.cluster_edges = cluster_edges

        # ------------------------------------
        # PEELING
        # ------------------------------------
        return self._peeling(clusters)

    # -------------------------
    # Peeling (correct)
    # -------------------------
    def _peeling(self, clusters):
        correction = []

        for root, nodes in clusters.items():
            nodes_set = set(nodes)

            # ----------------------------------------
            # BUILD FOREST FROM UF GROWN EDGES ONLY
            # ----------------------------------------
            edges = self.cluster_edges[root] if root in self.cluster_edges else set()

            tree = defaultdict(list)
            degree = defaultdict(int)

            # build tree from UF-accepted edges only
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)

            for n in nodes_set:
                degree[n] = len(tree[n])

            # ----------------------------------------
            # OPTIONAL: boundary handling
            # ----------------------------------------
            touches_boundary = any(self.boundary[n] for n in nodes_set)

            BOUNDARY = -1
            if touches_boundary:
                # attach one arbitrary node to boundary
                start = next(iter(nodes_set))
                tree[start].append(BOUNDARY)
                tree[BOUNDARY].append(start)
                degree[BOUNDARY] = 1
                nodes_set.add(BOUNDARY)

            # ----------------------------------------
            # INITIALIZE PARITY
            # ----------------------------------------
            parity = {n: 0 for n in nodes_set}
            for n in nodes_set:
                if n != BOUNDARY:
                    parity[n] = 1  # syndrome nodes are odd initially

            # ----------------------------------------
            # INITIAL LEAVES
            # ----------------------------------------
            leaf_queue = deque([n for n in nodes_set if len(tree[n]) == 1])

            # ----------------------------------------
            # PEELING PROCESS
            # ----------------------------------------
            while leaf_queue:
                leaf = leaf_queue.popleft()

                if len(tree[leaf]) == 0:
                    continue

                for parent in tree[leaf]:
                    if parity[leaf] == 1 and leaf != BOUNDARY and parent != BOUNDARY:
                        correction.append((leaf, parent))
                        parity[parent] ^= 1

                    # remove edge
                    tree[parent].remove(leaf)
                    tree[leaf].remove(parent)

                    if len(tree[parent]) == 1:
                        leaf_queue.append(parent)

                tree[leaf] = []

        return correction


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
    return UnionFindDecoder(error_model, False)

# Generates sample error syndrome
def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=False)


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

        predictions = []
        for shot in detections:
            result = dc.decode(shot)
            predictions.append(dc.predict_logicals(result))
        
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
    results = {}
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = []
        for i in range(1,21): # 0.005 - 0.1
            per = 5*i/1000
            code = surface_code(dist, synd_rounds, per)
            dc = decoder(code)
            detections = run_simulations(code, shots)
    
            fails = 0

            for shot in detections:
                correction = dc.decode(shot)
                predicted_logical = dc.predict_logicals(correction)

                # -----------------------------------
                # CORRECT FAILURE DEFINITION:
                # logical error = ANY nontrivial logical flip
                # (vector != 0)
                # -----------------------------------
                if np.sum(predicted_logical) % 2 == 1:
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

def robustness():
    pass

def scalability():
    pass

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