import sys
import numpy as np
import time
import json
import stim
from collections import defaultdict, deque


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 10


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

            targets = inst.targets_copy()
            dets = [t.val for t in targets if t.is_relative_detector_id()]

            if len(dets) == 2:
                edges.append((dets[0], dets[1]))
            elif len(dets) == 1:
                edges.append((dets[0], None))  # boundary

        return edges

    def _build_adj_list(self, edges):
        adj = defaultdict(list)
        for u, v in edges:
            if v is not None:
                adj[u].append(v)
                adj[v].append(u)
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
        logicals = set()

        edge_set = {tuple(sorted(e)) for e in correction_edges}

        for dets, obs in self.observable_map:
            if len(dets) == 2:
                edge = tuple(sorted(dets))
                if edge in edge_set:
                    for o in obs:
                        logicals.add(o)

        return logicals

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

    def decode(self, syndrome):
        syndrome = np.array(syndrome, dtype=np.uint8)

        # --- initialize UF ---
        self.parent = np.arange(self.num_detectors)
        self.size = np.ones(self.num_detectors)
        self.parity = syndrome.copy()
        self.boundary = np.zeros(self.num_detectors, dtype=bool)

        # boundary nodes
        for u, v in self.edges:
            if v is None:
                self.boundary[u] = True

        # -------------------------
        # TRUE UF GROWTH
        # -------------------------
        frontier = {i: set() for i in range(self.num_detectors)}
        active_roots = set()

        # initialize frontier
        for i in np.where(syndrome == 1)[0]:
            r = self._find(i)
            active_roots.add(r)
            for nb in self.adj[i]:
                frontier[r].add(tuple(sorted((i, nb))))

        grown_edges = set()

        max_iters = 10000
        iters = 0

        while active_roots:
            iters += 1
            if iters > max_iters:
                raise RuntimeError("UF growth did not terminate")

            # --- compute stats ---
            roots = {self._find(r) for r in active_roots}
            cluster_sizes = {r: 0 for r in roots}

            for i in range(self.num_detectors):
                ri = self._find(i)
                if ri in cluster_sizes:
                    cluster_sizes[ri] += 1

            if self.debug and iters % 10 == 0:
                self._log(f"\n[Growth Iter {iters}]")
                self._log(f"Active clusters: {len(roots)}")
                self._log(f"Cluster sizes: {list(cluster_sizes.values())[:10]}...")
                self._log(f"Frontier edges: {sum(len(frontier.get(r, [])) for r in roots)}")

            # --- stopping condition ---
            def is_active(root):
                root = self._find(root)
                return self.parity[root] == 1 and not self.boundary[root]

            if not any(is_active(r) for r in active_roots):
                self._log("All clusters resolved (even or boundary) → stopping growth")
                break
            
            new_frontier = {}
            merges_this_round = 0
            edges_processed = 0

            for r in list(active_roots):
                r = self._find(r)

                if self.parity[r] == 0:
                    continue
                
                for edge in frontier.get(r, []):
                    if edge in grown_edges:
                        continue
                    
                    grown_edges.add(edge)
                    edges_processed += 1

                    u, v = edge
                    ru, rv = self._find(u), self._find(v)

                    if ru != rv:
                        new_root = self._union(ru, rv)
                        merges_this_round += 1

                        frontier[new_root] = (
                            frontier.get(ru, set()) |
                            frontier.get(rv, set())
                        )

                        active_roots.discard(ru)
                        active_roots.discard(rv)

                        if self.parity[new_root] == 1:
                            active_roots.add(new_root)

                    # controlled expansion
                    for node in (u, v):
                        rnode = self._find(node)
                        for nb in self.adj[node]:
                            new_edge = tuple(sorted((node, nb)))
                            if new_edge not in grown_edges:
                                new_frontier.setdefault(rnode, set()).add(new_edge)

            # --- log per-iteration activity ---
            if self.debug:
                self._log(f"Edges processed: {edges_processed}")
                self._log(f"Merges this round: {merges_this_round}")

            # update frontier
            frontier = new_frontier

            # filter active roots
            active_roots = {
                self._find(r)
                for r in active_roots
                if self.parity[self._find(r)] == 1 and not self.boundary[self._find(r)]
            }
            
        # -------------------------
        # BUILD CLUSTERS
        # -------------------------
        clusters = defaultdict(list)
        for i in np.where(syndrome == 1)[0]:
            root = self._find(i)
            clusters[root].append(i)

        # -------------------------
        # PEELING
        # -------------------------
        return self._peeling(clusters)

    # -------------------------
    # Peeling (correct)
    # -------------------------
    def _peeling(self, clusters):
        correction = []

        for root, nodes in clusters.items():
            # build spanning tree
            tree_parent = {}
            tree_children = defaultdict(list)
            visited = set()

            start = nodes[0]
            queue = deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        tree_parent[v] = u
                        tree_children[u].append(v)
                        queue.append(v)

            # parity init
            parity = {n: 1 for n in nodes}
            for n in visited:
                parity.setdefault(n, 0)

            # degree
            degree = {}
            for n in visited:
                deg = len(tree_children[n])
                if n in tree_parent:
                    deg += 1
                degree[n] = deg

            # leaf queue
            leaf_queue = deque([n for n in visited if degree[n] == 1])

            # iterative peeling
            while leaf_queue:
                leaf = leaf_queue.popleft()

                if degree[leaf] == 0:
                    continue

                parent = tree_parent.get(leaf)

                if parity[leaf] == 1 and parent is not None:
                    correction.append((leaf, parent))
                    parity[parent] ^= 1

                degree[leaf] = 0

                if parent is not None:
                    degree[parent] -= 1

                    if leaf in tree_children[parent]:
                        tree_children[parent].remove(leaf)

                    if degree[parent] == 1:
                        leaf_queue.append(parent)

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