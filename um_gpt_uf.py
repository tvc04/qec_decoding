from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set
import heapq
import math

import numpy as np
import stim
import pymatching

import json


@dataclass(frozen=True)
class _AdjEdge:
    nb: int
    is_logical: int  # 0/1
 
 
class UnionFindDecoder:

    def __init__(self, circuit: stim.Circuit, logical: int = 0):
        self.circuit  = circuit
        self.dem      = circuit.detector_error_model(decompose_errors=True)
        self.num_det  = self.dem.num_detectors
        self.num_obs  = self.dem.num_observables
        self.logical  = logical
 
        if self.num_obs <= logical:
            raise ValueError(
                f"num_observables={self.num_obs}, logical={logical} out of range"
            )
 
        # Node layout: detectors 0..num_det-1, boundary = num_det
        self.boundary   = self.num_det
        self.num_nodes  = self.num_det + 1
        adj: List[List[_AdjEdge]] = [[] for _ in range(self.num_nodes)]
        seen_edges = set()  # frozenset({a, b}) already added
 
        def add_undirected(a: int, b: int, is_logical: int) -> None:
            key = frozenset((a, b))
            if key in seen_edges:
                return
            seen_edges.add(key)
            adj[a].append(_AdjEdge(b, is_logical))
            adj[b].append(_AdjEdge(a, is_logical))
 
        for inst in self.dem:
            if inst.type != "error":
                continue

            targets       = inst.targets_copy()
            det_ids       = [t.val for t in targets if t.is_relative_detector_id()]
            obs_ids       = [t.val for t in targets if t.is_logical_observable_id()]
            flips_logical = int(logical in obs_ids)

            if len(det_ids) == 0:
                pass  # pure logical flip, undetectable
            elif len(det_ids) == 1:
                add_undirected(det_ids[0], self.boundary, flips_logical)
            elif len(det_ids) == 2:
                add_undirected(det_ids[0], det_ids[1], flips_logical)
            else:
                # Hyperedge: decompose into all pairs
                # The logical flag applies to each pair — if the hyperedge flips
                # the logical, at least one pair in the decomposition should carry it.
                # Convention: assign is_logical only to the first pair to avoid
                # double-counting during peeling.
                for k, a in enumerate(det_ids):
                    for j, b in enumerate(det_ids):
                        if j <= k:
                            continue
                        il = flips_logical if (k == 0 and j == 1) else 0
                        add_undirected(a, b, il)
 
        self.adj = adj
 
        # Boundary flag array (used during UF merging to prefer boundary roots)
        self.is_boundary = np.zeros(self.num_nodes, dtype=np.uint8)
        self.is_boundary[self.boundary] = 1
 
    # ── Union-Find ────────────────────────────────────────────────────────────
 
    @staticmethod
    def _make_uf(n: int) -> list:
        return list(range(n))
 
    @staticmethod
    def _find(parent: list, x: int) -> int:
        """Path-halving find."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    def _union(self, parent: list, a: int, b: int) -> int:
        """Union by boundary preference then by smaller id. Returns new root."""
        ra, rb = self._find(parent, a), self._find(parent, b)
        if ra == rb:
            return ra
        if self.is_boundary[ra] and not self.is_boundary[rb]:
            parent[rb] = ra;  return ra
        if self.is_boundary[rb] and not self.is_boundary[ra]:
            parent[ra] = rb;  return rb
        if ra < rb:
            parent[rb] = ra;  return ra
        parent[ra] = rb;  return rb
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections).astype(np.uint8, copy=False)
        shots = detections.shape[0]
        out   = np.zeros((shots, self.num_obs), dtype=np.uint8)
        for s in range(shots):
            out[s] = self.decode(detections[s])
        return out
 
    def decode(self, dets_row: np.ndarray) -> np.ndarray:
        dets_row = np.asarray(dets_row).astype(np.uint8, copy=False)
        if dets_row.shape[0] != self.num_det:
            raise ValueError(f"Expected {self.num_det} detectors, got {dets_row.shape[0]}")

        odd_det = np.flatnonzero(dets_row & 1).astype(np.int32)
        if odd_det.size == 0:
            return np.zeros(self.num_obs, dtype=np.uint8)

        N      = self.num_nodes
        parent = self._make_uf(N)

        # Pass 1: union all edges reachable from odd detectors via BFS.
        # We only care about which nodes end up in the same component —
        # do NOT try to build prev[] here.
        visited = np.zeros(N, dtype=np.bool_)
        visited[self.boundary] = True
        q = deque()
        for d in odd_det:
            d = int(d)
            if not visited[d]:
                visited[d] = True
                q.append(d)

        while q:
            u = int(q.popleft())
            for e in self.adj[u]:
                v = int(e.nb)
                self._union(parent, u, v)
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        # Pass 2: build a correct spanning forest by scanning all visited edges
        # and greedily adding edges that connect two nodes where one has no parent
        # in the forest yet. This guarantees every node in a component has a path
        # to the component root with correct is_logical labels.
        prev     = np.full(N, -1, dtype=np.int32)
        prev_log = np.zeros(N, dtype=np.uint8)

        # Process edges in a consistent order: root nodes first so the tree
        # grows outward from roots (boundary or lowest-id odd detector).
        # We do this with a BFS seeded from component roots.
        in_forest = np.zeros(N, dtype=np.bool_)

        # Identify one root per component among visited nodes.
        # Root = boundary if present in component, else the UF root.
        comp_root: dict = {}
        for v in range(N):
            if not visited[v]:
                continue
            r = self._find(parent, v)
            if r not in comp_root:
                comp_root[r] = r
            if self.is_boundary[v]:
                comp_root[r] = v  # prefer boundary as forest root

        roots = set(comp_root.values())
        for r in roots:
            in_forest[r] = True

        q2 = deque(roots)
        while q2:
            u = int(q2.popleft())
            for e in self.adj[u]:
                v = int(e.nb)
                if visited[v] and not in_forest[v]:
                    # Only add if in same UF component
                    if self._find(parent, u) == self._find(parent, v):
                        in_forest[v]  = True
                        prev[v]       = u
                        prev_log[v]   = e.is_logical
                        q2.append(v)

        # Peeling
        deg   = np.zeros(N, dtype=np.int32)
        neigh: List[List[Tuple[int, int]]] = [[] for _ in range(N)]

        for v in range(N):
            u = int(prev[v])
            if u == -1:
                continue
            deg[v] += 1
            deg[u] += 1
            neigh[v].append((u, int(prev_log[v])))
            neigh[u].append((v, int(prev_log[v])))

        node_par          = np.zeros(N, dtype=np.uint8)
        node_par[odd_det] = 1

        peel         = deque(i for i in range(N) if deg[i] == 1)
        used_logical = 0

        while peel:
            x = int(peel.popleft())
            if deg[x] != 1:
                continue
            nb, il = -1, 0
            for y, e_il in neigh[x]:
                if deg[y] > 0:
                    nb, il = int(y), int(e_il)
                    break
            if nb == -1:
                deg[x] = 0
                continue
            if node_par[x] & 1:
                used_logical ^= il
                node_par[nb] ^= 1
            deg[x]  -= 1
            deg[nb] -= 1
            if deg[nb] == 1:
                peel.append(nb)

        pred               = np.zeros(self.num_obs, dtype=np.uint8)
        pred[self.logical] = used_logical & 1
        return pred
    




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
    #error_model = surface_code.detector_error_model(decompose_errors=True)
    #return UnionFindDecoder(error_model)
    return UnionFindDecoder(surface_code)

def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=True)


synd_rounds = 3
shots = 10000

# Plot logical error rate as distnace increases
def threshold():
    results = {}
    for dist in range(3,10,2):
        results[f'{dist}'] = []
        dist_results = []
        for r in range(1,21): # 0.0005 - 0.01
            per = 5*r/10000
            code = surface_code(dist, dist, per)
            dc = decoder(code)
            detections, observed_flips = run_simulations(code, shots)

            predictions = dc.decode_batch(detections)

            fails = 0
            #for i in range(shots):
            #    if not np.array_equal(predictions[i], observed_flips[i]):
            #        fails += 1


            fails = np.count_nonzero(
                np.any(predictions != observed_flips, axis=1)
            )

            #dem = code.detector_error_model(decompose_errors=True)
            #m = pymatching.Matching.from_detector_error_model(dem)

            #pred = m.decode_batch(detections)
            #fails = np.count_nonzero(np.any(pred != observed_flips, axis=1))


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

'''
results = threshold()
output_file = "results/union/results_threshold.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
'''

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01
)

dc = UnionFindDecoder(circuit)
import pymatching
dem = circuit.detector_error_model(decompose_errors=True)
m = pymatching.Matching.from_detector_error_model(dem)

sampler = circuit.compile_detector_sampler()
dets, obs = sampler.sample(shots=2000, separate_observables=True)

# Find first shot where yours is wrong and pymatching is right
for i in range(2000):
    odd = np.flatnonzero(dets[i]).tolist()
    if len(odd) == 0:
        continue
    your_pred = dc.decode(dets[i])
    pm_pred   = m.decode(dets[i])
    if not np.array_equal(your_pred, obs[i]) and np.array_equal(pm_pred, obs[i]):
        print(f"odd_dets = {odd}")
        print(f"obs      = {obs[i].tolist()}")
        print(f"your pred= {your_pred.tolist()}")
        print(f"pm pred  = {pm_pred.tolist()}")

        # Print every DEM edge that touches any of the odd detectors
        print("\nRelevant DEM edges:")
        odd_set = set(odd)
        for inst in dem:
            if inst.type != "error":
                continue
            det_ids = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            obs_ids = [t.val for t in inst.targets_copy() if t.is_logical_observable_id()]
            if any(d in odd_set for d in det_ids):
                print(f"  {det_ids} obs={obs_ids}")

        # Print the full forest for this shot
        N = dc.num_nodes
        parent   = dc._make_uf(N)
        visited  = np.zeros(N, dtype=np.bool_)
        visited[dc.boundary] = True
        q = deque()
        for d in odd:
            if not visited[d]:
                visited[d] = True
                q.append(d)
        while q:
            u = int(q.popleft())
            for e in dc.adj[u]:
                v = int(e.nb)
                dc._union(parent, u, v)
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        # Show adjacency for each odd detector
        print("\nAdj edges for odd detectors:")
        for d in odd:
            edges = [(e.nb, e.is_logical) for e in dc.adj[d]]
            print(f"  det {d}: {edges}")

        break