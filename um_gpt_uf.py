from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import stim

import json


# --------------------------
# Helper: parse Stim DEM
# --------------------------

@dataclass(frozen=True)
class _Edge:
    a: int
    b: int
    obs_mask: int  # xor mask of logical observables toggled by traversing/using this edge


def _iter_dem_instructions_flat(dem: stim.DetectorErrorModel):
    stack = [iter(dem)]
    while stack:
        it = stack[-1]
        try:
            inst = next(it)
        except StopIteration:
            stack.pop()
            continue
        if isinstance(inst, stim.DemRepeatBlock):
            body = inst.body_copy()
            for _ in range(inst.repeat_count):
                stack.append(iter(body))
            continue
        yield inst


def _parse_dem_to_graph_with_hyperedge_reduction(
    dem: stim.DetectorErrorModel,
) -> Tuple[List[_Edge], int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a pairwise graph from DEM, reducing k-detector hyperedges (k>=3) using an aux node per term.

    Node layout:
      - detectors: [0 .. num_det-1]
      - boundaries: [num_det .. 2*num_det-1]  (one boundary node per detector for 1-det terms)
      - aux nodes: appended after that, one per hyperedge term

    Reduction for a hyperedge touching detectors d0..d{k-1} with obs_mask M:
      introduce aux node h
      add edges (h, d0) with obs_mask M
      add edges (h, di) with obs_mask 0 for i>0

    Why this works (parity-wise):
      Selecting all k edges flips all k detectors (aux cancels out mod 2).
    """
    num_det = dem.num_detectors
    num_obs = dem.num_observables

    edges: List[_Edge] = []
    aux_count = 0

    # We'll finalize num_nodes after counting aux nodes.
    # Temporarily store hyperedges as lists of detectors + obs_mask, then expand.
    hyper_terms: List[Tuple[List[int], int]] = []

    for inst in _iter_dem_instructions_flat(dem):
        if inst.type != "error":
            continue
        targets = inst.targets_copy()

        dets: List[int] = []
        obs_mask = 0
        for t in targets:
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs_mask ^= (1 << t.val)

        k = len(dets)
        if k == 0:
            # Undetectable (pure logical) term: cannot infer from syndrome; ignore.
            continue
        elif k == 1:
            d = dets[0]
            b = num_det + d
            edges.append(_Edge(d, b, obs_mask))
        elif k == 2:
            a, b = dets
            if a != b:
                edges.append(_Edge(a, b, obs_mask))
        else:
            hyper_terms.append((dets, obs_mask))
            aux_count += 1

    # Expand hyperedges using aux nodes
    num_nodes = 2 * num_det + aux_count
    aux_base = 2 * num_det
    aux_i = 0
    for dets, obs_mask in hyper_terms:
        h = aux_base + aux_i
        aux_i += 1
        # Connect to first detector with obs_mask, rest with 0 (star)
        d0 = dets[0]
        edges.append(_Edge(h, d0, obs_mask))
        for d in dets[1:]:
            edges.append(_Edge(h, d, 0))

    # Build CSR adjacency
    deg = np.zeros(num_nodes, dtype=np.int32)
    for e in edges:
        deg[e.a] += 1
        deg[e.b] += 1

    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    np.cumsum(deg, out=offsets[1:])

    nbr = np.empty(offsets[-1], dtype=np.int32)
    eidx = np.empty(offsets[-1], dtype=np.int32)

    cursor = offsets[:-1].copy()
    for i, e in enumerate(edges):
        pa = cursor[e.a]
        nbr[pa] = e.b
        eidx[pa] = i
        cursor[e.a] += 1

        pb = cursor[e.b]
        nbr[pb] = e.a
        eidx[pb] = i
        cursor[e.b] += 1

    return edges, num_nodes, num_obs, offsets, nbr, eidx


class _UF:
    __slots__ = ("parent", "size", "parity")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.parity = np.zeros(n, dtype=np.uint8)

    def reset(self):
        self.parent[:] = np.arange(self.parent.size, dtype=np.int32)
        self.size[:] = 1
        self.parity[:] = 0

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.parity[ra] ^= self.parity[rb]
        return ra


class UnionFindDecoder:
    """
    UF-family decoder with hyperedge reduction via aux nodes.

    Fast enough for ~100k shots per (p, d) at modest distances in Python.
    """

    def __init__(self, error_model: stim.DetectorErrorModel):
        self.dem = error_model
        self.num_det = error_model.num_detectors
        self.num_obs = error_model.num_observables

        edges, num_nodes, _, offsets, nbr, eidx = _parse_dem_to_graph_with_hyperedge_reduction(error_model)

        self.edges = edges
        self.num_nodes = num_nodes
        self.offsets = offsets
        self.nbr = nbr
        self.eidx = eidx

        self.uf = _UF(num_nodes)

        # boundary nodes are [num_det .. 2*num_det-1]
        self._boundary_start = self.num_det
        self._boundary_end = 2 * self.num_det

        self._is_boundary = np.zeros(num_nodes, dtype=np.uint8)
        self._is_boundary[self._boundary_start:self._boundary_end] = 1

        # reusable buffers
        self._queue = np.empty(num_nodes, dtype=np.int32)
        self._root_seen = np.full(num_nodes, -1, dtype=np.int32)
        self._root_prev_r = np.full(num_nodes, -1, dtype=np.int32)
        self._root_prev_e = np.full(num_nodes, -1, dtype=np.int32)

        self._forest_deg = np.zeros(num_nodes, dtype=np.int32)
        self._forest_nbr = np.full(num_nodes, -1, dtype=np.int32)
        self._forest_edge = np.full(num_nodes, -1, dtype=np.int32)

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections)
        shots = detections.shape[0]
        out = np.zeros((shots, self.num_obs), dtype=np.uint8)
        for s in range(shots):
            out[s] = self.decode(detections[s])
        return out

    def decode(self, dets_row: np.ndarray) -> np.ndarray:
        dets_row = np.asarray(dets_row).astype(np.uint8, copy=False)
        if dets_row.shape[0] != self.num_det:
            raise ValueError(f"Expected {self.num_det} detectors, got {dets_row.shape[0]}")

        uf = self.uf
        uf.reset()

        fired = np.flatnonzero(dets_row & 1).astype(np.int32, copy=False)
        if fired.size == 0:
            return np.zeros(self.num_obs, dtype=np.uint8)

        uf.parity[fired] = 1

        root_seen = self._root_seen
        root_prev_r = self._root_prev_r
        root_prev_e = self._root_prev_e
        root_seen.fill(-1)
        root_prev_r.fill(-1)
        root_prev_e.fill(-1)

        q = self._queue
        qhead = 0
        qtail = 0

        # Seed with boundaries as source_id = -2
        for b in range(self._boundary_start, self._boundary_end):
            rb = uf.find(b)
            if root_seen[rb] == -1:
                root_seen[rb] = -2
                q[qtail] = b
                qtail += 1

        # Seed with odd fired components as unique sources (use detector index as id)
        for d in fired:
            rd = uf.find(int(d))
            if (uf.parity[rd] & 1) == 0:
                continue
            if root_seen[rd] == -1:
                root_seen[rd] = int(d)
                q[qtail] = int(d)
                qtail += 1

        offsets = self.offsets
        nbr = self.nbr
        eidx = self.eidx

        forest_edge_indices: List[int] = []

        def add_path_edges(r_from: int):
            r = r_from
            while True:
                pr = int(root_prev_r[r])
                if pr == -1:
                    break
                pe = int(root_prev_e[r])
                forest_edge_indices.append(pe)
                r = pr

        # Multi-source BFS with meet-and-merge
        while qhead < qtail:
            u = int(q[qhead]); qhead += 1
            ru = uf.find(u)
            su = root_seen[ru]
            if su == -1:
                continue

            start = int(offsets[u])
            end = int(offsets[u + 1])
            for k in range(start, end):
                v = int(nbr[k])
                ei = int(eidx[k])

                rv = uf.find(v)
                sv = root_seen[rv]
                if sv == -1:
                    root_seen[rv] = su
                    root_prev_r[rv] = ru
                    root_prev_e[rv] = ei
                    q[qtail] = v
                    qtail += 1
                    continue

                if sv == su:
                    continue

                # meet: connect the two trees and union
                add_path_edges(ru)
                add_path_edges(rv)
                forest_edge_indices.append(ei)

                newr = uf.union(u, v)
                merged_source = -2 if (su == -2 or sv == -2) else su
                root_seen[newr] = merged_source

        # Build a (possibly redundant) forest adjacency summary and peel
        forest_deg = self._forest_deg
        forest_nbr_arr = self._forest_nbr
        forest_edge_arr = self._forest_edge
        forest_deg[:] = 0
        forest_nbr_arr[:] = -1
        forest_edge_arr[:] = -1

        # Deduplicate edges
        if forest_edge_indices:
            forest_edge_indices = list(set(forest_edge_indices))

        for ei in forest_edge_indices:
            e = self.edges[ei]
            a = uf.find(e.a)
            b = uf.find(e.b)
            if a == b:
                continue
            forest_deg[a] += 1
            forest_deg[b] += 1
            # store one neighbor for leaf-peel
            forest_nbr_arr[a] = b
            forest_edge_arr[a] = ei
            forest_nbr_arr[b] = a
            forest_edge_arr[b] = ei

        peel_q = deque()

        # Candidate roots: roots of fired detectors plus boundaries
        candidate_nodes = np.concatenate(
            [fired, np.arange(self._boundary_start, self._boundary_end, dtype=np.int32)]
        )
        candidate_roots = np.unique([uf.find(int(x)) for x in candidate_nodes])

        for r in candidate_roots:
            if forest_deg[r] == 1 and (uf.parity[r] & 1) == 1 and self._is_boundary[r] == 0:
                peel_q.append(int(r))

        used_obs_mask = 0

        while peel_q:
            r = int(peel_q.popleft())
            if forest_deg[r] != 1:
                continue
            nb = int(forest_nbr_arr[r])
            ei = int(forest_edge_arr[r])
            if nb < 0 or ei < 0:
                continue

            # If r is odd, use the edge and push parity
            if uf.parity[r] & 1:
                used_obs_mask ^= self.edges[ei].obs_mask
                uf.parity[nb] ^= 1

            # Remove edge
            forest_deg[r] -= 1
            forest_deg[nb] -= 1

            if forest_deg[nb] == 1 and (uf.parity[nb] & 1) == 1 and self._is_boundary[nb] == 0:
                peel_q.append(nb)

        pred = np.zeros(self.num_obs, dtype=np.uint8)
        for k in range(self.num_obs):
            pred[k] = (used_obs_mask >> k) & 1
        return pred
    




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
    return UnionFindDecoder(error_model)

# Generates sample error syndrome
def run_simulations(surface_code, shots):
    samples = surface_code.compile_detector_sampler()
    return samples.sample(shots=shots, separate_observables=True)


synd_rounds = 1
shots = 100000

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


results = threshold()
output_file = "results/union/results_threshold.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")

#code = surface_code(5, 5, 0.005)
#dem = code.detector_error_model(decompose_errors=True)
#print(any(sum(t.is_relative_detector_id() for t in inst.targets_copy()) > 2
#        for inst in dem if getattr(inst, "type", None) == "error"))