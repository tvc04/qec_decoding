import numpy as np
import stim
import time
import quimb.tensor as qtn
import cotengra as ctg
from collections import Counter


# ============================================================
# 1. Stim circuit
# ============================================================

def make_circuit(distance, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=1,
        after_clifford_depolarization=p
    )


# ============================================================
# 2. DEM → parity checks
# ============================================================

def dem_to_checks(dem):
    checks = []

    for inst in dem:
        if inst.type == "error":
            detectors = []

            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    detectors.append(t.val)

            if len(detectors) > 0:
                checks.append(detectors)

    return checks


# ============================================================
# 3. Build Tensor Network
# ============================================================

def build_tn_from_dem(checks, syndrome, p):
    tn = qtn.TensorNetwork()

    # All detector variables
    all_vars = sorted(set(v for check in checks for v in check))  # <<< sorted for deterministic order
    var_inds = []

    # --- Variable tensors ---
    for v in all_vars:
        ind = f'd{v}'
        var_inds.append(ind)

        tn.add_tensor(qtn.Tensor(
            data=np.array([1 - p, p], dtype=float),
            inds=(ind,),
            tags={f'D{v}'}
        ))

    # --- Check tensors ---
    for i, check in enumerate(checks):
        inds = tuple(f'd{v}' for v in check)
        shape = (2,) * len(check)

        data = np.zeros(shape, dtype=float)
        s = int(sum(syndrome[v] for v in check) % 2)

        for idx in np.ndindex(shape):
            if (sum(idx) % 2) == s:
                data[idx] = 1.0

        tn.add_tensor(qtn.Tensor(
            data=data,
            inds=inds,
            tags={f'C{i}'}
        ))

    return tn, var_inds


# ============================================================
# 4. Cotengra path precompute + contraction wrapper  <<< ADDED
# ============================================================

def _find_open_inds(tn):
    c = Counter(ind for T in tn.tensors for ind in T.inds)
    return tuple(ind for ind, n in c.items() if n == 1)

def precompute_trees_with_hyperoptimizer(tn_template, var_inds,
                                        max_repeats=32, progbar=False):
    size_dict = tn_template.ind_sizes

    opt = ctg.HyperOptimizer(
        max_repeats=max_repeats,
        progbar=progbar,
        minimize='flops',
    )

    trees = {}

    # ----- Scalar tree (only if TN is closed) -----
    open_inds = _find_open_inds(tn_template)
    if len(open_inds) == 0:
        # quimb builds the tree using cotengra optimizer
        trees[()] = tn_template.contraction_tree(
            output_inds=(),
            optimize=opt,
        )
    else:
        # Not closed, skip scalar tree (or set output_inds=open_inds)
        # Often you actually want output_inds=open_inds for a "partition function with open legs".
        # We'll skip to avoid confusion.
        print(f"[warn] TN has open indices (scalar contraction invalid): {open_inds}")

    # ----- One tree per marginal -----
    for ind in var_inds:
        trees[(ind,)] = tn_template.contraction_tree(
            output_inds=(ind,),
            optimize=opt,
        )

    return trees


def contract_with_tree(tn, tree, output_inds=None):
    """
    Contract TN using a precomputed cotengra ContractionTree.
    """
    return tn.contract(output_inds=output_inds, optimize=tree)


# ============================================================
# 5. Decoder (now uses cached trees)  <<< MODIFIED
# ============================================================

def decode_bsv_with_trees(tn, var_inds, trees):
    marginals = {}

    for ind in var_inds:
        tn_copy = tn.copy()

        tree = trees[(ind,)]
        result = contract_with_tree(tn_copy, tree, output_inds=(ind,))

        prob = np.asarray(result.data, dtype=float)

        if prob.sum() == 0:
            prob = np.array([0.5, 0.5], dtype=float)
        else:
            prob = prob / prob.sum()

        marginals[ind] = prob

    correction = {k: int(v[1] > v[0]) for k, v in marginals.items()}
    return correction, marginals


# ============================================================
# 6. Simulation loop  <<< MODIFIED
# ============================================================

def run_bsv_simulation(distance=3, p=0.05, shots=100, chi=8,
                       hyper_max_repeats=32, hyper_progbar=False):

    circuit = make_circuit(distance, p)
    dem = circuit.detector_error_model()
    checks = dem_to_checks(dem)

    if len(checks) == 0:
        raise ValueError("No parity checks extracted from DEM.")

    sampler = circuit.compile_detector_sampler()

    # --- Build a template TN once and precompute trees ---
    dets0, obs0 = sampler.sample(1, separate_observables=True)
    dets0 = dets0[0]

    tn_template, var_inds = build_tn_from_dem(checks, dets0, p)

    trees = precompute_trees_with_hyperoptimizer(
        tn_template,
        var_inds,
        max_repeats=hyper_max_repeats,
        progbar=hyper_progbar
    )

    failures = 0
    latencies = []

    for shot in range(shots):
        dets, obs = sampler.sample(1, separate_observables=True)
        dets = dets[0]

        tn, var_inds2 = build_tn_from_dem(checks, dets, p)
        # sanity check structure didn't change
        assert var_inds2 == var_inds

        start = time.perf_counter()
        correction, _ = decode_bsv_with_trees(tn, var_inds, trees)
        end = time.perf_counter()
        latencies.append(end-start)

        # --- VERY SIMPLE logical check (placeholder) ---
        predicted = sum(correction.values()) % 2
        actual = obs[0][0]

        if predicted != actual:
            failures += 1

        if shot % 100 == 0:
            print(f"Shot {shot}: correction size = {len(correction)}")

    return failures / shots, np.mean(latencies)


# ============================================================
# 7. Run example
# ============================================================

if __name__ == "__main__":
    d = 5
    p = 0.001
    shots = 1000
    chi = 8

    logical_error_rate, avg_latency = run_bsv_simulation(
        distance=d,
        p=p,
        shots=shots,
        chi=chi,
        hyper_max_repeats=64,     # tune
        hyper_progbar=True
    )

    print("\n=== RESULT ===")
    print(f"Distance: {d}")
    print(f"Physical error rate: {p}")
    print(f"Logical error rate: {logical_error_rate}")
    print(f"Logical error rate: {avg_latency}")