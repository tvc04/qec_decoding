import numpy as np
import localuf

def run_threshold(
    distances=[3,5,7,9],
    phys_errs=np.linspace(0.005, 0.03, 10),
    shots=10000
):
    results = {}

    for dist in distances:
        print(f"Running distance {dist}")
        results[dist] = []

        # Build code (surface code)
        code = localuf.codes.Surface(dist, 'circuit-level')

        # Build decoder (Union-Find variant)
        decoder = localuf.decoders.uf.UF(code)

        for per in phys_errs:
            failures = 0

            for _ in range(shots):
                # Sample noise + syndrome
                error, syndrome = code.sample(per)

                # Decode
                correction = decoder.decode(syndrome)

                # Check logical failure
                if code.is_logical_error(error ^ correction):
                    failures += 1

            log_error_rate = failures / shots

            print()
            print(f"Distance: {dist}, Physical Error Rate: {per}")
            print(f"Logical Error Rate: {log_error_rate:.8f}")
            print()

run_threshold()