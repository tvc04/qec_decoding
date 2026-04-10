from qsurface.main import initialize, run


# 1. Initialize planar surface code with Union-Find decoder
code, decoder = initialize(
    size=5,                      # distance (5x5 planar code)
    Code="planar",
    Decoder="unionfind",
    enabled_errors=["pauli"],    # depolarizing-like errors
    check_compatibility=True
)

# 2. Run simulation
results = run(
    code,
    decoder,
    iterations=1000,
    error_rates={"p_bitflip": 0.01}
)

print(results)