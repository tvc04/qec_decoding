import numpy as np
import time

from panqec.codes.surface_2d import RotatedPlanar2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import UnionFindDecoder
from panqec.simulation import DirectSimulation


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5
shots = 100000



# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------

def run_UF_simulation():
    for dist in range(3,10,2):
        surface_code = RotatedPlanar2DCode(dist)
        error_model = PauliErrorModel(r_x=1/3, r_y=1/3, r_z=1/3)
        decoder = UnionFindDecoder(surface_code, error_model, per)
        sim = DirectSimulation(code=surface_code, error_model=error_model, decoder=decoder, error_rate=per)

        fails = 0
        latencies = []

        for i in range(1):
            start = time.perf_counter()
            sim.run(1)
            end = time.perf_counter()

            latencies.append(end - start)
            print(sim.results)
        
        log_error_rate = fails/shots
        avg_latency = np.mean(latencies)

        print(f"Distance: {dist}, Physical Error Rate: {per}")
        print(f"Logical Error Rate: {log_error_rate}")
        print(f"Average Decoding Latency: {avg_latency}")





# ---------------------------
#       TEST FUNCTIONS
# ---------------------------

def correctness():
    pass

def latency():
    pass

def threshold():
    pass

def robustness():
    pass

def scalability():
    pass

def union_find_test(test_num):
    output_plot = None

    if test_num == 1:
        output_plot = run_UF_simulation()
    if test_num == 2:
        output_plot = latency()
    if test_num == 3:
        output_plot = threshold()
    if test_num == 4:
        output_plot = robustness()
    if test_num == 5:
        output_plot = scalability()
    
    # Display plot?