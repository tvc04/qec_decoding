import sys

import mwpm
import union_find
import tensor

import create_correctness_plot
import create_latency_plot
import create_threshold_plot
import create_robustness_plot
import create_scalability_plot


# -----------------------------
#       MAIN MANAGER CODE
# -----------------------------

def main():
    if (len(sys.argv) != 3 and len(sys.argv) != 4) or (int(sys.argv[1]) not in range(1,5)) or (int(sys.argv[2]) not in range(1,6)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=TN, 4=NN) and Test Type " +
        "(1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability) " + 
        "*Optional flag \'plot\' to only generate plot for specified test\n" + 
        "Sample input: python manager.py <decoder type> <test type> [plot]")
        return
    
    generate_data = True
    if len(sys.argv) == 4 and sys.argv[3] == "plot":
        generate_data = False
    
    decoder_type = int(sys.argv[1])
    test_type = int(sys.argv[2])
    
    if generate_data:
        if (decoder_type == 1):
            mwpm.mwpm_test(test_type)
        if (decoder_type == 2):
            union_find.union_find_test(test_type)
        if (decoder_type == 3):
            tensor.tensor_test(test_type)
        if (decoder_type == 4):
            return
    
    if (test_type == 1):
        create_correctness_plot.generate_plot(decoder_type)
    if (test_type == 2):
        create_latency_plot.generate_plot(decoder_type)
    if (test_type == 3):
        create_threshold_plot.generate_plot(decoder_type)
    if (test_type == 4):
        create_robustness_plot.generate_plot(decoder_type)
    if (test_type == 5):
        create_scalability_plot.generate_plot(decoder_type)

if __name__ == "__main__":
    main()