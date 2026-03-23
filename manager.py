import sys
import mwpm


# -----------------------------
#       MAIN MANAGER CODE
# -----------------------------

def main():
    if len(sys.argv) != 3 or (int(sys.argv[1]) not in range(1,5)) or (int(sys.argv[2]) not in range(1,6)):
        print("Specify Decoder Type (1=MWPM, 2=Union, 3=NN, 4=TN) and Test Type " +
        "(1=Correctness, 2=Latency, 3=Threshold, 4=Robustness, 5=Scalability)")
        return
    
    decoder_type = int(sys.argv[1])
    test_type = int(sys.argv[2])
    
    if (decoder_type == 1):
        mwpm.mwpm_test(test_type)
    if (decoder_type == 2):
        return
    if (decoder_type == 3):
        return
    if (decoder_type == 4):
        return

if __name__ == "__main__":
    main()