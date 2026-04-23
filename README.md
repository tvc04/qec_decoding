# qec_decoding

Repository for all things QEC Surface Code Decoding! My research report can be found [here](Benchmarking Quantum Error Correction Decoders on the Surface Code.pdf).

## How to Use

Run ```pip install -r requirements.txt``` to access all required libraries. If certain ones aren't accessible (likely cudaq-qec for many devices), individual files can be tested by running ```python <decoder_file>.py <test_type>```.

If all libraries are installed, run tests with the following command: ```python manager.py <decoder_type> <test_type> ['plot']```

- ```<decoder_type>```: 1 = MWPM, 2 = Union Find, 3 = Tensor Network, 4 = Neural Network

- ```<test_type>```: 1 = Correctness, 2 = Latency, 3 = Threshold, 4 = Robustness, 5 = Scalability

- Optional flag ```plot```:  only generates plot for specified test

## Motivation

For my research project, I will be attempting to replicate the results found in each of the studies for the different decoders I want to compare. I want to choose one surface code decoder from each of the following categories; MWPM, Union-Find, Neural Network, and Tensor Network. My initial ideas for which specific decoders I will use are PyMatching, Helios or the Delfosse-Nickerman implementation, the neural network decoder from [Delft](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.013029), and either the blockBP or Bravyi–Suchara–Vargo (BSV) Tensor Network Decoder. 

For each of the decoder categories, I want to perform tests for the individual decoders’ correctness, latency, physical error rate threshold, robustness, and scalability. To test for decoder correctness, I will run many iterations generating error patterns, obtaining stabilizer measurements, decoding the syndrome, and performing corrective measures. If the logical state is preserved, the decoder will have performed correctly. To test for latency, I can simply add a timer that tracks how long each iteration takes between obtaining the stabilizer measurements and applying the corrective operations since that is the time during which the actual decoding occurs. To test for physical error rate threshold, I will run many similar simulations for increasing code distances and physical error rates within a certain range. I can then compute the logical error rate and plot that against the physical error rate for each distance, with the point where all the curves intersect being the threshold. To test for robustness, I can change the noise model from the previous simulations to include new errors like measurement errors, leakage, and depolarization. I can then repeat the correctness and/or threshold experiments to see if there are significant differences between the different noise models. Finally, to test scalability, I can simply increase the code distance for the tests for which that is applicable. This will mainly apply to the correctness and latency tests. 

After completing all of these tests, I will examine results from previously performed studies for these decoders and see if my results are similar to those. Additionally, I will have arrived at a comprehensive comparison between all of them which will allow me to make informed decisions about which types of decoders are better for certain applications, especially as it pertains to the transition from noisy intermediate scale quantum computing to fault tolerance.
