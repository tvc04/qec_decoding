import stim
import numpy as np
import json

import torch
import torch.nn as nn
import os


# -----------------------------
#       TEST PARAMETERS
# -----------------------------

# Default values (changed in tests)
dist = 5
per = 0.001     # 1/1000
synd_rounds = 5 # 1
shots = 1000    # 10,000
epochs = 8      # 15
batch_size = 128   #1024
nn_dir = "nn_models"
error_rates = [5*i/100000 for i in range(1,41)] # 0.00005 - 0.002

os.makedirs(nn_dir, exist_ok=True)


# --------------------------------------------------------
#       CIRCUIT CONSTRUCTION / SIMULATION FUNCTIONS
# --------------------------------------------------------

# Generate NN Data
def generate_data(distance, shots, phys_error_rate, depolarization = 0, measure = 0, reset = 0):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=synd_rounds,
        after_clifford_depolarization=phys_error_rate,
        before_round_data_depolarization=phys_error_rate*depolarization,
        before_measure_flip_probability=phys_error_rate*measure,
        after_reset_flip_probability=phys_error_rate*reset
    )

    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots, append_observables=True)

    num_detectors = circuit.num_detectors
    syndromes = samples[:, :num_detectors]
    observables = samples[:, num_detectors:]

    return syndromes.astype(np.float32), observables.astype(np.float32)

def build_dataset(distance, depolarization = 0, measure = 0, reset = 0):
    X_list, y_list = [], []

    for per in error_rates:
        X, y = generate_data(distance, shots, per, depolarization, measure, reset)
        X_list.append(X)
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y

# Decoder Class
class DecoderNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Train NN Model
def train_model(model, X, y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(y)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")



def make_model(depolarization = 0, measure = 0, reset = 0):
    metadata = {}

    for dist in range(3,10,2):
        print(f"\n=== Training for distance {dist} ===")

        # Build dataset across multiple error rates
        x, y = build_dataset(dist, depolarization, measure, reset)

        input_size = x.shape[1]
        model = DecoderNN(input_size)

        train_model(model, x, y)

        # Save model
        model_path = os.path.join(nn_dir, f"decoder_{depolarization}{measure}{reset}_d{dist}.pt")
        torch.save(model.state_dict(), model_path)

        # Save metadata
        metadata[f"{dist}"] = {
            "input_size": input_size,
            "error_rates": error_rates,
            "shots_per_rate": shots
        }

        print(f"Saved model to {model_path}")

    return metadata
    


def main():
    metadata = {}
    print("\nTraining basic model")
    metadata["000"] = []
    basic_data = make_model()
    metadata["000"] = basic_data
    print("\nTraining depolarization model")
    metadata["100"] = []
    depol_data = make_model(depolarization=1)
    metadata["100"] = depol_data
    print("\nTraining measure model")
    metadata["010"] = []
    meas_data = make_model(measure=1)
    metadata["010"] = meas_data
    print("\nTraining reset model")
    metadata["001"] = []
    reset_data = make_model(reset=1)
    metadata["001"] = reset_data
    print("\nTraining complete model")
    metadata["111"] = []
    comp_data = make_model(depolarization=1, measure=1, reset=1)
    metadata["111"] = comp_data

    print("\nTraining complete!")

    with open(os.path.join(nn_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
