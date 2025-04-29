import argparse
from pathlib import Path
import h5py
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class SimpleModel(nn.Module):
    # A simple model that contains a single linear layer
    def __init__(self, in_features, out_features):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features, out_features, False)
        self.initialize_weights()

    def initialize_weights(self):
        # Check if in_features == out_features for identity initialization
        if self.fc.weight.shape[0] == self.fc.weight.shape[1]:
            nn.init.eye_(self.fc.weight)  # Initialize with identity matrix
        else:
            raise ValueError("Identity initialization requires in_features == out_features")

    def forward(self, x):
        return self.fc(x)


class UQModel(nn.Module):
    def __init__(self, _base):
        super(UQModel, self).__init__()
        self.base = _base

    def forward(self, x):
        uncertainty = x / 1000
        out = self.base(x)
        return out, uncertainty


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and store a model.")
    parser.add_argument("--filename", "-fn", type=str, required=True, help="The file containing the data to train on")
    parser.add_argument("--epochs", "-e", help="Number of epochs", default=10)
    parser.add_argument("--model-file", "-m", help="Filename of model", default="model")
    args = parser.parse_args()
    if not Path(args.filename).exists():
        raise RuntimeError(f"Please provide an existing file, file {args.filename} does not exist")
    device = "cpu"

    with h5py.File(args.filename, "r") as fd:
        X, y = fd["input_data"][:], fd["output_data"][:]
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        model = SimpleModel(X.shape[-1], y.shape[-1])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            print(f"Epoch {epoch:2d}/{args.epochs} — Loss: {epoch_loss:.4f}")

        model = UQModel(model)  # Set to double precision (float64)
        prec = torch.float32
        example_input = torch.from_numpy(X)

        # Trace the model
        scripted_model = torch.jit.trace(model, example_input)

        # Generate the file name
        file_path = f"{args.model_file}.pt"
        # Save the scripted model
        scripted_model.save(file_path)

        print(f"Model saved to {file_path}")


if __name__ == "__main__":
    main()
