import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os


# LSTM Model
# Keep this OUTSIDE main().
# FastAPI needs this class when loading the existing .pth file.

import torch.nn as nn
# Configuration
input_size = 4
hidden_size = 64
num_layers = 2


sequence_length = 10
input_size = 4
hidden_input = 64
num_layers = 5
output_size = 2

batch_size = 64
epochs = 30
learning_rate = 0.001


# LSTM Model
# Keep this OUTSIDE main().
# FastAPI needs this class when loading the existing .pth file.

import torch.nn as nn
import torch.nn as nn

class TrajectoryGRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super().__init__()
        self.gru = nn.GRU(input_size = input_size,hidden_size = hidden_size, num_layers=num_layers,batch_first = True)
        self.fc1 = nn.Linear(hidden_size,64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(64,output_size)

    def forward(self,x):
        output, hidden = self.gru(x)

        last_output = output[:,-1,:]
        x = self.fc1(last_output)

        x = self.relu(x)

        x = self.dropout(x)

        prediction = self.fc2(x)

        return prediction


def main():

    # CUDA

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    device = torch.device("cuda:0")

    print("=" * 60)
    print("device configuration")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 60)


    # Paths

    base_dir = r"C:\Users\HP\Desktop\multi_object_tracking"

    x_train_path = os.path.join(
        base_dir,
        "data",
        "sequences",
        "x_train.npy"
    )

    y_train_path = os.path.join(
        base_dir,
        "data",
        "sequences",
        "y_train.npy"
    )
    x_test_path = os.path.join(
            base_dir,
            "data",
            "sequences",
            "x_test.npy"
        )
    
    y_test_path = os.path.join(
            base_dir,
            "data",
            "sequences",
            "y_test.npy"
        )

    x_val_path = os.path.join (base_dir,"data","sequences","x_val.npy")
    y_val_path = os.path.join(base_dir,"data","sequences","y_val.npy")


    model_dir = os.path.join(
        base_dir,
        "models"
    )
    model_path = os.path.join(
            model_dir,
            "trajectory_gru.pth"
        )
    
    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # Load Data

    print("\n" + "=" * 60)
    print("loading training data")
    print("=" * 60)

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_val = np.load(x_val_path)
    y_val = np.load(y_val_path)


    print(f"X shape: {x_train.shape}")
    print(f"Y shape: {y_train.shape}")


    print(f"\nTraining samples: {len(x_train)}")


    # Convert to PyTorch tensors

    x_train = torch.tensor(
        x_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    x_val = torch.tensor(x_val,dtype = torch.float32)
    y_val = torch.tensor(y_val,dtype = torch.float32)

#    x_test = torch.tensor(
#        x_test,
#        dtype=torch.float32
#   )

#    y_test = torch.tensor(
#        y_test,
#        dtype=torch.float32
#    )


    # Dataset and DataLoader

    train_dataset = TensorDataset(
        x_train,
        y_train
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
#test_dataset = TensorDataset(
#        x_test,
#       y_test
#   )

# test_loader = DataLoader(
#        test_dataset,
#        batch_size=batch_size,
#        shuffle=False
#    )
    val_dataset = TensorDataset(x_val,y_val)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)


    # Create model

    model = TrajectoryGRU(
        input_size,
        hidden_input,
        num_layers,
        output_size
    ).to(device)


    # Loss Function

    criterion = nn.MSELoss()


    # Optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # Training
    best_val_loss = float("inf")
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction = model(batch_x)

            loss = criterion(
                prediction,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        average_loss = (
            epoch_loss / len(train_loader)
        )
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch,y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions,y_batch)
                val_loss+=loss.item()
            val_loss /= len(val_loader)


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_loss:.6f}",
            f"Val loss: {val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model,model_path)
            print("Best_model saved.")


if __name__ == "__main__":
    main()