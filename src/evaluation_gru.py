import numpy as np
import torch
import os
from models.trajectory_gru_arch import TrajectoryGRU


base_dir = r"C:\Users\HP\Desktop\multi_object_tracking"

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

model_path = os.path.join(
    base_dir,
    "models",
    "trajectory_gru.pth"
)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


print("=" * 60)
print("DEVICE CONFIGURATION")
print("=" * 60)

print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print("=" * 60)



print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

print(f"X shape : {x_test.shape}")
print(f"Y shape : {y_test.shape}")


print("\n" + "=" * 60)
print("TEST DATA")
print("=" * 60)
print(f"Test samples  : {len(x_test)}")



x_test = torch.tensor(
    x_test,
    dtype=torch.float32
).to(device)



print("\n" + "=" * 60)
print("LOADING SAVED LSTM MODEL")
print("=" * 60)

model = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

model = model.to(device)

model.eval()

print("Saved LSTM model loaded successfully.")


print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

with torch.no_grad():

    predictions = model(x_test)


# Move predictions to CPU

predictions = predictions.cpu().numpy()


print(f"Prediction shape : {predictions.shape}")
print(f"Actual shape     : {y_test.shape}")



error = predictions - y_test

mae = np.mean(
    np.abs(error)
)



rmse = np.sqrt(
    np.mean(
        error ** 2
    )
)



distance_error = np.sqrt(
    np.sum(
        error ** 2,
        axis=1
    )
)



ade = np.mean(
    distance_error
)




fde = np.mean(
    distance_error
)



print("\n" + "=" * 60)
print("LSTM EVALUATION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f} normalised_unit_score")
print(f"RMSE : {rmse:.4f} normalised_unit_score")
print(f"ADE  : {ade:.4f} normalised_unit_score")
print(f"FDE  : {fde:.4f} normalised_unit_score")

print("=" * 60)
   

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(min(10, len(predictions))):

    print(
        f"Sample {i + 1}: "
        f"Actual = "
        f"({y_test[i][0]:.2f}, {y_test[i][1]:.2f}) | "
        f"Predicted = "
        f"({predictions[i][0]:.2f}, "
        f"{predictions[i][1]:.2f})"
    )

print("=" * 60)


print("\nEvaluation completed successfully.")
print("No training was performed.")