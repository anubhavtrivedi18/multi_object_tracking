import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.trajectory_lstm_arch import TrajectoryLSTM

BASE_DIR = r"C:\Users\HP\Desktop\multi_object_tracking"

x_test_path = os.path.join(BASE_DIR, "data", "sequences", "x_test.npy")
y_test_path = os.path.join(BASE_DIR, "data","sequences", "y_test.npy")

model_path = os.path.join(BASE_DIR, "models", "trajectory_lstm.pth")

output_dir = os.path.join(BASE_DIR, "data", "visualizations")

os.makedirs(output_dir, exist_ok = True)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


print("="*60)
print("Trajectory Visualization ")
print("="*60)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



# load the data
print("\n"+"="*60)
print("Lodaing Data")

x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

print(f"x_shape : {x_test.shape}")
print(f"y shape : {y_test.shape}")



print(f"Test.samples : {len(x_test)}")



model = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

model.eval()
print("Model loaded successfully")


# Selecting one sample
sample_index = 4
sample_x = x_test[sample_index]
actual_position = y_test[sample_index]

print(f"sample index : {sample_index}")
print(f"Actual future position:"
      f"({actual_position[0]:.2f},{actual_position[1]:.2f})")

# generating the prediction

sample_tensor = torch.tensor(
    sample_x,
    dtype=torch.float32
).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(sample_tensor)

prediction = prediction.cpu().numpy()[0]
print(
    f"Predicted future position : "
    f"({prediction[0]:.2f},{prediction[1]:.2f})"
)

# Prepare Trajectory

history_x = sample_x[:, 0]
history_y = sample_x[:, 1]

actual_x = actual_position[0]
actual_y = actual_position[1]

predicted_x = prediction[0]
predicted_y = prediction[1]

# calulate error

error = np.sqrt(
    (actual_x - predicted_x)**2 +
    (actual_y - predicted_y)**2
)

print(f"Prediction error : {error:.2f}pixels")

# Create Plot
plt.figure(figsize=(10,7))
# Past trajectory
plt.plot(history_x,history_y,marker = "o",label='Observed trajectory')

# Actual future position
plt.scatter(actual_x,actual_y,s=120,marker="X",label="Actual future position")

# Predicted future position
plt.scatter(
    predicted_x,
    predicted_y,
    s=120,
    marker = "*",
    label = "Predicted future position"
)

# Connect last observed point to prediction
plt.plot(
    [history_x[-1],predicted_x],
    [history_y[-1], predicted_y],
    linestyle="--",
    label = "Predicted future position"
)


plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")

plt.title("LSTM Trajectory Prediction - Version 1")
plt.legend()
plt.grid(True)

# Important for image coordinates
# y increases downwards in images.

plt.gca().invert_yaxis()
plt.tight_layout()

# Save figures
output_path = os.path.join(output_dir,"trajectory_visualization.png")
plt.savefig(output_path, dpi=150)
plt.show()

# Complete
print("\n"+"="*60)
print("Visualization completed")
print("="*60)

print(f"Plot saved to:")
print(output_path)
print("="*60)