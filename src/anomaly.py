
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.trajectory_lstm_arch import TrajectoryLSTM

BASE_DIR = r"C:\Users\HP\Desktop\multi_object_tracking"


x_test_path = os.path.join(
    BASE_DIR,
    "data",
    "sequences",
    "x_test.npy"
)

y_test_path = os.path.join(
    BASE_DIR,
    "data",
    "sequences",
    "y_test.npy"
)

x_val_path = os.path.join(BASE_DIR,"data","sequences","x_val.npy")
y_val_path = os.path.join(BASE_DIR,'data','sequences',"y_val.npy")


metadata_path = os.path.join(
    BASE_DIR,
    "data",
    "sequences",
    "test_metadata.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "trajectory_lstm.pth"
)


CONSECUTIVE_ANOMALIES = 3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

x_test = np.load(x_test_path)

y_test = np.load(y_test_path)

x_val = np.load(x_val_path)
y_val = np.load(y_val_path)


metadata = np.load(
    metadata_path,
    allow_pickle=True
)

print(
    f"X shape: "
    f"{x_test.shape}"
)

print(
    f"Y shape: "
    f"{y_test.shape}"
)

print(
    f"Metadata shape: "
    f"{metadata.shape}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = model.to(device)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# PREDICTION
# ============================================================

x_tensor = torch.tensor(
    x_val,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    validation_predictions = model(
        x_tensor
    )


validation_predictions = validation_predictions.cpu().numpy()


print(
    f"validation_Prediction shape: "
    f"{validation_predictions.shape}"
)


# ============================================================
# CALCULATE PREDICTION ERROR
# ============================================================

validation_errors = np.sqrt(
    np.sum(
        (validation_predictions- y_val) ** 2,
        axis=1
    )
)

# calulating the threshold from the validation
ANOMALY_THRESHOLD = np.percentile(validation_errors,95)

# ============================================================
# RAW ANOMALY CLASSIFICATION
# ============================================================

print(
    f"Threshold: "
    f"{ANOMALY_THRESHOLD:.4f} normalized units"
)


x_tensor = torch.tensor(x_test,dtype = torch.float32).to(device)
with torch.no_grad():
    test_predictions = model(x_tensor)
test_predictions = test_predictions.cpu().numpy()
print("test_predictions",test_predictions.shape)
errors = np.sqrt(np.sum((test_predictions-y_test)**2,axis = 1))

raw_labels = np.where(
    errors > ANOMALY_THRESHOLD,
    "ANOMALY",
    "NORMAL"
)


# ============================================================
# CREATE SMOOTHED LABEL ARRAY
# ============================================================

smoothed_labels = np.full(
    len(errors),
    "NORMAL",
    dtype=object
)


# ============================================================
# GROUP SAMPLES BY SEQUENCE + TRACK ID
# ============================================================

track_groups = {}


for index in range(len(metadata)):

    sequence = str(
        metadata[index][0]
    )

    track_id = int(
        metadata[index][1]
    )

    key = (
        sequence,
        track_id
    )

    if key not in track_groups:

        track_groups[key] = []

    track_groups[key].append(
        index
    )


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

for key, indexes in track_groups.items():

    indexes.sort(
        key=lambda i:
        int(metadata[i][2])
    )

    anomaly_streak = 0

    previous_frame = None


    for index in indexes:

        current_frame = int(
            metadata[index][2]
        )


        # Check whether frames are consecutive

        if (
            previous_frame is None
            or current_frame != previous_frame + 1
        ):

            anomaly_streak = 0


        if raw_labels[index] == "ANOMALY":

            anomaly_streak += 1

        else:

            anomaly_streak = 0


        if (
            anomaly_streak
            >= CONSECUTIVE_ANOMALIES
        ):

            smoothed_labels[index] = "ANOMALY"

        else:

            smoothed_labels[index] = "NORMAL"


        previous_frame = current_frame


# ============================================================
# TEMPORAL SMOOTHING INFORMATION
# ============================================================

print(
    f"Temporal smoothing: "
    f"{CONSECUTIVE_ANOMALIES} consecutive frames"
)


# ============================================================
# TRACK-LEVEL ANOMALY REPORT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "TRACK-LEVEL ANOMALY REPORT"
)

print(
    "=" * 80
)


track_report = []


for key, indexes in track_groups.items():

    sequence, track_id = key


    indexes.sort(
        key=lambda i:
        int(metadata[i][2])
    )


    total_frames = len(
        indexes
    )


    # --------------------------------------------------------
    # RAW ANOMALY COUNT
    # --------------------------------------------------------

    raw_anomalies = np.sum(
        raw_labels[indexes]
        == "ANOMALY"
    )


    # --------------------------------------------------------
    # SMOOTHED ANOMALY COUNT
    # --------------------------------------------------------

    smoothed_anomalies = np.sum(
        smoothed_labels[indexes]
        == "ANOMALY"
    )


    # --------------------------------------------------------
    # LONGEST SMOOTHED ANOMALY STREAK
    # --------------------------------------------------------

    longest_streak = 0

    current_streak = 0

    previous_frame = None


    for index in indexes:

        current_frame = int(
            metadata[index][2]
        )


        # Reset streak if frames are not consecutive

        if (
            previous_frame is None
            or current_frame != previous_frame + 1
        ):

            current_streak = 0


        if smoothed_labels[index] == "ANOMALY":

            current_streak += 1

            longest_streak = max(
                longest_streak,
                current_streak
            )

        else:

            current_streak = 0


        previous_frame = current_frame


    # --------------------------------------------------------
    # FINAL TRACK STATUS
    # --------------------------------------------------------

    if smoothed_anomalies > 0:

        final_status = "ANOMALY"

    else:

        final_status = "NORMAL"


    # --------------------------------------------------------
    # STORE REPORT
    # --------------------------------------------------------

    track_report.append([
        sequence,
        track_id,
        total_frames,
        raw_anomalies,
        smoothed_anomalies,
        longest_streak,
        final_status
    ])


# ============================================================
# PRINT TRACK REPORT
# ============================================================

for report in track_report:

    (
        sequence,
        track_id,
        total_frames,
        raw_anomalies,
        smoothed_anomalies,
        longest_streak,
        final_status
    ) = report


    print(
        f"\nSequence      : "
        f"{sequence}"
    )

    print(
        f"Track ID      : "
        f"{track_id}"
    )

    print(
        f"Total frames  : "
        f"{total_frames}"
    )

    print(
        f"Raw anomalies : "
        f"{raw_anomalies}"
    )

    print(
        f"Smoothed      : "
        f"{smoothed_anomalies}"
    )

    print(
        f"Longest streak: "
        f"{longest_streak}"
    )

    print(
        f"Final status  : "
        f"{final_status}"
    )


# ============================================================
# GLOBAL FINAL CLASSIFICATION
# ============================================================

normal_count = np.sum(
    smoothed_labels == "NORMAL"
)

anomaly_count = np.sum(
    smoothed_labels == "ANOMALY"
)


normal_percentage = (
    normal_count / len(errors)
) * 100


anomaly_percentage = (
    anomaly_count / len(errors)
) * 100


print(
    "\n" + "=" * 80
)

print(
    "FINAL GLOBAL ANOMALY SUMMARY"
)

print(
    "=" * 80
)


print(
    f"Normal samples  : "
    f"{normal_count}"
)

print(
    f"Anomaly samples : "
    f"{anomaly_count}"
)


print(
    f"Normal percentage  : "
    f"{normal_percentage:.2f}%"
)

print(
    f"Anomaly percentage : "
    f"{anomaly_percentage:.2f}%"
)


# ============================================================
# ERROR STATISTICS
# ============================================================

print(
    f"\nMinimum error : "
    f"{errors.min():.4f} normalized_unit"
)

print(
    f"Maximum error : "
    f"{errors.max():.4f} normalized_unit"
)

print(
    f"Mean error    : "
    f"{errors.mean():.4f} normalized_unit"
)

print(
    f"Median error  : "
    f"{np.median(errors):.4f} normalized_unit"
)


# ============================================================
# FIRST 10 PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SAMPLE PREDICTIONS"
)

print(
    "=" * 80
)


for i in range(
    min(10, len(test_predictions))
):

    actual_x, actual_y = y_test[i]

    predicted_x, predicted_y = test_predictions[i]


    sequence = metadata[i][0]

    track_id = metadata[i][1]

    frame = metadata[i][2]


    print(
        f"Sample {i + 1}: "
        f"Actual = "
        f"({actual_x:.2f}, {actual_y:.2f}) | "
        f"Predicted = "
        f"({predicted_x:.2f}, {predicted_y:.2f}) | "
        f"Error = {errors[i]:.2f} | "
        f"Status = {raw_labels[i]} | "
        f"Final = {smoothed_labels[i]}"
    )


# ============================================================
# ERROR PERCENTILES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "ERROR PERCENTILES"
)

print(
    "=" * 80
)


print(
    f"90th percentile : "
    f"{np.percentile(errors, 90):.4f}"
)

print(
    f"95th percentile : "
    f"{np.percentile(errors, 95):.4f}"
)

print(
    f"99th percentile : "
    f"{np.percentile(errors, 99):.4f}"
)

plt.figure(figsize=(10,6))
plt.hist(errors,bins = 50)
plt.axvline(ANOMALY_THRESHOLD,linestyle="--",linewidth=2,label=f"Threshold = { ANOMALY_THRESHOLD:.4f}")
plt.xlabel("Prediction Error")
plt.ylabel("Number of Samples")
plt.title("LSTM Trajectory Prediction Error Distribution")
plt.legend()
plt.grid(alpha = 0.3)
plt.show()

# ============================================================
# ERROR OVER TIME FOR ONE TRACK
# ============================================================

selected_sequence = metadata[0][0]
selected_track_id = 4

selected_indexes = [
    i
    for i in range(len(metadata))
    if (
        metadata[i][0] == selected_sequence
        and int(metadata[i][1]) == int(selected_track_id)
    )
]

selected_indexes.sort(
    key=lambda i: int(metadata[i][2])
)

frames = [
    int(metadata[i][2])
    for i in selected_indexes
]

track_errors = [
    errors[i]
    for i in selected_indexes
]

plt.figure(figsize=(12, 6))

plt.plot(
    frames,
    track_errors,
    linewidth=2,
    label="Prediction Error"
)

plt.axhline(
    ANOMALY_THRESHOLD,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {ANOMALY_THRESHOLD:.4f}"
)

plt.xlabel("Frame")
plt.ylabel("Prediction Error")
plt.title(
    f"Prediction Error Over Time | "
    f"{selected_sequence} | Track {selected_track_id}"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()

# ============================================================
# ACTUAL VS PREDICTED TRAJECTORY
# ============================================================

actual_x = []
actual_y = []

predicted_x = []
predicted_y = []

anomaly_x = []
anomaly_y = []

for i in selected_indexes:

    actual_x.append(y_test[i][0])
    actual_y.append(y_test[i][1])

    predicted_x.append(test_predictions[i][0])
    predicted_y.append(test_predictions[i][1])

    if smoothed_labels[i] == "ANOMALY":

        anomaly_x.append(test_predictions[i][0])
        anomaly_y.append(test_predictions[i][1])


plt.figure(figsize=(10, 8))

# Actual trajectory
plt.plot(
    actual_x,
    actual_y,
    linewidth=2,
    label="Actual Trajectory"
)

# Predicted trajectory
plt.plot(
    predicted_x,
    predicted_y,
    linestyle="--",
    linewidth=2,
    label="Predicted Trajectory"
)

# Anomaly points
if len(anomaly_x) > 0:

    plt.scatter(
        anomaly_x,
        anomaly_y,
        s=100,
        marker="x",
        linewidths=3,
        label="Anomaly Prediction"
    )

# Start point
plt.scatter(
    actual_x[0],
    actual_y[0],
    s=80,
    label="Start"
)

# End point
plt.scatter(
    actual_x[-1],
    actual_y[-1],
    s=80,
    label="End"
)

plt.xlabel("Normalized X")
plt.ylabel("Normalized Y")

plt.title(
    f"Actual vs Predicted Trajectory | "
    f"{selected_sequence} | Track {selected_track_id}"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()