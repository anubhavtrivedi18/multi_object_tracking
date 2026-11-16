import os
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# STEP 9.6
# COMPLETE REAL-TIME INFERENCE PIPELINE
# ============================================================


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

YOLO_MODEL_PATH = "yolo11n.pt"

LSTM_MODEL_PATH = (
    r"C:\Users\HP\Desktop\multi_object_tracking"
    r"\models\trajectory_lstm.pth"
)

SOURCE = (
    r"C:\Users\HP\Desktop\multi_object_tracking"
    r"\data\MOT17\mot17\train\MOT17-02-FRCNN\img1"
)

OUTPUT_DIR = (
    r"C:\Users\HP\Desktop\multi_object_tracking"
    r"\data\inference_results"
)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "step9_6_predictions.csv"
)


# ------------------------------------------------------------
# 2. PARAMETERS
# ------------------------------------------------------------

DEVICE = "cuda:0"

HISTORY_LENGTH = 10

# Temporary Version-1 threshold.
# This threshold is NOT scientifically optimized yet.
ANOMALY_THRESHOLD = 1200.0


# ------------------------------------------------------------
# 3. CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 4. CHECK CUDA
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Please check your PyTorch CUDA installation."
    )

print("=" * 60)
print("STEP 9.6 - COMPLETE INFERENCE PIPELINE")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print("Device:", DEVICE)


# ------------------------------------------------------------
# 5. DEFINE THE SAME LSTM ARCHITECTURE
# ------------------------------------------------------------

class TrajectoryLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_input,
        num_layers,
        output_size
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_input,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_input,
            output_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


# ------------------------------------------------------------
# 6. LOAD YOLO
# ------------------------------------------------------------

print("\nLoading YOLO model...")

yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLO model loaded successfully.")


# ------------------------------------------------------------
# 7. LOAD LSTM
# ------------------------------------------------------------

print("\nLoading LSTM model...")

lstm_model = torch.load(
    LSTM_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

lstm_model.to(DEVICE)

lstm_model.eval()

print("LSTM model loaded successfully.")


# ------------------------------------------------------------
# 8. TRACK HISTORY
# ------------------------------------------------------------

track_history = {}


# ------------------------------------------------------------
# 9. PREVIOUS PREDICTIONS
# ------------------------------------------------------------

previous_predictions = {}


# ------------------------------------------------------------
# 10. STATISTICS
# ------------------------------------------------------------

frame_count = 0

prediction_comparisons = 0

normal_count = 0

anomaly_count = 0

prediction_errors = []

unique_track_ids = set()


# ------------------------------------------------------------
# 11. START TIMER
# ------------------------------------------------------------

start_time = time.time()


# ------------------------------------------------------------
# 12. OPEN CSV FILE
# ------------------------------------------------------------

with open(
    RESULTS_CSV,
    mode="w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "frame",
        "track_id",
        "actual_x",
        "actual_y",
        "predicted_x",
        "predicted_y",
        "prediction_error",
        "status"
    ])


    # --------------------------------------------------------
    # 13. RUN YOLO + BYTE TRACK
    # --------------------------------------------------------

    results = yolo_model.track(
        source=SOURCE,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        device=DEVICE
    )


    # --------------------------------------------------------
    # 14. PROCESS EACH FRAME
    # --------------------------------------------------------

    for result in results:

        frame_count += 1

        frame = result.orig_img.copy()

        print(f"\nProcessing frame: {frame_count}")


        # ----------------------------------------------------
        # 15. CHECK TRACK IDs
        # ----------------------------------------------------

        if result.boxes.id is None:

            cv2.imshow(
                "Step 9.6 - Complete Inference",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue


        # ----------------------------------------------------
        # 16. GET TRACK INFORMATION
        # ----------------------------------------------------

        boxes = result.boxes

        track_ids = boxes.id.cpu().numpy().astype(int)

        xyxy = boxes.xyxy.cpu().numpy()

        confidences = boxes.conf.cpu().numpy()

        classes = boxes.cls.cpu().numpy().astype(int)


        current_frame_predictions = []


        # ----------------------------------------------------
        # 17. PROCESS EVERY TRACK
        # ----------------------------------------------------

        for box, track_id, confidence, class_id in zip(
            xyxy,
            track_ids,
            confidences,
            classes
        ):

            # ------------------------------------------------
            # Only track persons
            # COCO class 0 = person
            # ------------------------------------------------

            if class_id != 0:
                continue


            unique_track_ids.add(track_id)


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = box


            # ------------------------------------------------
            # Calculate center
            # ------------------------------------------------

            center_x = (x1 + x2) / 2

            center_y = (y1 + y2) / 2


            # ------------------------------------------------
            # Create history for new ID
            # ------------------------------------------------

            if track_id not in track_history:

                track_history[track_id] = []


            # ------------------------------------------------
            # Add current position
            # ------------------------------------------------

            track_history[track_id].append(
                (center_x, center_y)
            )


            # ------------------------------------------------
            # Keep only latest 10 positions
            # ------------------------------------------------

            if len(track_history[track_id]) > HISTORY_LENGTH:

                track_history[track_id].pop(0)


            history_length = len(
                track_history[track_id]
            )


            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW TRACK ID
            # ------------------------------------------------

            label = (
                f"ID: {track_id} "
                f"History: {history_length}/10"
            )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW CURRENT CENTER
            # ------------------------------------------------

            cv2.circle(
                frame,
                (int(center_x), int(center_y)),
                4,
                (255, 0, 0),
                -1
            )


            # ------------------------------------------------
            # DRAW TRAJECTORY
            # ------------------------------------------------

            history = track_history[track_id]

            for i in range(1, len(history)):

                pt1 = (
                    int(history[i - 1][0]),
                    int(history[i - 1][1])
                )

                pt2 = (
                    int(history[i][0]),
                    int(history[i][1])
                )

                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    (255, 0, 0),
                    2
                )


            # ------------------------------------------------
            # 18. LSTM PREDICTION
            # ------------------------------------------------

            if history_length == HISTORY_LENGTH:

                trajectory = np.array(
                    history,
                    dtype=np.float32
                )


                # --------------------------------------------
                # Convert to PyTorch tensor
                # --------------------------------------------

                input_tensor = torch.tensor(
                    trajectory,
                    dtype=torch.float32
                )


                # --------------------------------------------
                # Add batch dimension
                #
                # (10,2) -> (1,10,2)
                # --------------------------------------------

                input_tensor = input_tensor.unsqueeze(0)


                input_tensor = input_tensor.to(
                    DEVICE
                )


                # --------------------------------------------
                # LSTM prediction
                # --------------------------------------------

                with torch.no_grad():

                    prediction = lstm_model(
                        input_tensor
                    )


                prediction = (
                    prediction
                    .cpu()
                    .numpy()[0]
                )


                predicted_x = float(
                    prediction[0]
                )

                predicted_y = float(
                    prediction[1]
                )


                # ------------------------------------------------
                # 19. CHECK PREVIOUS PREDICTION
                # ------------------------------------------------

                if track_id in previous_predictions:

                    previous_prediction = (
                        previous_predictions[track_id]
                    )

                    previous_x = previous_prediction[0]

                    previous_y = previous_prediction[1]


                    # --------------------------------------------
                    # Actual current position
                    # compared with prediction from previous frame
                    # --------------------------------------------

                    error = np.sqrt(
                        (center_x - previous_x) ** 2
                        +
                        (center_y - previous_y) ** 2
                    )


                    prediction_errors.append(
                        error
                    )

                    prediction_comparisons += 1


                    # --------------------------------------------
                    # Anomaly detection
                    # --------------------------------------------

                    if error >= ANOMALY_THRESHOLD:

                        status = "ANOMALY"

                        anomaly_count += 1

                    else:

                        status = "NORMAL"

                        normal_count += 1


                    # --------------------------------------------
                    # Save result to CSV
                    # --------------------------------------------

                    writer.writerow([
                        frame_count,
                        track_id,
                        round(center_x, 3),
                        round(center_y, 3),
                        round(previous_x, 3),
                        round(previous_y, 3),
                        round(error, 3),
                        status
                    ])


                    # --------------------------------------------
                    # Display prediction/error
                    # --------------------------------------------

                    cv2.putText(
                        frame,
                        f"Error: {error:.1f}px",
                        (
                            int(x1),
                            int(y2) + 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )


                    cv2.putText(
                        frame,
                        status,
                        (
                            int(x1),
                            int(y2) + 40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (
                            (0, 0, 255)
                            if status == "ANOMALY"
                            else
                            (0, 255, 0)
                        ),
                        2
                    )


                # ------------------------------------------------
                # 20. STORE CURRENT PREDICTION
                # ------------------------------------------------

                previous_predictions[track_id] = (
                    predicted_x,
                    predicted_y
                )


                # ------------------------------------------------
                # 21. DRAW PREDICTED POSITION
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    (
                        int(predicted_x),
                        int(predicted_y)
                    ),
                    7,
                    (0, 255, 255),
                    -1
                )


                # ------------------------------------------------
                # Draw line from last position to prediction
                # ------------------------------------------------

                cv2.line(
                    frame,
                    (
                        int(center_x),
                        int(center_y)
                    ),
                    (
                        int(predicted_x),
                        int(predicted_y)
                    ),
                    (0, 255, 255),
                    2
                )


        # ----------------------------------------------------
        # 22. DISPLAY FRAME
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Tracked IDs: {len(unique_track_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Anomalies: {anomaly_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        cv2.imshow(
            "Step 9.6 - Complete Inference",
            frame
        )


        # ----------------------------------------------------
        # 23. PRESS Q TO STOP
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nStopped by user.")

            break


# ------------------------------------------------------------
# 24. CLOSE WINDOWS
# ------------------------------------------------------------

cv2.destroyAllWindows()


# ------------------------------------------------------------
# 25. CALCULATE TOTAL TIME
# ------------------------------------------------------------

total_time = time.time() - start_time


# ------------------------------------------------------------
# 26. FINAL STATISTICS
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("STEP 9.6 COMPLETED")
print("=" * 60)

print(f"Total frames processed : {frame_count}")

print(
    f"Total tracked IDs      : "
    f"{len(unique_track_ids)}"
)

print(
    f"Prediction comparisons : "
    f"{prediction_comparisons}"
)

print(
    f"Normal predictions     : "
    f"{normal_count}"
)

print(
    f"Anomaly predictions    : "
    f"{anomaly_count}"
)


# ------------------------------------------------------------
# 27. ERROR STATISTICS
# ------------------------------------------------------------

if len(prediction_errors) > 0:

    errors = np.array(
        prediction_errors
    )

    print("\nPrediction Error Statistics:")

    print(
        f"Minimum Error  : "
        f"{errors.min():.2f} pixels"
    )

    print(
        f"Maximum Error  : "
        f"{errors.max():.2f} pixels"
    )

    print(
        f"Mean Error     : "
        f"{errors.mean():.2f} pixels"
    )

    print(
        f"Median Error   : "
        f"{np.median(errors):.2f} pixels"
    )

    print(
        f"95th Percentile: "
        f"{np.percentile(errors, 95):.2f} pixels"
    )


# ------------------------------------------------------------
# 28. FPS
# ------------------------------------------------------------

if total_time > 0:

    fps = frame_count / total_time

    print(
        f"\nAverage Processing FPS: "
        f"{fps:.2f}"
    )


# ------------------------------------------------------------
# 29. OUTPUT FILE
# ------------------------------------------------------------

print("\nResults saved to:")

print(RESULTS_CSV)

print("=" * 60)