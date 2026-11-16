# Real-Time Multi-Object Tracking & Trajectory-Based Anomaly Detection

A real-time computer vision system that tracks people across video frames, learns their movement patterns using recurrent neural networks, predicts future trajectories, and identifies unusual movement based on trajectory prediction error.

The project combines **YOLO, ByteTrack, LSTM/GRU sequence modeling, trajectory analysis, anomaly detection, FastAPI, real-time video processing, and MLflow experiment tracking** into one end-to-end ML system.

---

## Project Overview

The goal of this project is to answer:

> **Can we learn how a person normally moves and identify when their movement significantly deviates from the learned trajectory?**

Instead of classifying an image independently, this project models **movement over time**.

For every tracked person:

1. Detect the person.
2. Assign a persistent tracking ID.
3. Store their movement history.
4. Extract position and velocity features.
5. Feed a sequence of past movements into an LSTM.
6. Predict the person's next position.
7. Compare the predicted position with the actual position.
8. Calculate trajectory prediction error.
9. Detect abnormal movement when the error exceeds a learned threshold.
10. Apply temporal smoothing to reduce isolated false alarms.
11. Display the result through a FastAPI web application.

---

# System Architecture

```text
                         MOT17 Dataset
                              │
                              ▼
                    Person Detection
                         YOLOv8n
                              │
                              ▼
                       ByteTrack
                              │
                              ▼
                     Persistent Track IDs
                              │
                              ▼
                    Trajectory Extraction
                              │
               ┌──────────────┴──────────────┐
               │                             │
        Position Features              Velocity Features
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    Sequence Preparation
                         10 Frames
                              │
                              ▼
                        LSTM Model
                              │
                              ▼
                  Future Position Prediction
                              │
                              ▼
                  Prediction Error / ADE / FDE
                              │
                              ▼
                    Anomaly Threshold
                              │
                              ▼
                    Temporal Smoothing
                              │
                              ▼
                     NORMAL / ANOMALY
                              │
                              ▼
                         FastAPI
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Recorded Video              Live Camera
                 │                         │
                 └────────────┬────────────┘
                              ▼
                         Web Dashboard
```

---

# Key Technologies

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLOv8n |
| Multi-Object Tracking | ByteTrack |
| Dataset | MOT17 |
| Sequence Model | LSTM |
| Experimental Model | GRU |
| Deep Learning Framework | PyTorch |
| Anomaly Detection | Trajectory Prediction Error |
| API | FastAPI |
| Video Processing | OpenCV |
| Experiment Tracking | MLflow |
| Testing | Pytest |

---

# Dataset

This project uses the **MOT17 multi-object tracking benchmark**.

MOT17 contains real-world pedestrian sequences designed for multi-object tracking research.

The dataset is used to learn and evaluate:

- person detection
- identity tracking
- trajectory generation
- temporal movement patterns

The project processes multiple MOT17 sequences rather than relying on a single video.

---

# Version 1 → Version 2 Improvements

A major focus of this project was iterative improvement rather than building a single model and stopping.

## Version 1

The initial pipeline used:

```text
MOT17
  ↓
YOLO
  ↓
ByteTrack
  ↓
Trajectory History
  ↓
LSTM
  ↓
Prediction Error
  ↓
Anomaly Detection
```

The first model revealed an important problem:

- predictions were close to a nearly constant point
- trajectory prediction error was very high
- MAE ≈ 706 pixels
- RMSE ≈ 814 pixels
- ADE/FDE ≈ 1069 pixels

This showed that simply feeding raw pixel coordinates into the model was not sufficient.

---

# Version 2 Improvements

Version 2 introduced several changes to make the pipeline more robust and more suitable for an ML engineering workflow.

## 1. Multiple MOT17 Sequences

Instead of training on a single sequence, the pipeline processes multiple MOT17 sequences.

Example:

```text
MOT17-02-FRCNN
MOT17-04-FRCNN
MOT17-05-FRCNN
MOT17-09-FRCNN
...
```

This increases the diversity of movement patterns available during training.

---

## 2. Coordinate Normalization

Raw pixel coordinates depend heavily on video resolution.

Instead of directly learning:

```text
x = 850
y = 430
```

coordinates are normalized:

```python
normalized_x = center_x / frame_width
normalized_y = center_y / frame_height
```

This makes the model less dependent on the resolution of a particular video.

The original pixel coordinates are also preserved for visualization and analysis.

---

## 3. Velocity Features

Version 2 adds movement information instead of using position alone.

For each tracked person:

```text
normalized_x
normalized_y
velocity_x
velocity_y
```

are used as model inputs.

Velocity is calculated from the change in normalized position between consecutive frames:

```text
velocity_x = current_x - previous_x
velocity_y = current_y - previous_y
```

This gives the model information about both:

- where the person is
- how the person is moving

---

# Sequence Modeling

The model receives a sequence of the previous **10 frames**.

Each timestep contains four features:

```text
[x, y, velocity_x, velocity_y]
```

Therefore, one input sequence has the shape:

```text
10 × 4
```

The model predicts the person's future:

```text
[x_future, y_future]
```

---

# LSTM Architecture

The final trajectory prediction model uses:

```text
Input
  │
  ▼
LSTM
Input Size = 4
Hidden Size = 64
Layers = 2
  │
  ▼
Last LSTM Output
  │
  ▼
Fully Connected Layer
64 → 2
  │
  ▼
Predicted X, Y
```

Implementation:

```python
LSTM(
    input_size=4,
    hidden_size=64,
    num_layers=2,
    batch_first=True
)

Linear(
    64,
    2
)
```

The model predicts the next normalized position of the tracked person.

---

# LSTM vs GRU Experiment

The project also includes a GRU implementation as an experimental baseline.

The repository contains:

```text
models/
├── trajectory_lstm.pth
├── trajectory_lstm_arch.py
├── trajectory_gru.pth
└── trajectory_gru_arch.py
```

and:

```text
src/
├── train_lstm.py
├── train_gru.py
├── evaluation.py
└── evaluation_gru.py
```

This allows the project to compare different recurrent architectures instead of assuming that one architecture is automatically better.

After experimentation, **LSTM was selected as the final trajectory prediction model**.

---

# Data Split

To reduce leakage between training and testing, the project performs a sequence-level split:

```text
70% → Training
15% → Validation
15% → Testing
```

The split is performed at the sequence level rather than randomly mixing individual trajectory samples.

This helps preserve the temporal structure of the tracking data.

---



# Model Evaluation

The trajectory prediction model is evaluated using:

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual positions.

### RMSE

Root Mean Squared Error gives more weight to larger prediction errors.

### ADE

Average Displacement Error measures average trajectory prediction distance.

### FDE

Final Displacement Error measures the distance between the predicted and actual future position.

---

# Anomaly Detection

The project does not classify anomalies directly from individual images.

Instead, it uses **trajectory prediction error**.

For each prediction:

```text
Actual Position
      │
      │
      ▼
Prediction Error
      ▲
      │
Predicted Position
```

The Euclidean error is calculated as:

```python
error = sqrt(
    (predicted_x - actual_x)^2 +
    (predicted_y - actual_y)^2
)
```

Because the model operates on normalized coordinates, the anomaly threshold is also calculated in normalized coordinate space.

---

# Automatic Threshold Selection

Instead of manually selecting an arbitrary threshold, the validation set is used.

The threshold is currently selected as the:

```text
95th percentile
```

of validation prediction errors.

Conceptually:

```text
Validation Errors
       │
       ▼
95th Percentile
       │
       ▼
Anomaly Threshold
```

This allows the threshold to be derived from the model's normal validation behavior.

Current validation statistics:

```text
Minimum Error : 0.0001
Median Error  : 0.0034
90th Percentile: 0.0093
95th Percentile: 0.0119
99th Percentile: 0.0204
Maximum Error : 0.1287
```

---

# Temporal Smoothing

A single anomalous prediction should not immediately cause a final anomaly alert.

Therefore, the system applies temporal smoothing.

The current logic requires:

```text
3 consecutive anomalous predictions
```

for the same tracked person before producing a smoothed anomaly event.

This helps reduce:

- isolated prediction errors
- detection noise
- temporary tracking fluctuations
- false alarms

---

# Version 2 Anomaly Results

On the current evaluation data:

```text
Total samples      : 4612
Normal samples     : 4549
Anomaly samples    : 63

Normal percentage  : 98.63%
Anomaly percentage : 1.37%
```

These numbers represent **trajectory-deviation-based anomaly labels generated by the system**, not ground-truth anomaly labels.

Therefore, they should not be interpreted as real-world anomaly classification accuracy.

---

# Real-Time Inference

The trained model is integrated into a real-time inference pipeline.

For every incoming frame:

```text
Frame
  ↓
YOLO Person Detection
  ↓
ByteTrack
  ↓
Track IDs
  ↓
Position + Velocity
  ↓
Last 10 Frames
  ↓
LSTM
  ↓
Future Position
  ↓
Prediction Error
  ↓
NORMAL / ANOMALY
```

The same processing function is used for both:

- uploaded videos
- live camera input

This avoids maintaining two completely different inference pipelines.

---

# FastAPI Application

FastAPI provides the backend API.

Main endpoints:

```text
GET  /
GET  /health
POST /process-video
POST /process-frame
```

### `/`

Serves the web interface.

### `/health`

Checks whether the API is running.

### `/process-video`

Processes recorded video frames.

### `/process-frame`

Processes individual frames from the live camera.

---

# Real-Time Web Dashboard

The frontend provides:

- recorded video processing
- live camera mode
- person bounding boxes
- persistent tracking IDs
- previous movement trajectory
- predicted next position
- prediction error
- NORMAL / ANOMALY status
- anomaly threshold information

Visualization:

```text
Past Movement
     │
     ▼
●──●──●──●──●
             \
              \
             Predicted Position
                    ●
```

---

# MLflow Experiment Tracking

MLflow is used to track model training experiments.

Experiment:

```text
MOT17-Trajectory-LSTM
```

The training pipeline logs:

### Parameters

```text
sequence_length
input_size
hidden_size
num_layers
output_size
batch_size
epochs
learning_rate
device
GPU
```

### Metrics

```text
train_loss
val_loss
best_val_loss
```

### Artifacts

The trained LSTM model is logged as an MLflow artifact.

This makes it possible to compare experiments and maintain a record of how the model evolved.

---

# Example MLflow Workflow

```text
Train Model
     │
     ▼
MLflow Run
     │
     ├── Parameters
     │
     ├── Training Loss
     │
     ├── Validation Loss
     │
     ├── Best Validation Loss
     │
     └── Model Artifact
```

This turns the project from a simple deep learning experiment into a more structured ML engineering workflow.

---

# Project Structure

```text
multi_object_tracking/
│
├── app.py
├── demo.py
├── template.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── yolov8n.pt
│
├── models/
│   ├── trajectory_lstm.pth
│   ├── trajectory_lstm_arch.py
│   ├── trajectory_gru.pth
│   └── trajectory_gru_arch.py
│
├── src/
│   ├── anomaly.py
│   ├── evaluation.py
│   ├── evaluation_gru.py
│   ├── inference.py
│   ├── prepare_sequences.py
│   ├── tracker.py
│   ├── train_gru.py
│   ├── train_lstm.py
│   ├── trajectory.py
│   └── visualize.py
│
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── tests/
    └── test_app.py
```

Local experiment-generated directories such as MLflow tracking data, cached files, training outputs, and Python cache files are intentionally excluded from version control.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/anubhavtrivedi18/multi_object_tracking.git
cd multi_object_tracking
```

Create a virtual environment:

```bash
conda create -n torch_test python=3.11
conda activate torch_test
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the FastAPI Application

Start the server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Run MLflow

Start the local MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

The training script logs experiments to:

```text
MOT17-Trajectory-LSTM
```

---

# Training

The LSTM training pipeline can be executed using:

```bash
python src/train_lstm.py
```

The training pipeline uses:

```text
MOT17 tracking data
       ↓
Normalized trajectories
       ↓
Velocity features
       ↓
10-frame sequences
       ↓
Train / Validation / Test split
       ↓
LSTM training
       ↓
Validation monitoring
       ↓
MLflow logging
       ↓
Model artifact
```

---

# Testing

The project includes API tests using Pytest.

Run:

```bash
pytest tests -v
```

The tests verify important FastAPI endpoints such as:

```text
/
 /health
```

---

# Engineering Practices Used

This project focuses not only on model development but also on engineering practices.

### Modular code

Different responsibilities are separated into:

```text
tracker.py
prepare_sequences.py
train_lstm.py
evaluation.py
anomaly.py
inference.py
```


### Experiment tracking

MLflow records training parameters and metrics.

### API development

FastAPI exposes the computer vision pipeline through HTTP endpoints.

### Automated testing

Pytest is used to test the API.

### Version control

Git and GitHub are used to manage project development.

### Real-time processing

The same inference pipeline supports both recorded video and live camera input.

---

# Challenges and Improvements

## Challenge 1: Poor initial trajectory predictions

Version 1 produced very large prediction errors.

The model tended to predict almost constant positions.

### Improvement

Version 2 introduced:

- multiple MOT17 sequences
- coordinate normalization
- velocity features
- sequence-level train/validation/test splitting

This significantly improved trajectory prediction.

---

## Challenge 2: Resolution dependency

Raw pixel coordinates caused the model to depend strongly on the resolution of the training videos.

### Improvement

Coordinates were normalized using the actual frame dimensions:

```text
x / frame_width
y / frame_height
```

---

## Challenge 3: Position-only representation

Position alone does not fully describe movement.

### Improvement

Velocity was added:

```text
x
y
velocity_x
velocity_y
```

This gives the recurrent model information about movement direction and speed.

---

## Challenge 4: Isolated anomaly predictions

Individual prediction errors can be noisy.

### Improvement

Temporal smoothing was introduced so that an anomaly requires multiple consecutive anomalous predictions.

---

## Challenge 5: Manual anomaly threshold

A manually selected threshold is difficult to justify.

### Improvement

The threshold is derived from validation prediction errors using the 95th percentile.

---

# What This Project Demonstrates

This project demonstrates practical experience with:

```text
Computer Vision
      +
Deep Learning
      +
Time-Series Modeling
      +
Object Tracking
      +
Anomaly Detection
      +
API Development
      +
Experiment Tracking
      +
Testing
      +
Software Engineering
```

More specifically, it demonstrates the ability to:

- build an end-to-end ML pipeline
- work with real-world video data
- implement multi-object tracking
- construct trajectory datasets
- engineer temporal features
- train recurrent neural networks
- compare model architectures
- evaluate forecasting models
- design anomaly detection logic
- build a real-time inference system
- expose ML models through APIs
- track experiments using MLflow
- write automated tests
- manage the project using Git/GitHub

---

# Limitations

The current system has several limitations.

### 1. No ground-truth anomaly labels

MOT17 is primarily a multi-object tracking benchmark and does not provide labels specifically for abnormal human behavior.

Therefore, anomaly detection currently uses trajectory prediction error rather than supervised anomaly labels.

### 2. Short-horizon prediction

The current model predicts the next position rather than a long future trajectory.

### 3. Camera dependency

Movement patterns depend on camera viewpoint, scene geometry, and pedestrian density.

### 4. Threshold-based anomaly decision

The current anomaly detector relies on a validation-derived statistical threshold.

### 5. Tracking errors

Incorrect tracking IDs can affect trajectory quality and consequently anomaly detection.

---

# Future Improvements

Potential future directions include:

- multi-step trajectory forecasting
- attention-based trajectory models
- Transformer-based trajectory prediction
- social interaction modeling between pedestrians
- scene-aware anomaly detection
- homography / world-coordinate transformation
- trajectory clustering
- probabilistic trajectory prediction
- uncertainty estimation
- learned anomaly scoring
- larger-scale MOT datasets
- stronger ground-truth anomaly datasets
- model monitoring
- cloud GPU deployment

---

# Resume-Level Summary

A concise description of the project:

> **Real-Time Multi-Object Tracking & Trajectory-Based Anomaly Detection** — Built an end-to-end computer vision system using YOLOv8, ByteTrack and LSTM-based trajectory forecasting to identify movement anomalies from MOT17 pedestrian sequences. Improved the initial pipeline through multi-sequence training, normalized coordinates, velocity features, sequence-level data splitting and validation-based anomaly thresholding. Deployed the inference pipeline through FastAPI with live-camera/video processing, visual trajectory prediction, MLflow experiment tracking and automated API tests.

---

# Project Highlights

```text
✓ YOLOv8 person detection
✓ ByteTrack multi-object tracking
✓ Persistent track IDs
✓ MOT17 multi-sequence processing
✓ Normalized trajectory features
✓ Velocity feature engineering
✓ LSTM trajectory forecasting
✓ GRU experimental comparison
✓ MAE / RMSE / ADE / FDE evaluation
✓ Validation-based anomaly threshold
✓ Temporal anomaly smoothing
✓ FastAPI inference API
✓ Live camera inference
✓ Recorded video inference
✓ Real-time visualization
✓ MLflow experiment tracking
✓ Pytest API testing
✓ Git/GitHub version control
```

---

# Author

**Anubhav Trivedi**

BS-MS Mathematics and Data Science  
Harcourt Butler Technical University, Kanpur

GitHub: `anubhavtrivedi18`

---

## Disclaimer

This project is an experimental research/engineering system for trajectory-based anomaly detection. The NORMAL/ANOMALY labels are generated from trajectory prediction deviation and should not be interpreted as ground-truth behavioral anomaly classification.
