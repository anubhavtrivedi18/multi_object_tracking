import pandas as pd
import numpy as np
import os
from pathlib import Path
track_dir = Path(r"C:\Users\HP\Desktop\multi_object_tracking\data\tracking")

sequences = sorted([folder for folder in track_dir.iterdir()])

OUTPUT_DIR = (
    r"C:\Users\HP\Desktop\multi_object_tracking"
    r"\data\sequences"
)


sequence_length = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("LOADING TRAJECTORY DATA")
print("="*60)

train_split = int(0.70 * len(sequences))
val_split = int(0.85 * len(sequences))

train_sequences = sequences[:train_split]
val_sequences = sequences[train_split:val_split]
test_sequences = sequences[val_split:]




x_train = [ ]
y_train = [ ]

x_test = [ ]
y_test = [ ]

x_val = []
y_val = []

test_metadata = [ ]


# Training Sequences
for sequence in train_sequences:
    INPUT_PATH = sequence
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows: {len(df)}")
    df = df.sort_values(['id','frame']).reset_index(drop = True)
    print("Creating Training Sequences")
    for track_id, group in df.groupby("id"):
        positions = group[['normalized_center_x','normalized_center_y','velocity_x','velocity_y']].values
        if len(positions)<= sequence_length:
            continue
        for i in range(len(positions)-sequence_length):
            input_sequence = positions[i:i+sequence_length]
            target_position = positions[i+sequence_length][:2]
            x_train.append(input_sequence)
            y_train.append(target_position)


# validation Sequences
for sequence in val_sequences:
    INPUT_PATH = sequence
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows: {len(df)}")
    df = df.sort_values(['id','frame']).reset_index(drop = True)
    print("Creating Training Sequences")
    for track_id, group in df.groupby("id"):
        positions = group[['normalized_center_x','normalized_center_y','velocity_x','velocity_y']].values
        if len(positions)<= sequence_length:
            continue
        for i in range(len(positions)-sequence_length):
            input_sequence = positions[i:i+sequence_length]
            target_position = positions[i+sequence_length][:2]
            x_val.append(input_sequence)
            y_val.append(target_position)

# Testing Sequences
for sequence in test_sequences:
    INPUT_PATH = sequence
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows: {len(df)}")
    df = df.sort_values(['id','frame']).reset_index(drop=True)
    for track_id, group in df.groupby('id'):
        positions = group[['normalized_center_x','normalized_center_y','velocity_x','velocity_y']].values
        frames = group['frame'].values
        if len(positions) <= sequence_length:
            continue

        for i in range(len(positions)-sequence_length):
            input_sequence = positions[i:i+sequence_length]
            target_position = positions[i+sequence_length][:2]
            x_test.append(input_sequence)
            y_test.append(target_position)
            test_metadata.append([sequence.name,int(track_id),int(frames[i+sequence_length])])

# Convert to Numpy arrays
x_train = np.array(x_train,dtype=np.float32)
y_train = np.array(y_train,dtype=np.float32)
x_test = np.array(x_test, dtype = np.float32)
y_test = np.array(y_test,dtype=np.float32)
x_val = np.array(x_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
test_metadata = np.array(test_metadata,dtype = object)
# Save training data
x_train_path = os.path.join(OUTPUT_DIR,'x_train.npy')
np.save(x_train_path,x_train)
y_train_path = os.path.join(OUTPUT_DIR,'y_train.npy')
np.save(y_train_path,y_train)

x_test_path = os.path.join(OUTPUT_DIR,'x_test.npy')
np.save(x_test_path,x_test)
y_test_path = os.path.join(OUTPUT_DIR,'y_test.npy')
np.save(y_test_path,y_test)

np.save(os.path.join(OUTPUT_DIR,"x_val.npy"),x_val)

np.save(os.path.join(OUTPUT_DIR,"y_val.npy"),y_val)

metadata_path = os.path.join(OUTPUT_DIR,'test_metadata.npy')
np.save(metadata_path,test_metadata)
print(
    os.path.join(
        OUTPUT_DIR,
        "test_metadata.npy"
    )
)


x_train = np.load("data/sequences/x_train.npy")
y_train = np.load("data/sequences/y_train.npy")
x_test = np.load("data/sequences/x_test.npy")
y_test = np.load("data/sequences/y_test.npy")
x_val = np.load("data/sequences/x_val.npy")
y_val = np.load("data/sequences/y_val.npy")

print("X train:", x_train.shape)
print("Y train:", y_train.shape)
print("X test :", x_test.shape)
print("Y test :", y_test.shape)
print("X val  :", x_val.shape)
print("Y val  :", y_val.shape)