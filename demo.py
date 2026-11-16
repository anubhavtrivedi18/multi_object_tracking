from pathlib import Path
import os
track_dir = Path(r"C:\Users\HP\Desktop\multi_object_tracking\data\tracking")

sequences = sorted([folder for folder in track_dir.iterdir()])

output_dir = r"C:\Users\HP\Desktop\multi_object_tracking\data\tracking"

for sequences in sequences:
    source = sequences/"img1"
    print("processing:",sequences.name)
    print("Images",source)
    CSV_PATH = os.path.join(output_dir,f"{sequences.name}_tracks.csv")
    print("CSV_PATH:",CSV_PATH)