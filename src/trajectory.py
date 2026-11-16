import pandas as pd
import os


CSV_PATH = os.path.join(r"C:\Users\HP\Desktop\multi_object_tracking"
                        r"\data\tracking\MOT17-02-FRCNN_tracks.csv")

OUTPUT_DIR = os.path.join( r"C:\Users\HP\Desktop\multi_object_tracking"
    r"\data\trajectories")


# Create output Directory

os.makedirs(OUTPUT_DIR, exist_ok = True)

# Load Tracking Data

print("="*60)
print("TRAJECTORY EXTRACTION COMPLETED")
print("=" * 60)

print(f"Output saved to:")
print(OUTPUT_DIR)

print("=" * 60)