
import torch
from ultralytics import YOLO
import csv
import os
from pathlib import Path

train_dir = Path(r"C:\Users\HP\Desktop\multi_object_tracking\data\MOT17\mot17\train")

sequences = sorted([folder for folder in train_dir.iterdir()
                    if folder.is_dir() and (folder/"img1").exists()])
output_dir = r"C:\Users\HP\Desktop\multi_object_tracking\data\tracking"

for sequence in sequences:
    source = sequence/"img1"
    CSV_PATH = os.path.join(output_dir,f"{sequence.name}_tracks.csv")
    os.makedirs(output_dir, exist_ok = True)


    model = YOLO("yolov8n.pt")
    Device = "cuda" if torch.cuda.is_available() else "cpu"

    previous_centers = {}



    with open(CSV_PATH,'w',newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame','id','x1','y1','width','height','center_x','center_y','confidence','class','normalized_center_x','normalized_center_y','velocity_x','velocity_y'])
        results = model.track(source = source, persist=True,tracker='bytetrack.yaml',stream=True,device=Device)


# Process every frame

        for frame_id, result in enumerate(results, start=1):
            boxes = result.boxes
            frame_height, frame_width = result.orig_shape
            if boxes.id is None:
                continue
            track_id = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy()
            for track_id, box, confidence, class_id in zip( track_id, xyxy, confidence,class_id):
                if class_id != 0:
                    continue

                x1,y1,x2,y2 = box
                width = x2-x1
                height = y2-y1
                center_x = x1+width/2
                center_y = y1+height/2
                normalized_center_x = center_x/frame_width
                normalized_center_y = center_y/frame_height
                current_center = (normalized_center_x,normalized_center_y)
                #caculate the velocity
                if track_id in previous_centers:
                    previous_center = previous_centers[track_id]
                    velocity_x = (current_center[0]-previous_center[0])
                    velocity_y = (current_center[1]-previous_center[1])
                else:
                    velocity_x = 0.0
                    velocity_y = 0.0

                # upadate the previous centers
                previous_centers[track_id]  = current_center
                writer.writerow([ frame_id, track_id, x1, y1, width,height,center_x,center_y,confidence,class_id,normalized_center_x,normalized_center_y,velocity_x,velocity_x])
                print("Done")


print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"GPU         : {torch.cuda.get_device_name(0)}")
