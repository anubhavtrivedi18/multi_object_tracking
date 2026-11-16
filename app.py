from fastapi import FastAPI, UploadFile, File, Response,BackgroundTasks,Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
from models.trajectory_lstm_arch import TrajectoryLSTM


import __main__
__main__.TrajectoryLSTM = TrajectoryLSTM

ANOMALY_THRESHOLD = 0.0119

base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(base_dir,"yolov8n.pt")


print("Loading yolov8n.pt model")
yolo_model = YOLO(yolo_model_path)
print("yolov8n.pt model loaded scuccessfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model_path = os.path.join(base_dir,"models","trajectory_lstm.pth")
print("Loading LSTm model")
lstm_model = torch.load(lstm_model_path,map_location=device,weights_only=False)
lstm_model = lstm_model.to(device)
lstm_model = lstm_model.to(device)
lstm_model.eval()
print("LSTM model loaded successfully")
print("Device:",device)


track_history = defaultdict(lambda: deque(maxlen=10))
previous_positions = {}
anomaly_streak = defaultdict(int)




def process_frame(frame):
    processed_frame = frame.copy()
    result = yolo_model.track(processed_frame,persist=True,tracker ="bytetrack.yaml",classes=[0])
    processed_frame = result[0].plot()

    boxes = result[0].boxes
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        print("Track IDS:",track_ids)
        xyxy = boxes.xyxy.cpu().numpy()
        for box, track_id in zip(xyxy, track_ids):
            x1,y1,x2,y2 = box
            width = x2-x1
            height = y2-y1
            center_x = x1+width/2
            center_y = y1+height/2
            frame_height, frame_width = processed_frame.shape[:2]
            normalized_x = center_x/frame_width
            normalized_y = center_y/frame_height
            current_position = ( normalized_x,normalized_y)
            if track_id in previous_positions:
                previous_position = previous_positions[track_id]
                velocity_x = (current_position[0]-previous_position[0])
                velocity_y = (current_position[1]-previous_position[1])
            else:
                velocity_x = 0.0
                velocity_y = 0.0

            previous_positions[track_id] = current_position
            track_history[track_id].append([normalized_x,normalized_y,velocity_x,velocity_y])
            if len(track_history[track_id])>= 10:
                print("ID:",track_id,"History length:",len(track_history[track_id]))
                history = list(track_history[track_id])
                input_tensor = torch.tensor(history,dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = lstm_model(input_tensor)
                prediction_x = prediction[0][0].item()
                prediction_y = prediction[0][1].item()
                predicted_pixel_x = int(prediction_x*frame_width)
                predicted_pixel_y = int(prediction_y*frame_height)
                actual_pixel_x = int(normalized_x * frame_width)
                actual_pixel_y = int(normalized_y * frame_height)
                cv2.circle(processed_frame,(predicted_pixel_x,predicted_pixel_y),6,(0,0,255),-1)
                cv2.circle(processed_frame,(actual_pixel_x,actual_pixel_y),6,(0,255,0),-1)
                cv2.line(processed_frame,(actual_pixel_x,actual_pixel_y),(predicted_pixel_x,predicted_pixel_y),(0,0,255),2)
                for i in range(1,len(history)):
                    previous_x = int(history[i-1][0]*frame_width)
                    previous_y = int(history[i-1][1]*frame_height)
                    current_x = int(history[i][0]*frame_width)
                    current_y = int(history[i][1]*frame_height)
                    cv2.line(processed_frame,(previous_x,previous_y),(current_x,current_y),(225,225,0),4)
                    cv2.circle(processed_frame,(current_x,current_y),4,(225,225,0),-1)
                error = ((normalized_x-prediction_x)**2+(normalized_y-prediction_y)**2)**0.5
                if error > ANOMALY_THRESHOLD:
                    anomaly_streak[track_id]+=1
                else:
                    anomaly_streak[track_id] = 0
                if anomaly_streak[track_id] >= 3:
                    status = "Anomaly"
                else:
                    status = "Normal"
                cv2.putText(processed_frame,f"ID: {track_id} | Error: {error:.4f} | {status}",(int(x1),max(int(y1)-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0) if status == "Normal" else (0,0,255),2)
                
                cv2.putText(processed_frame, "Green = Actual",(20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)

                cv2.putText(processed_frame,"Red = Predicted",(20, 55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)

                cv2.putText(processed_frame,"Sky Blue = Past Trajectory",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225, 225, 0),2)
            

                print("ID:", track_id, "Prediction Error:", error, "streak:", anomaly_streak[track_id], "Status:", status)
                print("ID:", track_id, "Prediction:", prediction_x, prediction_y)
            cv2.putText(processed_frame,f"Threshold: {ANOMALY_THRESHOLD:.4f}",(20,135),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225,0,225),2)
            print("ID:", track_id, "Normalized:", normalized_x, normalized_y, "Velocity:", velocity_x, velocity_y)
    return processed_frame

def process_vedio(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        success, frame = video.read()
        if not success:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Processed Vedio",processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


app = FastAPI(
    title = "Real-Time Multi-Object Tracking System",
    description = "YOLO + ByteTrack + LSTM trajectory prediction + anomaly detection",
    version = "1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
        return FileResponse(
        os.path.join(base_dir, "static", "index.html"))

@app.get("/health")
def health():
    return {'status':"healthy"}


@app.post("/process-video")
async def process_video_endpoint(
    file: UploadFile = File(...)
):

    contents = await file.read()

    temp_video_path = os.path.join(
        base_dir,
        "temp_video.mp4"
    )

    with open(temp_video_path, "wb") as video_file:
        video_file.write(contents)

    def generate_frames():

        video = cv2.VideoCapture(
            temp_video_path
        )

        while True:

            success, frame = video.read()

            if not success:
                break

            processed_frame = process_frame(frame)

            success, buffer = cv2.imencode(
                ".jpg",
                processed_frame
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        video.release()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def delete_file(file_path):

    if os.path.exists(file_path):
        os.remove(file_path)


@app.post("/process-video")
async def process_video_endpoint(background_tasks: BackgroundTasks,file: UploadFile = File(...)):
    contents = await file.read()

    temp_video_path = os.path.join(base_dir, "temp_video.mp4")

    with open(temp_video_path, "wb") as video_file:
        video_file.write(contents)
    video = cv2.VideoCapture(temp_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = os.path.join(
    base_dir,
    "temp_processed_video.mp4"
)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_video = cv2.VideoWriter(
    temp_output_path,
    fourcc,
    fps,
    (width, height)
)
    while True:

        success, frame = video.read()

        if not success:
            break
        processed_frame = process_frame(frame)
        output_video.write(processed_frame)
    video.release()
    output_video.release()
    os.remove(temp_video_path)
    background_tasks.add_task(delete_file,temp_output_path)

    return FileResponse(
    temp_output_path,media_type="video/mp4")

@app.post("/process-frame")
async def process_frame_endpoint(request: Request):
    image_bytes = await request.body()
    image_array = np.frombuffer(image_bytes,dtype = np.uint8)
    frame = cv2.imdecode(image_array,cv2.IMREAD_COLOR)
    if frame is None:
        return Response(content = "Invalid image",status_code = 400)

    processed_frame = process_frame(frame)
    success, buffer = cv2.imencode(".jpg",processed_frame)
    if not success:
        return Response(content = "frame encoding failed",status_code=500)

    return Response(content=buffer.tobytes(),media_type = 'image/jpeg')

