import numpy as np
import torch
import os
from PIL import Image
from models.trajectory_lstm_arch import TrajectoryLSTM

base_dir = r"C:\Users\HP\Desktop\multi_object_tracking"
x_test_path = os.path.join(base_dir,'data','sequences','x_test.npy')
y_test_path = os.path.join(base_dir,'data','sequences','y_test.npy')
metadata_path = os.path.join(base_dir,'data','sequences','test_metadata.npy')
model_path = os.path.join(base_dir,'models','trajectory_lstm.pth')


x_test = np.load(x_test_path)
y_test = np.load(y_test_path)
metadata = np.load(metadata_path,allow_pickle=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device:",device)

if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))
# Predictions
x_test = torch.tensor(x_test, dtype = torch.float32).to(device)

model = torch.load(model_path,map_location=device,weights_only=False)
model.eval

with torch.no_grad():
    predictions = model(x_test)

predictions = predictions.cpu().numpy()
print("prediction_shape:",predictions.shape)

error = predictions - y_test
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error**2))
distance_error = np.sqrt(np.sum(error**2,axis=1))
ade = np.mean(distance_error)
fde = np.mean(distance_error)

# pixcles based metrics
pixel_errors = []
for i in range(len(predictions)):
    sequence_name = metadata[i][0]
    sequence_folder = sequence_name.replace("_tracks.csv", "")
    image_folder = os.path.join(base_dir,'data','MOT17','mot17','train',sequence_folder,'img1')
    first_image = os.path.join(image_folder,'000001.jpg')
    with Image.open(first_image) as image:
        frame_width, frame_height = image.size
    x_error_pixels = ( error[i][0]*frame_width)
    y_error_pixles = (error[i][1]*frame_height)
    pixel_distance = np.sqrt(x_error_pixels**2+y_error_pixles**2)

    pixel_errors.append(pixel_distance)
pixel_errors = np.array(pixel_errors)
pixel_mae = np.mean(np.abs(pixel_errors))
pixel_rmse = np.sqrt(np.mean(pixel_errors**2))
pixel_ade = np.mean(pixel_errors)
pixel_fde = np.mean(pixel_errors)

print(f"MAE  : {mae:.4f} normalised_unit_score")
print(f"RMSE : {rmse:.4f} normalised_unit_score")
print(f"ADE  : {ade:.4f} normalised_unit_score")
print(f"FDE  : {fde:.4f} normalised_unit_score")

print()

print(f"MAE  : {pixel_mae:.2f} pixels")
print(f"RMSE : {pixel_rmse:.2f} pixels")
print(f"ADE  : {pixel_ade:.2f} pixels")
print(f"FDE  : {pixel_fde:.2f} pixels")