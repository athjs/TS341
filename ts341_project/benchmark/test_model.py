from ultralytics import YOLO
import cv2

model = YOLO("model_weigths/basic_yolo.pt")
video = cv2.VideoCapture("videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4")

start_frame = 0     
end_frame = 800       

start_frame = int(start_frame)
end_frame = int(end_frame)

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while True:
    frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    if frame_id > end_frame:
        break

    ret, frame = video.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    print(results)

video.release()
