from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional

# --- Chargement du modèle ---
model: YOLO = YOLO("model_weigths/basic_yolo.pt")

# --- Chargement de la vidéo ---
video: cv2.VideoCapture = cv2.VideoCapture(
    "videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4"
)

# --- Frames à traiter ---
start_frame: int = 0
end_frame: int = 800

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while True:
    frame_id: int = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    if frame_id > end_frame:
        break

    ret: bool
    frame: Optional[np.ndarray]
    ret, frame = video.read()
    if not ret or frame is None:
        break

    results = model.predict(frame, verbose=False)  # type: ignore
    print(results)

video.release()
