"""Module for running YOLO detection on videos."""

from ultralytics import YOLO

# Charger un modèle pré-entraîné (par ex. COCO)
model = YOLO("kaggle_training.pt")  # tu peux utiliser yolov8s.pt, yolov8m.pt, etc.

# Traiter une vidéo
results = model.predict(
    source="capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4", show=True, save=True
)
