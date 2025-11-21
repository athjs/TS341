from ultralytics import YOLO
import cv2

# Charger un modèle pré-entraîné (par ex. COCO)
model = YOLO("ts341_project/yolo_test/kaggle_training.pt")  # tu peux utiliser yolov8s.pt, yolov8m.pt, etc.

def yolo_predict(image_path):
    results = model.predict(source=image_path)
    print(results)
    return results
