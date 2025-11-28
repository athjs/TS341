"""Prédiction de Yolo sur une image."""

from ultralytics import YOLO
import cv2

# Charger le modele
model = YOLO("ts341_project/yolo_test/kaggle_training.pt")


def yolo_predict(image_path):
    """_Effectue une prédiction YOLO sur une image donnée."""
    results = model.predict(source=image_path)
    print(results)
    return results
