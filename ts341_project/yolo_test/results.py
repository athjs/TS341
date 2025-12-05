"""Importing YOLO for learning."""

# pyright: reportPrivateImportUsage=false
# pyright: reportMissingImports=false
from ultralytics import YOLO
from typing import Any

# Charger un modèle pré-entraîné (par ex. COCO)
model: YOLO = YOLO(
    "ts341_project/yolo_test/kaggle_training.pt"
)  # tu peux utiliser yolov8s.pt, yolov8m.pt, etc.


def yolo_predict(image_path: str) -> Any:
    """Effectue la prédiction YOLO sur une image donnée.

    Args:
        image_path (str): Chemin vers l'image à analyser.

    Returns:
        Any: Résultat retourné par YOLO (non typé officiellement).
    """
    results = model.predict(source=image_path)  # type: ignore
    print(results)
    return results
