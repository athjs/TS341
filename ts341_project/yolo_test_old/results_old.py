"""Module pour exécuter YOLO sur une vidéo et renvoyer les détections frame par frame."""

import os
import sys
from typing import Generator, Tuple
import cv2
from ultralytics import YOLO
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(CURRENT_DIR, "..", "..", "videos")
sys.path.append(VIDEO_PATH)


def run_yolo(
    video_path: str = None,
) -> Generator[
    Tuple[int, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray], None, None
]:
    """Exécute YOLO sur une vidéo et renvoie les résultats frame par frame.

    Args:
        video_path (str, optional): Nom de la vidéo (sans extension) ou chemin complet.
            Si None, utilise "video2_short.mp4" par défaut.

    Yields:
        Tuple contenant:
            - frame_number (int)
            - frame_resized (np.ndarray)
            - height_ratio_resize (float)
            - width_ratio_resize (float)
            - detections (np.ndarray): coordonnées des boîtes [x1, y1, x2, y2]
            - confidences (np.ndarray)
            - class_ids (np.ndarray)
    """
    if video_path is None:
        video_path = os.path.join(VIDEO_PATH, "video2_short.mp4")
    else:
        video_path = os.path.join(VIDEO_PATH, video_path + ".mp4")

    MODEL_PATH = os.path.join(CURRENT_DIR, "kaggle_training.pt")
    model = YOLO(MODEL_PATH)

    frame_number = 0

    for result in model.predict(source=video_path, stream=True, verbose=False):
        frame = result.plot()
        original_height, original_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (1280, 720))
        height, width = frame_resized.shape[:2]

        height_ratio_resize = height / original_height
        width_ratio_resize = width / original_width

        detections = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        yield (
            frame_number,
            frame_resized,
            height_ratio_resize,
            width_ratio_resize,
            detections,
            confidences,
            class_ids,
        )

        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
