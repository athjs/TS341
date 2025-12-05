"""Module pour exécuter YOLO sur une vidéo et renvoyer les détections frame par frame."""

# pyright: reportPrivateImportUsage=false
# pyright: reportMissingImports=false
from ultralytics import YOLO
import os
import sys
from typing import Generator, Tuple, Optional
import cv2
import numpy as np
import torch  # <-- ajouter pour typer les Tensors

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(CURRENT_DIR, "..", "..", "videos")
sys.path.append(VIDEO_PATH)


def run_yolo(
    video_path: Optional[str] = None,
) -> Generator[
    Tuple[int, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray],
    None,
    None,
]:
    """Exécute YOLO sur une vidéo et renvoie les résultats frame par frame."""

    if video_path is None:
        video_path = os.path.join(VIDEO_PATH, "video2_short.mp4")
    else:
        video_path = os.path.join(VIDEO_PATH, video_path + ".mp4")

    MODEL_PATH = os.path.join(CURRENT_DIR, "kaggle_training.pt")
    model = YOLO(MODEL_PATH)

    frame_number = 0

    for result in model.predict(source=video_path, stream=True, verbose=False):
        frame: np.ndarray = result.plot()

        original_height, original_width = frame.shape[:2]
        frame_resized: np.ndarray = cv2.resize(frame, (1280, 720))
        height, width = frame_resized.shape[:2]

        height_ratio_resize: float = height / original_height
        width_ratio_resize: float = width / original_width

        # -----------------------------------------
        # Gestion des boxes
        # -----------------------------------------
        boxes = result.boxes
        if boxes is None:
            detections: np.ndarray = np.empty((0, 4), dtype=np.float32)
            confidences: np.ndarray = np.empty((0,), dtype=np.float32)
            class_ids: np.ndarray = np.empty((0,), dtype=np.int64)
        else:
            # forcer Pyright à comprendre que c'est un Tensor
            xyxy_tensor: torch.Tensor = boxes.xyxy  # type: ignore
            conf_tensor: torch.Tensor = boxes.conf  # type: ignore
            cls_tensor: torch.Tensor = boxes.cls    # type: ignore

            detections = xyxy_tensor.cpu().numpy()
            confidences = conf_tensor.cpu().numpy()
            class_ids = cls_tensor.cpu().numpy()

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
