"""Module pour traitement vidéo avec suppression du fond et détection des objets en mouvement."""

from typing import Generator, Sequence
from typing_extensions import Any
import numpy as np
import cv2 as cv
import utils as u
from threading import Thread
from queue import Queue
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(CURRENT_DIR, "..", "videos")
sys.path.append(VIDEO_PATH)

display_q: Queue = Queue(maxsize=1)


def display_loop() -> None:
    """Affiche en continu les frames mises dans la queue display_q."""
    while True:
        item: Any = display_q.get()
        if item is None:
            return
        frame = item
        cv.imshow("Frame", frame)
        if cv.waitKey(30) & 0xFF == ord("q"):
            return


def frame_generator(
    cap: cv.VideoCapture, fgbg, kernel: np.ndarray
) -> Generator[tuple[int, list[tuple[int, int]], np.ndarray], None, None]:
    """Génère les frames d'une vidéo, leurs centroids et frame brute.

    Args:
        cap (cv.VideoCapture): Objet vidéo ouvert.
        fgbg: Background Subtractor (cv.createBackgroundSubtractorMOG2()).
        kernel (np.ndarray): Kernel pour les opérations morphologiques.

    Yields:
        Tuple[int, list[tuple[int,int]], np.ndarray]: frame_id, centroids et frame.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            return

        fgmask: np.ndarray = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        _, threshold = cv.threshold(fgmask.copy(), 121, 255, cv.THRESH_BINARY)
        threshold: np.ndarray
        contours: Sequence[np.ndarray] = get_contours(threshold)
        centroids: list[tuple[int, int]] = get_moving_centroïds(contours)
        frame_id: int = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        yield frame_id, centroids, frame


def get_movings(
    video_name: str,
) -> Generator[tuple[int, list[tuple[int, int]], np.ndarray], None, None]:
    """Ouvre une vidéo et génère les objets en mouvement.

    Args:
        video_name (str): Nom de la vidéo à traiter (fichier dans le dossier 'videos').

    Yields:
        Tuple[int, list[tuple[int,int]], np.ndarray]: frame_id, centroids et frame.
    """
    video_path = os.path.join(VIDEO_PATH, video_name)
    cap: cv.VideoCapture = u.openvideo(video_path)
    if cap is None or not cap.isOpened():
        raise Exception(f"La vidéo {video_path} n'a pas pu être ouverte.")
    kernel: np.ndarray = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgbg = cv.createBackgroundSubtractorMOG2()

    # Thread d'affichage
    # t: Thread = Thread(target=display_loop, daemon=True)
    # t.start()

    for frame_id, centroids, frame in frame_generator(cap, fgbg, kernel):
        try:
            display_q.put_nowait(frame)
        except:
            pass  # si l'afficheur n’a pas consommé, on écrase l’ancien
        yield frame_id, centroids, frame

    # display_q.put(None)
    # t.join()
    cap.release()
    cv.destroyAllWindows()


def get_contours(threshold: np.ndarray) -> Sequence[np.ndarray]:
    """Retourne les contours détectés dans une frame seuilée."""
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_moving_centroïds(contours: Sequence[np.ndarray]) -> list[tuple[int, int]]:
    """Retourne les centroids des contours détectés (objets en mouvement)."""
    centroids: list[tuple[int, int]] = []
    for contour in contours:
        area: float = cv.contourArea(contour)
        if 10 < area < 1500:
            M: dict[str, float] = cv.moments(contour)
            cx: int = int(M["m10"] / M["m00"])
            cy: int = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids
