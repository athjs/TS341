"""Filtre et suivi de centroïdes en utilisant YOLO et détection de mouvement."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from typing import List, Tuple, Sequence
import csv

from Butterworth import ButterworthLPF

# Filtre Butterworth pour X et Y
Butterworth_x = ButterworthLPF(cutoff=0.05, fs=1, order=3)
Butterworth_y = ButterworthLPF(cutoff=0.05, fs=1, order=3)

# Chemins
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_TEST_PATH = os.path.join(CURRENT_DIR, "..", "yolo_test_old")
sys.path.append(YOLO_TEST_PATH)

MOVINGS_PATH = os.path.join(CURRENT_DIR, "..")
sys.path.append(MOVINGS_PATH)

from results_old import run_yolo
from background import get_movings

# --- Fonctions utilitaires --- #


def carre_distance(
    centroid1: list[float], centroid2: list[float]
) -> float:
    """Calcule la distance euclidienne au carré entre deux centroïdes."""
    return (centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2


def closest_centroid(
    centroid_goal: list[float], centroids_list: Sequence[list[float]]
) -> list[float]:
    """Retourne le centroïde dans centroids_list le plus proche de centroid_goal."""
    distance_min = math.inf
    indice_min = 0
    if len(centroids_list) != 0:
        for k, centroid in enumerate(centroids_list):
            distance = (centroid_goal[0] - centroid[0]) ** 2 + (
                centroid_goal[1] - centroid[1]
            ) ** 2
            if distance <= distance_min:
                distance_min = distance
                indice_min = k
        return centroids_list[indice_min]
    return [0.0, 0.0]

def ecrit_csv(data : list[int]) -> None:
    """Ecrit une ligne dans le fichier csv."""
    with open(CURRENT_DIR+"/data_butterworth.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Écrire toutes les lignes
        writer.writerow(data)

def run_filter(video_name: str) -> None:
    """La fonction secrete qui fait tout le filtre"""
    buffer_drone: List[List] = []

    values: List[float] = []
    fig, ax = plt.subplots()
    plt.ion()

    with open(CURRENT_DIR+"/data_butterworth.csv", "w", newline="", encoding="utf-8") as csvfile:
        pass

    first_yolo_detection: bool = False

    gen_yolo = run_yolo(video_name)
    gen_motion = get_movings(video_name)

    # Variables pour Pyright
    previous_centroid_yolo: list[float] = [0.0, 0.0]
    previous_close_centroid: list[float] = [0.0, 0.0]
    previous_centroid_passe_bas: list[float] = [0.0, 0.0]

    # --- Boucle principale --- #
    for (
        frame_number,
        frame_resized,
        height_ratio_resize,
        width_ratio_resize,
        detections,
        confidences,
        class_ids,
    ), (frame_id, centroids_raw, motion_frame_raw) in zip(gen_yolo, gen_motion):
        # Forcer tous les centroïdes en float
        centroids: List[list[float]] = [
            [float(cx), float(cy)] for cx, cy in centroids_raw
        ]
        motion_frame: np.ndarray = motion_frame_raw

        centroid_passe_bas: list[float] = [0.0, 0.0]
        close_centroid: list[float] = [0.0, 0.0]

        print(f"Frame numéro {frame_number} : ", end="")
        if len(detections) == 0:
            print("Aucune détection YOLO, ", end="")
            if first_yolo_detection:
                print("mise à jour du filtre et du opencv")
                close_centroid = closest_centroid(
                    previous_close_centroid, centroids
                )
                cv2.circle(
                    frame_resized,
                    (
                        int(close_centroid[0] * width_ratio_resize),
                        int(close_centroid[1] * height_ratio_resize),
                    ),
                    10,
                    (0, 0, 255),
                    -1,
                )
                centroid_passe_bas[0]= Butterworth_x.update(close_centroid[0])
                centroid_passe_bas[1]= Butterworth_y.update(close_centroid[1])

                cv2.circle(
                    frame_resized,
                    (
                        int(centroid_passe_bas[0] * width_ratio_resize),
                        int(centroid_passe_bas[1] * height_ratio_resize),
                    ),
                    10,
                    (0, 255, 255),
                    -1,
                )
                previous_close_centroid = close_centroid
            else:
                print("pas encore d'initialisation")
        else:
            print("Détection YOLO, ", end="")
            x1, y1, x2, y2 = detections[0]
            centroid_yolo: list[float] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            if not first_yolo_detection:
                print("initialisation")
                first_yolo_detection = True
                Butterworth_x.update(centroid_yolo[0])
                Butterworth_y.update(centroid_yolo[1])
                close_centroid = closest_centroid(
                    centroid_yolo, centroids
                )
                previous_centroid_passe_bas = centroid_yolo
                previous_close_centroid = centroid_yolo
                previous_centroid_yolo = centroid_yolo
            else:
                print("mise à jour du filtre, du opencv et du YOLO. ", end="")
                if carre_distance(centroid_yolo, previous_centroid_yolo) <= 10000:
                    print("Pas de saut de yolo")
                    cv2.circle(
                        frame_resized,
                        (
                            int(centroid_yolo[0] * width_ratio_resize),
                            int(centroid_yolo[1] * height_ratio_resize),
                        ),
                        10,
                        (0, 255, 0),
                        -1,
                    )
                    centroid_passe_bas[0]= Butterworth_x.update(centroid_yolo[0])
                    centroid_passe_bas[1]= Butterworth_y.update(centroid_yolo[1])
                    cv2.circle(
                        frame_resized,
                        (
                            int(centroid_passe_bas[0] * width_ratio_resize),
                            int(centroid_passe_bas[1] * height_ratio_resize),
                        ),
                        10,
                        (0, 255, 255),
                        -1,
                    )
                    close_centroid = closest_centroid(
                        centroid_passe_bas, centroids
                    )
                    cv2.circle(
                        frame_resized,
                        (
                            int(close_centroid[0] * width_ratio_resize),
                            int(close_centroid[1] * height_ratio_resize),
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )
                    previous_centroid_passe_bas = centroid_passe_bas
                    previous_centroid_yolo = centroid_yolo
                else:
                    print("Saut de yolo")
                    close_centroid = closest_centroid(
                        previous_close_centroid, centroids
                    )
                    cv2.circle(
                        frame_resized,
                        (
                            int(close_centroid[0] * width_ratio_resize),
                            int(close_centroid[1] * height_ratio_resize),
                        ),
                        10,
                        (0, 0, 255),
                        -1,
                    )
                    centroid_passe_bas[0]= Butterworth_x.update(close_centroid[0])
                    centroid_passe_bas[1]= Butterworth_y.update(close_centroid[1])
                    cv2.circle(
                        frame_resized,
                        (
                            int(centroid_passe_bas[0] * width_ratio_resize),
                            int(centroid_passe_bas[1] * height_ratio_resize),
                        ),
                        10,
                        (0, 255, 255),
                        -1,
                    )
                    previous_close_centroid = close_centroid

            for det, conf, cls in zip(detections, confidences, class_ids):
                buffer_drone.append([det, conf, cls])
                sys.stdout.flush()
        data = [frame_number, int(centroid_passe_bas[0]), int(centroid_passe_bas[1])]
        ecrit_csv(data)

        # Affichage
        cv2.putText(
            frame_resized, "YOLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame_resized, "OpenCV", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            frame_resized,
            "Butterworth",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Resultat YOLO", frame_resized)

# --- Lecture vidéo --- #

video_name = "video2_short"
if len(sys.argv) > 1:
    video_name = sys.argv[1]
print("Utilisation de la vidéo :", video_name)

run_filter(video_name)
