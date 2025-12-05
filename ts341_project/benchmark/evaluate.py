"""Evaluation d'un modele par rapport à la métric faite main."""

import csv
import numpy as np
from numpy.typing import NDArray
from typing import Iterator, List
from pathlib import Path

import cv2

def result_score(xReel: int, yReel: int, xPred: int, yPred: int) -> float:
    """Estimate precision of results compared to expected results.

    Estimation de la précision des résultats obtenus par rapport aux résultats attendus.
    Cette fonction ne marche que s'il n'y a qu'un seul drone dans la vidéo.

    Args:
        xReel: coordonnée X réelle du drone (attendue).
        yReel: coordonnée Y réelle du drone (attendue).
        xPred: coordonnée X prédite par le modèle.
        yPred: coordonnée Y prédite par le modèle.

    Returns:
        Score de précision (plus il est bas, plus la prédiction est précise).
    """
    score: float = 0.0
    # ---------------------------------
    # Cas de faux positifs
    if xReel == xReel == -1 and yReel == -1 and (xPred != -1 or yPred != -1):
        score = 0  # du coup on s'en fou mais peut être qu'on voudra le pénaliser quand même plus tard

    # Cas de faux négatifs
    elif (xReel != -1 or yReel != -1) and xPred == -1 and yPred == -1:
        score += 50  # pénalité MAXIMUM

    else:
        # Calcul de la distance euclidienne entre les coordonnées réelles et prédites
        distance: float = ((xReel - xPred) ** 2 + (yReel - yPred) ** 2) ** 0.5
        # Définir un seuil pour considérer une prédiction comme correcte
        seuil1: float = 50.0  # par exemple, 50 pixels
        seuil2: float = 200.0  # par exemple, 100 pixels

        if distance <= seuil1:
            return 0.0  # Prédiction correcte
        elif distance <= seuil2:
            return 20.0  # Prédiction partiellement correcte
        else:
            return 50.0  # Prédiction incorrecte
    return score


def confusion_matrix_score(
    xReel: int, yReel: int, xPred: int, yPred: int, seuil_tolerance: int
) -> NDArray[np.int_]:
    """Estimate precision of results compared to expected results.

    Estimation de la précision des résultats obtenus par rapport aux résultats attendus.
    Cette fonction ne marche que s'il n'y a qu'un seul drone dans la vidéo.

    Args:
        xReel: coordonnée X réelle du drone (attendue).
        yReel: coordonnée Y réelle du drone (attendue).
        xPred: coordonnée X prédite par le modèle.
        yPred: coordonnée Y prédite par le modèle.
        seuil_tolerance: tolérance pour considérer la prédiction comme fausse

    Returns:
        index of [PP, PN, NP, NN] premier étant réel, deuxieme étant prédi

    """
    PP: NDArray[np.int_] = np.array([1, 0, 0, 0])
    PN: NDArray[np.int_] = np.array([0, 1, 0, 0])
    NP: NDArray[np.int_] = np.array([0, 0, 1, 0])
    NN: NDArray[np.int_] = np.array([0, 0, 0, 1])

    # ---------------------------------
    # Cas NN
    if (xReel == -1) and (xPred == -1):
        return NN

    # Cas NP
    if (xReel == -1) and (xPred != -1):
        return NP

    # Cas PN
    if (xReel != -1) and (xPred == -1):
        return PN

    # Cas PP
    if (xReel != -1) and (xPred != -1):
        distance: float = ((xReel - xPred) ** 2 + (yReel - yPred) ** 2) ** 0.5

        # Si erreur trop grande → assimilé à double faute (NP + PN)
        if distance >= seuil_tolerance:
            return NP + PN

        # Sinon vrai positif
        return PP

    # Par sécurité, même si on ne devrait jamais arriver là
    return NN


def evaluate_score(csv_path: str | Path) -> float:
    """Évalue les performances du modèle sur une vidéo annotée.

    Args:
        csv_path: chemin vers le fichier CSV contenant les annotations et les prédictions.

    Returns:
        float: Score total pour la vidéo.
    """
    total_score: float = 0.0

    # Ouverture des fichiers CSV
    csvfile_pred = open(csv_path, newline="\n")
    csvfile_real = open(
        "ts341_project/benchmark/model_results/real_results.csv", newline="\n"
    )

    reader_pred: Iterator[List[str]] = csv.reader(csvfile_pred, delimiter=",")
    reader_real: Iterator[List[str]] = csv.reader(csvfile_real, delimiter=",")

    # Sauter les en-têtes
    next(reader_pred)
    next(reader_real)

    for row_pred, row_real in zip(reader_pred, reader_real):
        x_real: int = int(row_real[1])
        y_real: int = int(row_real[2])
        x_pred: int = int(row_pred[1])
        y_pred: int = int(row_pred[2])

        total_score += result_score(x_real, y_real, x_pred, y_pred)

    # Fermeture des fichiers
    csvfile_pred.close()
    csvfile_real.close()

    return total_score


def evaluate_confusion_matrix(csv_path: str | Path, tolerance : int) -> np.ndarray:
    """Évalue les performances du modèle sur une vidéo annotée.

    Args:
        csv_path: chemin vers le fichier CSV contenant les annotations et les prédictions.

    Returns:
        Matrice de confusion
    """
    Matrix: np.ndarray = np.array([0, 0, 0, 0], dtype=int)

    csvfile_pred = open(csv_path, newline="\n")
    csvfile_real = open(
        "ts341_project/benchmark/model_results/real_results.csv", newline="\n"
    )

    reader_pred = csv.reader(csvfile_pred, delimiter=",")
    reader_real = csv.reader(csvfile_real, delimiter=",")

    next(reader_pred)
    next(reader_real)

    for row_pred, row_real in zip(reader_pred, reader_real):
        x_real: int = int(row_real[1])
        y_real: int = int(row_real[2])
        x_pred: int = int(row_pred[1])
        y_pred: int = int(row_pred[2])
        """[PP, PN, NP, NN]"""
        score: np.ndarray = confusion_matrix_score(x_real, y_real, x_pred, y_pred, tolerance)
        Matrix += score

    return Matrix


def full_evaluation(nom_model: str, csv_path: str | Path, tolerance) -> None:
    """Évalue un modèle sur une vidéo annotée, affiche score et matrice de confusion, et écrit le résultat dans un CSV."""
    # Calcul du score et de la matrice de confusion
    score: float = evaluate_score(csv_path)
    confusion_matrix: np.ndarray = evaluate_confusion_matrix(csv_path, tolerance)

    # Affichage
    print("Score :", score)
    print("Matrix :", confusion_matrix)

    # Écriture du CSV
    data = [
        [nom_model+"_"+str(tolerance), score, confusion_matrix.tolist()]
    ]  # .tolist() pour écrire proprement la matrice
    output_csv = Path("ts341_project/benchmark/evaluations.csv")

    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def read_csv_points(path):
    """
    Lit un CSV 'frame,x,y' et retourne un dict : {frame: (x, y)}
    """
    points = {}
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines[1:]:   # on saute l'entête
            frame, x, y = line.split(",")
            points[int(frame)] = (int(float(x)), int(float(y)))
    return points


def generate_annotated_video(csv1_path, csv2_path, video_path, output_path="output.mp4"):
    # Lecture manuelle des CSV
    points1 = read_csv_points(csv1_path)
    points2 = read_csv_points(csv2_path)

    # Lecture vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo")

    # Infos vidéo
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Writer de sortie
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Points du premier CSV (rouge)
        if frame_idx in points1:
            x1, y1 = points1[frame_idx]
            cv2.circle(frame, (x1, y1), 6, (0, 0, 255), -1)

        # Points du second CSV (vert)
        if frame_idx in points2:
            x2, y2 = points2[frame_idx]
            cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Vidéo générée : {output_path}")

for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    csv_path = f"ts341_project/benchmark/model_results/Kaggle_dataset_{i}.csv"
    full_evaluation(f"Kaggle_dataset_{i}", csv_path, 100)
#generate_annotated_video("ts341_project/benchmark/model_results/real_results.csv", csv_path, "videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4")