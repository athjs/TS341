import csv
import numpy as np

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
    score = 0.0
    # ---------------------------------
    # Cas de faux positifs
    if xReel == xReel == -1 and yReel == -1 and (xPred != -1 or yPred != -1):
        score = 0  # du coup on s'en fou mais peut être qu'on voudra le pénaliser quand même plus tard

    # Cas de faux négatifs
    elif (xReel != -1 or yReel != -1) and xPred == -1 and yPred == -1:
        score += 50  # pénalité MAXIMUM

    else:
        # Calcul de la distance euclidienne entre les coordonnées réelles et prédites
        distance = ((xReel - xPred) ** 2 + (yReel - yPred) ** 2) ** 0.5
        # Définir un seuil pour considérer une prédiction comme correcte
        seuil1 = 50.0  # par exemple, 50 pixels
        seuil2 = 200.0  # par exemple, 100 pixels

        if distance <= seuil1:
            return 0  # Prédiction correcte
        elif distance <= seuil2:
            return 20  # Prédiction partiellement correcte
        else:
            return 50  # Prédiction incorrecte


def confusion_matrix_score(xReel: int, yReel: int, xPred: int, yPred: int, seuil_tolerance: int) :
    """Estimate precision of results compared to expected results.

    Estimation de la précision des résultats obtenus par rapport aux résultats attendus.
    Cette fonction ne marche que s'il n'y a qu'un seul drone dans la vidéo.

    Args:
        xReel: coordonnée X réelle du drone (attendue).
        yReel: coordonnée Y réelle du drone (attendue).
        xPred: coordonnée X prédite par le modèle.
        yPred: coordonnée Y prédite par le modèle.

    Returns:
        index of [PP, PN, NP, NN] premier étant réel, deuxieme étant prédi

    """
    PP, PN, NP, NN = np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0,0,1,0]), np.array([0,0,0,1])

    # ---------------------------------
    # Cas NN
    if (xReel == - 1) and (xPred == - 1):
        return NN

    # Cas NP
    if (xReel == - 1) and (xPred != - 1):
        return NP
    
    #Cas PN 
    if (xReel != - 1) and (xPred == -1):
        return PN
    
    #Cas PP
    if (xReel != - 1) and (xPred != - 1):
        distance = ((xReel - xPred) ** 2 + (yReel - yPred) ** 2) ** 0.5
        if distance >= seuil_tolerance :
            return NP + PN 
        else :
            return PP

def evaluate_score(csv_path):
    """Évalue les performances du modèle sur une vidéo annotée.

    Args:
        csv_path: chemin vers le fichier CSV contenant les annotations et les prédictions.

    Returns:
        Score total pour la vidéo.
    """

    total_score = 0.0
    csvfile_pred = open(csv_path, newline='\n')
    csvfile_real = open("ts341_project/benchmark/model_results/real_results.csv", newline='\n')
    reader_pred = csv.reader(csvfile_pred, delimiter=',')
    reader_real = csv.reader(csvfile_real, delimiter=',')
    next(reader_pred)
    next(reader_real)
    for row_pred, row_real in zip(reader_pred, reader_real):
        total_score += result_score(int(row_real[1]), int(row_real[2]), int(row_pred[1]), int(row_pred[2]))
    return total_score 

def evaluate_confusion_matrix(csv_path):
    """Évalue les performances du modèle sur une vidéo annotée.

    Args:
        csv_path: chemin vers le fichier CSV contenant les annotations et les prédictions.

    Returns:
        Matrice de confusion
    """

    Matrix = np.array([0,0,0,0])
    csvfile_pred = open(csv_path, newline='\n')
    csvfile_real = open("ts341_project/benchmark/model_results/real_results.csv", newline='\n')
    reader_pred = csv.reader(csvfile_pred, delimiter=',')
    reader_real = csv.reader(csvfile_real, delimiter=',')
    next(reader_pred)
    next(reader_real)
    for row_pred, row_real in zip(reader_pred, reader_real):
        score = confusion_matrix_score(int(row_real[1]), int(row_real[2]), int(row_pred[1]), int(row_pred[2]), 15)
        Matrix += score
    return Matrix

def full_evaluation(nom_model,csv_path) :
    score = evaluate_score(csv_path)
    confusion_matrix =   evaluate_confusion_matrix(csv_path)  
    print("Score : ", score)
    print("Matrix : ", confusion_matrix) 
    
    # Écriture du CSV
    data = [[str(nom_model), score, confusion_matrix]]
    with open("ts341_project/benchmark/evaluations.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
csv_path = "ts341_project/benchmark/model_results/test_results.csv"
full_evaluation("Test", csv_path)


