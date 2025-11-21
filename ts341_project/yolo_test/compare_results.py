# import resultsReel
# import resultsPred

def result_score(n_frame: int, xReel: int, yReel: int, xPred: int, yPred: int) -> float:
    """
    Estimation de la précision des résultats obtenus par rapport aux résultats attendus.
    Cette fonction ne marche que s'il n'y a qu'un seul drone dans la vidéo.
    
    Args:
        n_frame (int): numéro de la frame de la video analysée
        xReel, yReel (int): coordonnées réelles du drone (attendues)
        xPred, yPred (int): coordonnées prédites par le modèle

    Returns:
        float: score de précision (plus il est bas, plus la prédiction est précise)
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
            score = 0  # Prédiction correcte
        elif distance <= seuil2:
            score = 20  # Prédiction partiellement correcte
        else: score = 50  # Prédiction incorrecte

    return score