# Cahier des charges

_à rendre pour le 12/11/2025_

## Contexte 

Les drones civils sont de plus en plus présents dans notre espace aérien. Si leur utilisation légitime ne cesse de croître, ils représentent également des risques potentiels pour la sécurité des infrastructures sensibles, des événements publics et de la vie privée. La détection et l’identification de drones non autorisés constituent un enjeu majeur pour la sécurité.
Dans ce cadre, une entreprise spécialisée dans la sécurité des sites sensibles a fait appel au CATIE pour développer un système de détection de drones à partir d’images capturées par différents types de caméras

### Enjeux 


### Objectifs 

### Contraintes

- La solution proposée doit être fonctionnelle sur différents types de caméra
- Le drone doit être reconnaissable dans tout type de configuration et conditions externes
- Le taux de faux négatif doit être de 0%
- Le taux de faux positifs sera minimisé
- La résolution de la caméra  
- Temps de calcul inférieur au temps de déplacement du drone
- Qualité de l'image 
- Fonctionnement sur de multiples caméras

## État des lieux et contexte existant 

- Bases de données Kaggle 
- Yolo Python 

## Description du projet

### Description de la solution proposée long terme
La solution visée repose sur un modèle de vision par ordinateur basé sur des réseaux de neurones convolutifs (type YOLO), combiné à une approche d’analyse de la dynamique de la scène pour améliorer la robustesse du système.
L’idée est d’intégrer plusieurs couches d’analyse — détection, classification, suivi temporel et contextualisation — afin de permettre une reconnaissance stable, rapide et adaptable à différents types de caméras et de conditions (lumière, météo, mouvement de caméra, etc.).

En parallèle, un traitement du fond statique (via soustraction ou effacement de fond) permettrait d’isoler plus efficacement les objets en mouvement, en ne traitant que les régions d’intérêt.

### Description du prototype qui sera réalisé

Étant donné le temps limité, le prototype visera à valider la faisabilité de la détection automatique de drones à partir d’un flux vidéo.
Nous utiliserons un modèle YOLO pré-entraîné pour effectuer la détection d’objets, en se concentrant sur la reconnaissance des drones à partir des vidéos fournies.

Nous expérimenterons également une approche d’effacement statique :

- un algorithme simple de soustraction de fond permettra d’identifier les zones en mouvement dans la scène ;

- si le fond reste statique et qu’un nouvel élément apparaît, le système pourra déterminer qu’un objet (potentiellement un drone) entre dans le champ.

L’objectif du prototype est donc d’obtenir une première preuve de concept démontrant la détection de drones dans un contexte simplifié, ouvrant la voie à un développement plus robuste et temps réel.