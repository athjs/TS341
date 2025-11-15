# Cahier des charges

_à rendre pour le 12/11/2025_

Groupe composé de : Durand Clément, Garcia Amandine, Pereira Mathis, Thiery Antony

## Contexte

Les drones civils sont de plus en plus présents dans notre espace aérien. Si leur utilisation légitime ne cesse de croître, ils représentent également des risques potentiels pour la sécurité des infrastructures sensibles, des événements publics et de la vie privée. La détection et l’identification de drones non autorisés constituent un enjeu majeur pour la sécurité.
Dans ce cadre, une entreprise spécialisée dans la sécurité des sites sensibles a fait appel au CATIE pour développer un système de détection de drones à partir d’images capturées par différents types de caméras.

### Objectifs

L'objectif du projet est de détecter un drone à partir d'images et de vidéos capturées par différentes caméras.
La détection ne doit pas dépendre des conditions ni du format de la prise de vue.
La priorité est d'éviter à tout prix de ne pas détecter les drones présents sur la vidéo.

### Technologies envisagées
**Langage / librairies :** Python avec OpenCV, NumPy, éventuellement SciPy pour certaines manipulations. Blender API sera potentiellement utilisé pour la génération d'un dataset d'entraînement simulé.

**Modèle de détection :** YOLO (pré-entraîné).

**Bases de données :** Kaggle, Anti-UAV Dataset, Drone Detection Dataset, Création de notre propre dataset avec l'API Blender et un model du drone.

**Techniques complémentaires :** Soustraction de fond / effacement de l'arrière-plan statique pour détecter le mouvement du drone par rapport à l’arrière-plan. Tracking et prévision de trajectoire pour minimiser le pourcentage de faux négatifs et, dans un second temps, de faux positifs.

### Contraintes

- Détection fiable des drones en temps réel.

- **Taux de faux négatifs = 0%**, faux positifs minimisés.

- Fonctionnement sur divers types de caméras connues à l'avance, dont une monochrome. Conditions d’éclairage variables.

- Rapidité de détection cohérente avec la vitesse du drone.

## Verrous technologiques

- **Difficulté de détection dans des conditions réelles**
Les performances des modèles de vision diminuent fortement en cas de mouvement de caméra, de faible luminosité, de météo variable ou de drones de petite taille.
Ces facteurs rendent la reconnaissance instable, notamment lorsque le drone se confond avec le décor ou d’autres objets en mouvement. La résolution optique parfois limitée empêche, même pour l'oeil humain, de détecter un drone de 20 cm à 20 mètres.

- **Absence de suivi temporel**
Le modèle YOLO réalise une détection image par image, sans continuité temporelle.
Cette approche peut provoquer des pertes de détection ponctuelles et nuit à la stabilité du suivi d’objet, essentielle pour une reconnaissance cohérente dans le temps.  Il est donc nécessaire d'implémenter un modèle de tracking.

- **Traitement en temps réel**
Avec le matériel à disposition, il est difficilement envisageable de faire du traitement en temps réel. Nous essayerons néanmoins de faire une estimation de la puissance GPU nécessaire pour que ce soit le cas.

- **Disponibilité et diversité limitées des données vidéo**
Les vidéos actuellement disponibles pour le projet ne couvrent qu’un nombre restreint de situations de vol.
Pour améliorer la robustesse et la généralisation du modèle, il est préferable de collecter ou rechercher des séquences supplémentaires de drones dans des contextes variés (angles, environnements, conditions lumineuses, tailles, vitesses). Si une base de données de vidéo plus large avait été mise à disposition, l'entraînement aurait été plus spécifique et plus efficace.

## Description du projet

### Description de la solution proposée long terme
La solution visée repose sur un modèle de vision par ordinateur basé sur des réseaux de neurones convolutifs (type YOLO), combiné à une approche d’analyse de la dynamique de la scène pour améliorer la robustesse du système.
L’idée est d’intégrer plusieurs couches d’analyse — détection, classification, suivi temporel et contextualisation — afin de permettre une reconnaissance stable, rapide et adaptable à différents types de caméras et de conditions (lumière, météo, mouvement de caméra, etc.).

En parallèle, un traitement du fond statique (via soustraction ou effacement de fond) permettrait d’isoler plus efficacement les objets en mouvement, en ne traitant que les régions d’intérêt. L'objectif principal est d'améliorer la vitesse de détection de l'algorithme et de minimiser l'erreur possible du système.

### Description du prototype qui sera réalisé

Étant donné le temps limité, le prototype visera à valider la faisabilité de la détection automatique de drones à partir d’un flux vidéo.
Nous utiliserons un modèle YOLO pré-entraîné pour effectuer la détection d’objets, en se concentrant sur la reconnaissance des drones à partir des vidéos fournies.

Nous expérimenterons également plusieurs approches de perfectionnement :

- un algorithme simple de soustraction de fond permettra d’identifier les zones en mouvement dans la scène ;
- un tracking pour améliorer la cohérence du suivi ;
- un entraînement de YOLO sur un dataset spécifique au drone utilisé.

L’objectif du prototype est donc d’obtenir une première preuve de concept démontrant la détection de drones dans un contexte simplifié, ouvrant la voie à un développement plus robuste et en temps réel.
