# TS341

Projet Outil d'Imagerie pour l'Informatique
Ce projet sera sur les drones

Model de drône : Mavic pro 4.0
Vitesse max envisageable : 100 km / h

## Environnement de travail

Dans ce projet nous travaillerons avec un Docker dont l'image est créée par nos soins à partir de l'image de python@3.13. Pour l'environnement de travail python
sera utilisé poetry afin de gérer au mieux le versionning.

## 📌 Installation

### Prérequis
- Python **≥ 3.13**
- Poetry **≥ 1.8**
- Docker (optionnel, pour l'exécution conteneurisée)
- Un GPU *n’est pas nécessaire* pour utiliser le projet, mais accélère YOLO.

### Installation via Poetry

```bash
git clone <url-du-repo>
cd TS341
poetry install
```

La commande pour créer et lancer le Docker : 
```bash
docker build -t mon-app .
docker run --rm -p 8080:5000 \
  -v $(pwd)/videos:/app/videos \
  mon-app video2_short
```
Cependant, il y a un problème avec la vidéo, donc pour lancer le projet en local, il est nécessaire de rentrer la commande suivante: 
```bash
poetry run python ts341_project/filtre/filtre.py video
``
#### Ressources utilisées
- [delpeuch.net](https://delpeuch.net/blog)
- [GitHub Doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks?utm_source=chatgpt.com)
