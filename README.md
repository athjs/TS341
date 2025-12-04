# TS341

Projet Outil d'Imagerie pour l'Informatique
Ce projet sera sur les drones

Model de drône : Mavic pro 4.0
Vitesse max envisageable : 100 km / h

## Environnement de travail

Dans ce projet nous travaillerons avec un Docker dont l'image est créée par nos soins à partir de l'image de python@3.13. Pour l'environnement de travail python
sera utilisé poetry afin de gérer au mieux le versionning.

### Outils pour le code

- Pour le linting est utilisé **Pyright** pour la `CI`, pour le formattage **Ruff** pour `Pre-Commit`
- Pour la propreté du code : **Pre-Commit**
- Pour la vérification du typage : pyright avec sa configuration dans `pyproject.toml`

#### Ressources utilisées
- [delpeuch.net](https://delpeuch.net/blog)
- [GitHub Doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks?utm_source=chatgpt.com)
