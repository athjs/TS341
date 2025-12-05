# Rapport Drones

## Les différentes technologies

### Règles de bonnes pratiques

Dans le projet était demandé de respecter certaines règles de développement. Pour la gestion de des librairies nous avons utilisé la librairie `Poetry` afin d'avoir une maitrîse sur les différentes versions de chaque librairie.
Avec `Poetry`, les outils suivants on été utilisés:
- `Logging` : Pour la gestion des logs des fonctions
- `Pyright` : Pour la vérification du typing

Ont été ajoutées d'autres technologies afin d'améliorer l'environnement de travail telle qu'une **CI/CD** qui permet de limiter les actions faites sur Github. Par exemple empêcher des **Pull Request** qui ne respectent pas les règles telles que le non typage ou la Docstring.

Les différentes librairies utilisées pour de telles actions sont :
- `Ruff` : Permettant de gérer à la fois le **Linting** (Black) et la vérification de la **Docstring**
- `Pre-commit`: Permettant d'instaurer des hooks garantissant le respect les règles avant un commit

L'avantage de la configuration actuelle est que toutes les options de ces différents outils peuvent être directement spécifiées dans le fichier **pyproject.toml**. Cette particularité permet de donner à tout le monde la même configuration.


### Evaluation de la précision de nos réalisations

Une première chose à faire est d'établir une métrique pour évaluer nos modèles et savoir dans quelle direction aller.
Pour cela, 2 métriques sont de mises :
* Matrice de confusion : Evaluation binaire à partir d'une tolérance (en pixel) par rapport à la vraie valeur,
* Score : Evaluation plus précises, calculée à partir de la distance entre la valeur prédite et la vraie valeur.

Ce qui est considéré comme une "vraie" valeur est l'étiquettage d'une vidéo de référence à la main. Avec une certaine tolérance, cette évaluation humaine est jusqu'à maintenant la plus précise, nécessitant parfois le sens commun plutôt que la vue. Les faux négatifs sont donc courants lorsque le drône est presque indétectable.

### YOLO et son optimisation pour notre usage
Yolo est un élément centrale dans la détection du drône dans l'image. Les poids de bases n'ont pas pour vocations de détecter une classe drône. Il est donc nécessaire de l'entraîner, plus particulièrement son CNN, dans l'objectif de détecter de manière efficace le drône. 3 datasets ont été essayés.

#### Entraînement sur des images de drône aléatoires
Avantage(s) : Capable de détecter plusieurs types de drônes. Base de données avec Kaggle retrouvée, dont l'algorithme (mettre lien).
Inconvénient(s) : Pas spécialisé pour le modèle robot connu à l'avance. Dataset limité en nombre de valeur et d'angles/distance du robot.

#### Modèle simulé
Une technique que nous souhaitions utilisé est une approche sim2real, c'est-à-dire réaliser une simulation informatique assez proche de la réalité pour pouvoir l'utiliser afin de détecter de réelles images de drône.
Pour cela, nous avons utilisé Blender API (Python), afin de :
* Ouvrir une scène blender contenant une caméra et le modèle de drône
* Déplacer le modèle de drône dans le champs de vision de la caméra
* Effectuer un rendu en conservant les coordonnées et la bounding box

Avantage(s) : Possibilité d'avoir un dataset aussi grand que nous le souhaitons, avec des angles et des distances paramètrables.
Inconvénient(s) : La simulation se doit d'être le plus réaliste. Ce qui peut demander l'achat d'un modèle 3D, en plus du coût temporel pour la génération du dataset.

Pour un même dataset de rendus, nous extrayons 2 datasets fonctionnels :
* Un premier dataset avec fond uni et variable (Jaune, Rouge ....etc)
* Un second dataset avec des images de natures en arrière-plan parmi des images récupérées sur internet principalement.

L'objectif est de comparer les résultats et obtenir le meilleur modèle YOLO possible. Mais pour cela nous devons fixer certains paramètres tels que la probabilité minimale que YOLO prédit afin de prendre en compte les éléments qui sont les plus sûrs (ou pas)
Nous posons l'hypothèse que cette valeur ne dépend pas du dataset d'entraînement. L'objectif est de trouver la valeur qui contrebalance le mieux les faux négatifs et vrais positifs.

![](https://codimd.math.cnrs.fr/uploads/upload_1e18ff16d36bfaf7ec3f1e19afbdfc45.png)
![](https://codimd.math.cnrs.fr/uploads/upload_c996324bb3d81ab8649508c636a5fd7e.png)


Comme nous pouvons le voir, et en prenant en compte le cahier des charges qui spécifie la minimisation du nombre de faux négatif et la minimisation du score (ce dernier étant une pénalité), la probabilité minimale choisie pour prendre en compte la prédiction du modèle YOLO est de 0.1 %.
Si la sécurité était de mise et que le nombre de faux positifs comptait, nous serions partis sur 0.5.

#### Résultats et conclusion sur le modèle YOLO

##### Butterworth
Score : 43090.0
Matrix : [ 665  885 1629    0]

##### Yolo seul (Kaggle)
Score : 39110.0
Matrix : [714 836 543 609]

Nous utiliserons pour la suite le modèle yolo entraîné sur un dataset <...>

### OpenCV: la solution alternative

La bibliothèque **OpenCV** permet le traitement d'images. Après sa présentation lors du cours, il est apparu naturel de l'utiliser afin de répondre aux exigences du projet.

#### La reconnaissance de contours

L'utilisation d'OpenCV a été de trouver les objets en mouvement afin de trouver le drone étant un objet en mouvement ou s'il est assez proche, ses hélices sont visibles et en mouvement. Le traitement de la vidéo a été simplement d'enlever le bruit, donc les objets immobiles et avec un seuillage assez élevé pour supprimer les objets très peu mobile et le bruit. Ensuite a été fait simplement une reconaissance de contours entre les objets en mouvements (blancs) et le reste (partie noire). Enfin, a été récupéré chaque objets pour en trouver le centroïde. La liste des centroïdes est renvoyé pour qu'elle soit utilisée par le filtre permettant de retrouver le drone.

#### La nécessité de cette méthode

Il a été choisi d'utiliser plusieurs méthode de reconaissances de par le fait de la limite de Yolo. En effet, Yolo étant entrainer sur une simulation Blender, s'est entraîné avec le drone en haute résolution. Dans le cas de notre utilisation, la résolution de la caméra bien qu'en étant correcte, ne peut pas avoir une résolution idéale sur le drone lorque celu-ci monte en altitude. Par conséquent, le modèle a tendance à ne plus reconnaître ce dernier "assez rapidement" lorsqu'il s'éloigne. Toutefois, OpenCV permet d'observer les mouvements des objets petite taille. C'est pour cela, qu'a été choisie la solution de l'utiliser. Il est possible de suivre un objet se déplaçant même s'il est de petite taille. Ensuite, avec le passage dans le filtre, il est possible de récupérer la coordonnée du drone.

### Le filtre : Sélection du modèle adapté

Deux méthodes sont actuellement utilisées pour détecter le drone. Chacune possède ses inconvénients et ses avantages. Pour une détection optimale, nous devons donc nous appuyer soit sur la méthode YOLO, soit sur la méthode OpenCV, en fonction de la cohérence de chacune de ces deux approches à chaque image.

La méthode YOLO a une excellente capacité à détecter le drone lorsqu’il est proche de la caméra et bien discernable de son environnement.

La méthode OpenCV peut détecter le mouvement du drone même lorsqu’il est trop éloigné pour YOLO. Cependant, elle détecte également tout autre mouvement dans l’image. Il faut donc choisir quel mouvement correspond réellement au drone.

Pour une détection optimale, nous devons être capables, à chaque instant, de :
- Détecter si la détection YOLO est cohérente
- Déterminer quel centroïde de la détection OpenCV est le plus cohérent
- Faire un choix sur la position réelle du drone

#### Le filtre de Butterworth

Nous sommes partis du principe que le drone ne pouvait pas se téléporter dans l’image. Ainsi, tout mouvement trop rapide provenant d’un des outils de détection correspondrait à une mesure incohérente. De plus, pour modéliser la position du drone à chaque instant, nous avons utilisé deux filtres de Butterworth (un pour l’axe x et un pour l’axe y). L’avantage de ce filtre passe-bas est qu’il limite toute accélération trop brutale selon les deux axes. Pour le faire fonctionner en temps réel, nous ajoutons simplement à chaque instant une valeur issue de YOLO ou d’OpenCV en entrée du filtre.

#### Choix des valeurs YOLO ou OpenCV

Une première détection YOLO est nécessaire pour initialiser le filtre Butterworth et la détection OpenCV. En effet, la cohérence des valeurs OpenCV étant évaluée en fonction de la distance à la détection précédente, OpenCV est incapable de choisir quelle zone de mouvement est la plus cohérente lors de son initialisation.

YOLO est toujours priorisé car c’est l’algorithme le plus fiable. Une détection YOLO permet de mettre à jour le filtre Butterworth et de sélectionner la zone de mouvement la plus proche de la position supposée du drone.

Si YOLO fournit une position trop éloignée de celle estimée par Butterworth à l’instant précédent, alors cette position YOLO n’est pas prise en compte. Dans ce cas, OpenCV suit la zone de mouvement précédemment sélectionnée, et le filtre de Butterworth prend les valeurs d’OpenCV.

![Image du filtre](img/Schema_filtre.png)

## Perspectives d’améliorations

Dans le cadre de ce projet, plusieurs éléments auraient pu être améliorés, autant sur le plan technique que sur la solution apportée.

## Qualité de la solution

La solution proposée n’est pas optimale : le drone est détecté trop tardivement par rapport au flux vidéo. Deux raisons principales l’expliquent :
- Le drone ne peut pas être reconnu immédiatement par OpenCV et doit être identifié une première fois par YOLO.
- Le drone n’est pas reconnu lorsqu’il est trop éloigné.

Plusieurs pistes d’amélioration sont envisageables. Un entraînement de modèle dédié à la reconnaissance de contours serait pertinent, tirant parti de la capacité d’OpenCV à isoler les objets en mouvement même avec une faible résolution. Cette approche ne permettrait pas une détection à très longue distance, mais offrirait une marge de manœuvre supérieure à celle de YOLO.

Une autre possibilité serait une approche basée sur la dynamique : un objet mobile détecté par OpenCV mais quasi immobile en position sur plusieurs frames pourrait être considéré comme un drone. Cette méthode permettrait d’améliorer la détection à distance.

## Qualité de la technique

### Qualité et tests

La solution, bien que fonctionnelle, n’est pas optimisée en complexité temporelle et spatiale. Le dépôt GitHub en témoigne par sa taille importante. Certaines fonctions volumineuses pourraient être découpées en modules plus petits et réutilisables.

Aucun test unitaire n’a été réalisé. Malgré la facilité de tester le système visuellement, des tests unitaires simples et des tests de limites auraient permis de garantir davantage la robustesse du code. `pytest` est présent dans le projet et dans la CI/CD, mais seulement en option. Quelques tests d’intégration ont été mis en place avec des métriques associées, mais une partie du code reste non testée.

### CI/CD

La CI/CD pourrait également être renforcée. Il aurait été possible d’imposer des règles plus strictes, comme un typage plus rigoureux ou l’obligation d’une revue par un pair lors d’une Pull Request. Néanmoins, il a semblé préférable de ne pas surcharger l’environnement pour un projet de cette taille et de ne pas freiner la contribution.

### Architecture

L’architecture du projet pourrait être améliorée. Certains fichiers ne sont pas placés de manière optimale dans l’arborescence, et certaines fonctions gagneraient à être mieux découpées, comme mentionné précédemment. Une phase de conception préalable, incluant éventuellement la recherche d’un design pattern adapté, aurait permis d’obtenir une structure plus cohérente.
