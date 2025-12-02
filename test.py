# FONCTION GENEREE AVEC CHATGPT EN GRANDE PARTIE

import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def afficher_image_labels(dataset_path, image_name):
    """Affichage de chaque image pour vérifier la validité de la labélisation du fichier "simulate_blender_images.py".
    dataset_path : chemin vers le dataset contenant 'images/' et 'labels/'
    image_name   : nom du fichier image (ex: 'img1.jpg')
    """
    dataset_path = Path(dataset_path)
    image_path = dataset_path / "images" / image_name
    label_path = dataset_path / "labels" / (image_path.stem + ".txt")

    # 1️⃣ Lire l'image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")

    height, width = img.shape[:2]

    # 2️⃣ Lire les labels YOLO
    if not label_path.exists():
        print("Aucun label pour cette image")
        labels = []
    else:
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                # YOLO format: class x_center y_center width height (tout normalisé 0..1)
                cls, x_center, y_center, w, h = map(float, line.strip().split())
                # Convertir en pixels
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)
                labels.append((cls, x1, y1, x2, y2))

    # 3️⃣ Dessiner les rectangles
    for cls, x1, y1, x2, y2 in labels:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(int(cls)),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # 4️⃣ Convertir BGR → RGB pour affichage matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5️⃣ Afficher
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


for i in range(5):
    afficher_image_labels(
        "ts341_project/model_training/sim2real_approach/dataset", f"image_{i:03d}.png"
    )
