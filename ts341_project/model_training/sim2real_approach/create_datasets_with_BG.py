"""Ajout d'un arrière-plan en sortir du rendu blender."""

import numpy as np
import PIL.Image as Image
import random
from typing import List

import shutil
from pathlib import Path


def update_label(source_path, destination_path, label_name):
    """Copie le label de source_path vers destination_path."""
    dossier_source = Path(source_path)
    dossier_destination = Path(destination_path)

    item = dossier_source / label_name
    if not item.exists():
        print(f"Erreur : '{label_name}' n'existe pas dans le dossier source.")
        return

    dest = dossier_destination / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)


def import_images(n_images):
    """Importe n images depuis le dossier dataset/ et les redimensionne en 1920x1080."""
    image_path = "ts341_project/model_training/sim2real_approach/dataset/images"

    images: List[np.ndarray] = []
    for i in range(n_images):
        img: Image.Image = Image.open(f"{image_path}/image_{i:03d}.png")
        img = img.resize((1920, 1080))
        images.append(np.array(img))

    return images


def RGBA_to_RGB(image) -> np.ndarray:
    """Convertit une image RGBA en RGB en supprimant le canal alpha."""
    return image[:, :, :3]


def import_BG(n_BG: int) -> List[np.ndarray]:
    """Importe n images de fond depuis le dossier BG_dataset/ et les redimensionne en 1920x1080."""
    BG_path: str = "ts341_project/model_training/sim2real_approach/nature_BG_dataset/"

    BG_images: List[np.ndarray] = []
    for i in range(n_BG):
        img: Image.Image = Image.open(f"{BG_path}{i+1}.jpg")
        img = img.resize((1920, 1080))
        BG_images.append(np.array(img))

    return BG_images


def get_random_BG(BG_images: List[np.ndarray]) -> np.ndarray:
    """Sélectionne une image de fond aléatoire parmi une liste d'images."""
    return random.choice(BG_images)


def replace_with_nature_BG(
    image: np.ndarray, BG_images: List[np.ndarray]
) -> np.ndarray:
    """Remplace les pixels transparents d'une image par des pixels d'une image de fond aléatoire."""
    BG_image: np.ndarray = get_random_BG(BG_images)
    mask: np.ndarray = image[..., 3] == 0
    image = RGBA_to_RGB(image)
    image[mask] = BG_image[mask]
    return image


def replace_with_color_BG(image: np.ndarray) -> np.ndarray:
    """Remplace les pixels transparents d'une image par une couleur aléatoire."""
    random_color: List[int] = [
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    ]
    mask: np.ndarray = image[..., 3] == 0
    image = RGBA_to_RGB(image)
    image[mask] = random_color
    return image


def __main__() -> None:
    """Point d'entrée principal."""
    n_images: int = 200
    n_BG: int = 103

    images: List[np.ndarray] = import_images(n_images)
    BG_images: List[np.ndarray] = import_BG(n_BG)

    for i in range(n_images):
        end_path: str
        if i < (0.80 * n_images):
            end_path = "train"
        else:
            end_path = "valid"

        # Nature BG
        image_with_nature_BG: np.ndarray = replace_with_nature_BG(images[i], BG_images)
        img: Image.Image = Image.fromarray(image_with_nature_BG)
        img.save(
            f"ts341_project/model_training/sim2real_approach/final_nature_dataset/{end_path}/images/image_{i:03d}.jpg"
        )
        update_label(
            f"ts341_project/model_training/sim2real_approach/dataset/labels",
            f"ts341_project/model_training/sim2real_approach/final_nature_dataset/{end_path}/labels",
            f"image_{i:03d}.txt",
        )

        # Color BG
        image_with_color_BG = replace_with_color_BG(images[i])
        img = Image.fromarray(image_with_color_BG)
        img.save(
            f"ts341_project/model_training/sim2real_approach/final_color_dataset/{end_path}/images/image_{i:03d}.jpg"
        )
        update_label(
            "ts341_project/model_training/sim2real_approach/dataset/labels",
            f"ts341_project/model_training/sim2real_approach/final_color_dataset/{end_path}/labels",
            f"image_{i:03d}.txt",
        )


__main__()
