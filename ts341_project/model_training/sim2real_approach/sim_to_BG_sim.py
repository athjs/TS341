"""Ajout d'un arrière-plan en sortir du rendu blender."""

import numpy as np
import PIL.Image as Image
import random


def import_images(n_images):
    """Importe n images depuis le dossier dataset/ et les redimensionne en 1920x1080."""
    image_path = "ts341_project/model_training/sim2real_approach/dataset/"

    images = []
    for i in range(n_images):
        img = Image.open(f"{image_path}image_{i:03d}.png")
        img = img.resize((1920, 1080))
        images.append(np.array(img))

    return images


def RGBA_to_RGB(image):
    """Convertit une image RGBA en RGB en supprimant le canal alpha."""
    return image[:, :, :3]


def import_BG(n_BG):
    """Importe n images de fond depuis le dossier BG_dataset/ et les redimensionne en 1920x1080."""
    BG_path = "ts341_project/model_training/sim2real_approach/nature_BG_dataset/"

    BG_images = []
    for i in range(n_BG):
        img = Image.open(f"{BG_path}{i+1}.jpg")
        img = img.resize((1920, 1080))
        BG_images.append(np.array(img))

    return BG_images


def get_random_BG(BG_images):
    """Sélectionne une image de fond aléatoire parmi une liste d'images."""
    return random.choice(BG_images)


def replace_with_nature_BG(image, BG_images):
    """Remplace les pixels transparents d'une image par des pixels d'une image de fond aléatoire."""
    # replace empty pixel with random BG
    BG_image = get_random_BG(BG_images)
    mask = image[..., 3] == 0
    image = RGBA_to_RGB(image)
    image[mask] = BG_image[mask]
    return image

def replace_with_color_BG(image):
    random_color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    mask = image[..., 3] == 0
    image = RGBA_to_RGB(image)
    image[mask] = random_color
    return image


def __main__():
    """Point d'entrée principal."""
    n_images = 5
    n_BG = 103

    images = import_images(n_images)
    BG_images = import_BG(n_BG)

    for i in range(n_images):
        image_with_nature_BG = replace_with_nature_BG(images[i], BG_images)
        img = Image.fromarray(image_with_nature_BG)
        img.save(f"ts341_project/model_training/sim2real_approach/final_nature_dataset/{i}.jpg")
        image_with_color_BG = replace_with_color_BG(images[i])
        img = Image.fromarray(image_with_color_BG)
        img.save(f"ts341_project/model_training/sim2real_approach/final_color_dataset/{i}.jpg")

__main__()
