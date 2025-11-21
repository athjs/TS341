import numpy as np
import PIL.Image as Image
import random

def import_images(n_images):
    image_path = 'dataset/'

    images = []
    for i in range(n_images):
        img = Image.open(f'{image_path}image_{i:03d}.png')
        img = img.resize((1920, 1080))
        images.append(np.array(img))

    return images

def RGBA_to_RGB(image):
    return image[:, :, :3]

def import_BG(n_BG):
    BG_path = 'BG_dataset/'

    BG_images = []
    for i in range(n_BG):
        img = Image.open(f'{BG_path}BG_{i}.jpg')
        img = img.resize((1920, 1080))
        BG_images.append(np.array(img))

    return BG_images

def get_random_BG(BG_images):
    return random.choice(BG_images)

def replace_BG(image, BG_images):
    #replace empty pixel with random BG
    BG_image = get_random_BG(BG_images)
    mask = image[..., 3] == 0
    image = RGBA_to_RGB(image)
    image[mask] = BG_image[mask]
    return image

def __main__():
    n_images = 5
    n_BG = 1

    images = import_images(n_images)
    BG_images = import_BG(n_BG)

    for i in range(n_images):
        image_with_BG = replace_BG(images[i], BG_images)
        img = Image.fromarray(image_with_BG)
        img.save(f'final_dataset/final_{i}.jpg')

__main__()