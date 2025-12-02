"""Création de dataset à partir de blender."""
# blender_scene_render.py
# Script compatible Blender 2.80+
# Crée une scène simple (sol + objet), matériaux, lumière, caméra, et effectue un rendu PNG.
# A executer avec la commande suivante :
# blender --background --python ts341_project/dataset_create_from_file.py

import bpy
import os
import mathutils
import math
import random
from typing import Tuple

# ---------- Config ----------
RENDER_ENGINE: str = "CYCLES"  # 'CYCLES' ou 'BLENDER_EEVEE'
RESOLUTION_X: int = 1920
RESOLUTION_Y: int = 1080
SAMPLES: int = 64  # pour Cycles
USE_DENOISING: bool = True
# ----------------------------


def render_and_save(filepath: str) -> None:
    """Configure le moteur de rendu et lance le rendu."""
    # assure le dossier existe
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    bpy.context.scene.render.filepath = filepath
    print("Rendering to:", filepath)
    bpy.ops.render.render(write_still=True)  # blocking call


def drone_space_to_camera_space(camera_obj: bpy.types.Object, FOV: float, borne_dist: Tuple[float, float]) -> mathutils.Vector:
    """Convertit une position aléatoire dans l'espace drone en position dans l'espace monde."""
    ratio: float = RESOLUTION_Y / RESOLUTION_X
    # Point dans l'espace caméra
    FOV_X: float = math.radians(34)
    FOV_Y: float = math.atan(math.tan(FOV_X / 2) * ratio) * 2

    pente_x: float = math.tan(FOV_X / 2)
    pente_y: float = math.tan(FOV_Y / 2)
    print(pente_x, pente_y)

    alea_x: float = random.uniform(-1, 1)
    alea_y: float = random.uniform(-1, 1)
    x_pixel: float = (alea_x + 1) * 0.5 
    y_pixel: float = 1 - (alea_y + 1) * 0.5 

    alea_z: float = random.uniform(borne_dist[0], borne_dist[1])
    alea_x_space: float = alea_x * pente_x * alea_z
    alea_y_space: float = alea_y * pente_y * alea_z
    target_point: float = mathutils.Vector(
        (alea_x_space, alea_y_space, -alea_z)
    )  # Z négatif = devant la caméra

    # Conversion espace caméra → espace monde
    point_world: mathutils.Vector = camera_obj.matrix_world @ target_point

    scale_factor: float = (borne_dist[1] - alea_z)/borne_dist[1]
    width: float= 0.2* scale_factor
    height: float = 0.2* scale_factor
    return (point_world, (x_pixel, y_pixel, width, height))


def move_scene(drone_obj: bpy.types.Object, borne_dist: Tuple[float, float], theta_bornes: Tuple[float, float]) -> list:
    """Déplace la scène (drone + caméra) à une nouvelle position aléatoire."""
    # RAJOUTER LE POV

    alea_theta: float = random.uniform(theta_bornes[0], theta_bornes[1])
    camera_obj: bpy.types.Object = bpy.data.objects["Camera"]
    camera_obj.rotation_euler = (math.radians(90 + alea_theta), 0, 0)
    move_infos = drone_space_to_camera_space(
        camera_obj, math.radians(60), borne_dist
    )
    work_obj_coos = move_infos[0]
    drone_obj.location = work_obj_coos

    return move_infos[1]


def give_output(n: int, drone_obj: bpy.types.Object, theta_bornes: Tuple[float, float]) -> None:
    """Génère et sauvegarde n images de la scène avec des positions aléatoires."""
    
    for i in range(n):
        print("Rendering image", i + 1, "/", n)
        label: list[float] = move_scene(drone_obj, (20, 100), theta_bornes)

        # On export l'image
        render_and_save(f"ts341_project/model_training/sim2real_approach/dataset/images/image_{i:03d}.png")

        # On export le label correspondant
        with open(f"ts341_project/model_training/sim2real_approach/dataset/labels/image_{i:03d}.txt", "w") as f :
            f.write(f"0 {label[0]} {label[1]} {label[2]} {label[3]}")
        


def main() -> None:
    """Point d'entrée principal."""
    bpy.ops.wm.open_mainfile(filepath="ts341_project/model_training/sim2real_approach/base.blend")
    drone: bpy.types.Object = bpy.data.objects.get("model")
    if drone is None:
        raise ValueError("L'objet 'model' n'existe pas dans la scène Blender.")
    give_output(200, drone, (0, 0))

    print("Done.")

    # bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.stl_format_legacy')

    """if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.use_gtao = True """


if __name__ == "__main__":
    main()