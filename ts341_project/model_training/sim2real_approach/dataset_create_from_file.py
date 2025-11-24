"""Création de dataset à partir de blender."""
# blender_scene_render.py
# Script compatible Blender 2.80+
# Crée une scène simple (sol + objet), matériaux, lumière, caméra, et effectue un rendu PNG.

import bpy
import os
import mathutils
import math
import random

# ---------- Config ----------
RENDER_ENGINE = "CYCLES"  # 'CYCLES' ou 'BLENDER_EEVEE'
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
SAMPLES = 64  # pour Cycles
USE_DENOISING = True
# ----------------------------


def render_and_save(filepath):
    """Configure le moteur de rendu et lance le rendu."""
    # assure le dossier existe
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    bpy.context.scene.render.filepath = filepath
    print("Rendering to:", filepath)
    bpy.ops.render.render(write_still=True)  # blocking call


def drone_space_to_camera_space(camera_obj, FOV, borne_dist):
    """Convertit une position aléatoire dans l'espace drone en position dans l'espace monde."""
    ratio = 1080 / 1920  # hauteur / largeur
    # Point dans l'espace caméra
    FOV_X = math.radians(34)
    FOV_Y = math.atan(math.tan(FOV_X / 2) * ratio) * 2

    pente_x = math.tan(FOV_X / 2)
    pente_y = math.tan(FOV_Y / 2)
    print(pente_x, pente_y)

    alea_z = random.uniform(borne_dist[0], borne_dist[1])
    alea_x = random.uniform(-1, 1) * pente_x * alea_z
    alea_y = random.uniform(-1, 1) * pente_y * alea_z
    target_point = mathutils.Vector(
        (alea_x, alea_y, -alea_z)
    )  # Z négatif = devant la caméra
    print(
        "Pente_x:",
        pente_x,
        "Pente_y:",
        pente_y,
        "x:",
        alea_x,
        "y:",
        alea_y,
        "z:",
        alea_z,
    )

    # Conversion espace caméra → espace monde
    point_world = camera_obj.matrix_world @ target_point

    return point_world


def move_scene(drone_obj, borne_dist, theta_bornes):
    """Déplace la scène (drone + caméra) à une nouvelle position aléatoire dans la zone visible du drone."""
    # RAJOUTER LE FOV

    # D'abord theta pour l'angle de la caméra
    alea_theta = random.uniform(theta_bornes[0], theta_bornes[1])
    camera_obj = bpy.data.objects["Camera"]
    camera_obj.rotation_euler = (math.radians(90 + alea_theta), 0, 0)
    work_obj_coos = drone_space_to_camera_space(
        camera_obj, math.radians(60), borne_dist
    )
    drone_obj.location = work_obj_coos


def give_output(n, drone_obj, theta_bornes):
    """Génère et sauvegarde n images de la scène avec des positions aléatoires."""
    for i in range(5):
        print("Rendering image", i + 1, "/", n)
        move_scene(drone_obj, (20, 100), theta_bornes)
        render_and_save(f"dataset/image_{i:03d}.png")


def main():
    """Point d'entrée principal."""
    bpy.ops.wm.open_mainfile(filepath="base.blend")
    drone = bpy.data.objects.get("model")
    give_output(5, drone, (0, 0))

    # bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.stl_format_legacy')

    """if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.use_gtao = True """

    print("Done.")


if __name__ == "__main__":
    main()
