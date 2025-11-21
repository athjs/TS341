# blender_scene_render.py
# Script compatible Blender 2.80+
# Crée une scène simple (sol + objet), matériaux, lumière, caméra, et effectue un rendu PNG.

import bpy
import os
from mathutils import Vector
import math

# ---------- Config ----------
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), "blender_render.png")  # changez le chemin si besoin
RENDER_ENGINE = 'CYCLES'   # 'CYCLES' ou 'BLENDER_EEVEE'
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
SAMPLES = 64               # pour Cycles
USE_DENOISING = True
# ----------------------------

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)

def create_subject(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier STL introuvable : {path}")

    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_mesh.stl(filepath=path)
    obj = bpy.context.selected_objects[0]

    obj.name = "Drone"

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    obj.scale = (1, 1, 1)

    bpy.ops.object.shade_smooth()

    return obj


def make_material_principled(name, base_color=(0.8, 0.2, 0.1, 1.0), metallic=0.0, roughness=0.35):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # nettoie nodes par défaut
    for n in nodes:
        nodes.remove(n)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = base_color
    principled.inputs['Metallic'].default_value = metallic
    principled.inputs['Roughness'].default_value = roughness

    principled.location = (-200, 0)
    output.location = (200, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat

def apply_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def add_lighting():
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(45))

    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
    area = bpy.context.active_object
    area.data.size = 2.0
    area.data.energy = 200.0

def add_camera(target=Vector((0,0,1.0)), distance=4.0, elevation_deg=20.0, azimuth_deg=-30.0):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = target.x + distance * math.cos(el) * math.cos(az)
    y = target.y + distance * math.cos(el) * math.sin(az)
    z = target.z + distance * math.sin(el)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.name = "Camera"

    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')  
    cam.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam
    return cam

def setup_scene_settings():
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE

    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100

    if RENDER_ENGINE == 'CYCLES':
        scene.cycles.samples = SAMPLES
        # device (auto GPU/CPU) - user can change in prefs
        try:
            scene.cycles.device = 'GPU'
        except Exception:
            pass
        # denoiser
        if USE_DENOISING:
            scene.cycles.use_denoising = True
            # For Blender 3.x+ there is a denoiser node setting possibility; simple flag used here.
    else:
        # EEVEE settings
        scene.eevee.taa_render_samples = 64

    # Color management (filmic)
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium Contrast'

def set_world_light(strength=0.6, color=(1.0, 1.0, 1.0)):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # clear existing nodes
    for n in nodes:
        nodes.remove(n)

    background = nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    background.inputs['Strength'].default_value = strength

    output = nodes.new(type='ShaderNodeOutputWorld')
    links.new(background.outputs['Background'], output.inputs['Surface'])

def render_and_save(filepath):
    # assure le dossier existe
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    bpy.context.scene.render.filepath = filepath
    print("Rendering to:", filepath)
    bpy.ops.render.render(write_still=True)  # blocking call

def main():
    clean_scene()
    bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.stl_format_legacy')
    subject = create_subject("model.stl")

    # Matériaux
    mat_suz = make_material_principled("Drone_Mat", base_color=(0.2, 0.6, 0.9, 1.0), metallic=0.05, roughness=0.25)
    apply_material(subject, mat_suz)

    add_lighting()
    cam = add_camera(target=Vector((0,0,1.0)), distance=4.0, elevation_deg=15.0, azimuth_deg=-35.0)
    setup_scene_settings()
    set_world_light(strength=0.45)

    # Optionnel : activer approx ambient occlusion / shadows selon moteur
    if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.use_gtao = True

    render_and_save(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
