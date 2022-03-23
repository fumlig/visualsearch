import cv2 as cv
import numpy as np
import pyrender
from pydelatin import Delatin
import trimesh
from gym_search.generators import TerrainGenerator
from gym_search.palette import pick_color, EARTH_TOON

generator = TerrainGenerator((256, 256), 3)
terrain = generator.terrain(0).astype(np.float32)

amplitude = 50

tin = Delatin(terrain*amplitude)
colors = pick_color(tin.vertices, EARTH_TOON)

print(colors)


#mesh = Mesh(tin.vertices, [("triangle", tin.triangles)])
tm = trimesh.Trimesh(tin.vertices, tin.triangles)
tm.visual.face_colors = trimesh.visual.color.vertex_to_face_color(colors, tin.triangles)

mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
# smooth=False)

# compose scene
scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02, 1.0], bg_color=[0, 0, 0])
camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)

scene.add(mesh, pose=np.eye(4))
#scene.add(camera)

# render scene
#r = pyrender.OffscreenRenderer(256, 256)
#color, _ = r.render(scene)

#plt.figure(figsize=(8,8)), plt.imshow(color)

pyrender.Viewer(scene)

# https://pyrender.readthedocs.io/en/latest/_modules/pyrender/viewer.html