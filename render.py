import os
# switch to "osmesa" or "egl" before loading pyrender
#os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
from pydelatin import Delatin
from gym_search.generators import TerrainGenerator

generator = TerrainGenerator((256, 256), 3)
terrain = generator.terrain(0).astype(np.float32)

tin = Delatin(terrain)

# generate mesh
mesh = pyrender.Mesh.from_points(tin.triangles)
# smooth=False)

# compose scene
scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

scene.add(mesh, pose=np.eye(4))
scene.add(light, pose=np.eye(4))
#scene.add(camera)

# render scene
#r = pyrender.OffscreenRenderer(256, 256)
#color, _ = r.render(scene)

#plt.figure(figsize=(8,8)), plt.imshow(color)

pyrender.Viewer(scene)

# https://pyrender.readthedocs.io/en/latest/_modules/pyrender/viewer.html