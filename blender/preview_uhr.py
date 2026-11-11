import bpy, math
from mathutils import Vector
sc = bpy.context.scene
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.65, 0.72, 0.82, 1); bg.inputs[1].default_value = 0.9
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.0; so = bpy.data.objects.new('sun', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(55), 0, math.radians(200))
bpy.ops.mesh.primitive_plane_add(size=40); g = bpy.context.object
m = bpy.data.materials.new('pave'); m.use_nodes = True
next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.35, 0.34, 0.33, 1); g.data.materials.append(m)
try: sc.render.engine = 'BLENDER_EEVEE'
except TypeError: sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x, sc.render.resolution_y = 900, 1300
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
B = math.radians(33)
def shot(name, dist, h, tz, lens, side=0.0):
    a = B + side
    cam.location = Vector((math.sin(a) * dist, math.cos(a) * dist, h))
    d = Vector((0, 0, tz)) - cam.location; cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = lens
    sc.render.filepath = '//previews/' + name + '.png'; bpy.ops.render.render(write_still=True)
shot('uhr_front', 11, 1.6, 3.0, 40, side=-0.35)
sc.render.resolution_x, sc.render.resolution_y = 1000, 1000
shot('uhr_dial', 2.4, 4.1, 4.1, 50, side=0.0)
shot('uhr_base', 4.0, 1.0, 0.8, 35, side=0.5)
