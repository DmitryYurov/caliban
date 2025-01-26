import blenderproc as bproc

import numpy as np
import os
import bpy
import cv2
import argparse

from blenderproc.python.material import MaterialLoaderUtility

parser = argparse.ArgumentParser()
parser.add_argument("--num_poses", type=int, required=True)
parser.add_argument("--pattern", type=str, default="radon_checkerboard.png")
parser.add_argument("--image_hw", type=tuple, default=(1920, 1080))
parser.add_argument("--pattern_hw", type=float, default=(0.140, 0.200))
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--curvature", type=float, default=0.0)
parser.add_argument("--random_seed", type=int, default=1234)
args = parser.parse_args()

bproc.init()

z_dist_marker = -0.4
pat_h, pat_w = args.pattern_hw

chessboard_texture_path = os.path.join("./", args.pattern)
if not os.path.exists(chessboard_texture_path):
    raise FileNotFoundError("Chessboard image not found. Place chessboard.png in the script directory.")

marker = bproc.object.create_primitive("PLANE", scale=[pat_w / 2., pat_h / 2., 0.01])
material = bproc.material.create("chessboard_material")
material.blender_obj.use_nodes = True 

table = bproc.object.create_primitive("PLANE", scale=[4, 4, 0.01])
mat = MaterialLoaderUtility.create(name="ambient")
            
principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
color = np.array([0.4, 0.4, 0.4, 1.])  
principled_node.inputs["Base Color"].default_value = color

table.replace_materials(mat)
table.set_location([0, 0, z_dist_marker - 1e-4])
                                      
material_nodes = material.blender_obj.node_tree.nodes
texture_node = material_nodes.new(type='ShaderNodeTexImage')
texture_node.image = bpy.data.images.load(chessboard_texture_path)
bsdf_node = material_nodes.get("Principled BSDF")
material.blender_obj.node_tree.links.new(bsdf_node.inputs["Base Color"], texture_node.outputs["Color"])
marker.add_material(material)

fx, fy = 1400, 1400
image_width, image_height = args.image_hw
cx, cy = image_width / 2, image_height / 2
k1, k2, k3 = -0.172992, 0.0248708, 0.00149384
p1, p2 = 0.000311976, -9.62967e-5

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)
mapping_coords = bproc.camera.set_lens_distortion(k1, k2, k3, p1, p2)

cams = []

zero2marker = np.eye(4)
zero2marker[2, 3] = -z_dist_marker

world2zero = np.linalg.inv(zero2marker)

marker.set_location([0, 0, z_dist_marker])  

# set random seed
if args.random_seed is not None and args.random_seed != 0:
    np.random.seed(args.random_seed)

for _ in range(args.num_poses):
    marker_location = np.array([0, 0, z_dist_marker])

    location = np.random.uniform([-0.2, -0.2, -0.05], [0.2, 0.2, 0.05])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(marker_location - location, 
                                                             inplane_rot=np.random.uniform(-0.4, 0.4),
                                                             up_axis='Z')
    # Add homog cam pose based on location an rotation
    cam2zero = bproc.math.build_transformation_mat(location, rotation_matrix)

    add_on_location = np.random.uniform([-0.1, -0.05, -0.03], [0.1, 0.05, 0.03])
    cam2zero = cam2zero @ bproc.math.build_transformation_mat(add_on_location, np.eye(3))

    world2cam = np.linalg.inv(cam2zero) @ world2zero

    cams.append(world2cam)
    bproc.camera.add_camera_pose(cam2zero)

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 0, 4])
light.set_energy(500)

bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.set_noise_threshold(1e-3)

output_dir = os.path.join(".", args.output_dir + "/data")
data = bproc.renderer.render()

os.makedirs(output_dir, exist_ok=True)

np.savetxt(os.path.join(args.output_dir, "K.txt"), K)
np.savetxt(os.path.join(args.output_dir, "dist_coeff.txt"), np.array([k1, k2, p1, p2, k3]))

for key in ['colors']:
    use_interpolation = key == "colors"
    data[key] = bproc.postprocessing.apply_lens_distortion(data[key], mapping_coords, image_width, image_height,
                                                           use_interpolation=use_interpolation)

# the additional transformation is needed to convert from blender to opencv coordinate system
bl_to_cv = np.array([[1., 0, 0, 0, ], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
for i, img in enumerate(data["colors"]):
    save_cam = np.linalg.inv(bl_to_cv) @ cams[i] @ bl_to_cv
    np.savetxt(os.path.join(output_dir, f"cam_pose_{i}.txt"), save_cam)
    cv2.imwrite(os.path.join(output_dir, f"image_{i:03d}.png"), img)
