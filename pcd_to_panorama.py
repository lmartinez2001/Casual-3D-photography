import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.parsers import compute_rotation_matrix

# ==============================
# Configuration
# ==============================

dataset = "boardgames_mobile"
main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
N =  218
trunc = 12500.
scale = 0.12

center = np.array([0.117654, 0.0759836, -0.141303])*scale
panoUp = np.array([-0.933171, 0.0182682, 0.358972])
panoForward = np.array([-0.342018, 0.261994, -0.90243])
R = compute_rotation_matrix(panoForward, panoUp)

pcd_dir = main_dir + "generated_pcd/"

subsample = True
if subsample :
    pcd_dir = main_dir + "generated_sub_pcd/"

panorama_width = 2048#//2  # Output width
panorama_height = 1024# //2 # Output height
use_z_test = True  # Enable depth-based occlusion handling

# Initialize panorama and depth buffer
panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
depth_buffer = np.full((panorama_height, panorama_width),trunc)  # Z-buffer

# ==============================
# Load Precomputed 3D Points
# ==============================
# Assume 'points_3d' is a NumPy array of shape (N, 3)
# Assume 'colors' is a NumPy array of shape (N, 3) with values in [0,1]
# Example: Load from a PLY file (modify as needed)
for k in tqdm(range(N)):
    #print(k)
    pcd = o3d.t.io.read_point_cloud(f"{pcd_dir}/img_to_pcd_{k}.ply") 
    pcd.rotate(R, center)  
    points_3d = pcd.point.positions.numpy()  # Shape (N, 3)
    colors = pcd.point.colors.numpy()  # Shape (N, 3)

    # ==============================
    # Projection and Mapping
    X, Y, Z = (points_3d - center).T  # Extract coordinates

    r = np.linalg.norm(points_3d, axis=1)  # Compute radius in one go
    theta = np.arctan2(X, Z)  # Longitude
    phi = np.arcsin(Y / r)  # Latitude
    
    pano_x = ((theta + np.pi) / (2 * np.pi) * panorama_width).astype(int)
    pano_y = ((phi + (np.pi / 2)) / np.pi * panorama_height).astype(int)


    # ==============================
    # Efficiently Filter Valid Indices
    # ==============================
    valid_mask = (0 <= pano_x) & (pano_x < panorama_width) & (0 <= pano_y) & (pano_y < panorama_height)
    
    pano_x = pano_x[valid_mask]
    pano_y = pano_y[valid_mask]
    r = r[valid_mask]
    color_vals = (colors[valid_mask] * 255).astype(np.uint8)

    # ==============================
    # Z-Buffer Depth Test (Optimized)
    # ==============================
    if use_z_test:
        update_mask = r < depth_buffer[pano_y, pano_x]
        depth_buffer[pano_y[update_mask], pano_x[update_mask]] = r[update_mask]
        panorama[pano_y[update_mask], pano_x[update_mask]] = color_vals[update_mask]
    else:
        panorama[pano_y, pano_x] = color_vals
    #plt.imsave(f"panorama_open3d_{k}.png", panorama)

# ==============================
# Save and Display Panorama
# ==============================
plt.imsave("panorama_rgb.png", panorama)
plt.imsave("panorama_d.png",np.log(depth_buffer))
