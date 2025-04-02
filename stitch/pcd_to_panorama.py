import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import meshio,os

from utils.parsers import compute_rotation_matrix

# ==============================
# Configuration
# ==============================

dataset = "creepyattic"
main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
N = 50

### Values from param.txt
minD = 3.3333333333333335
trunc = 10000.0
scale = 0.15
factor = (2**16-1)
center = np.array([0.0886422, 0.00969138, 0.0281881])*scale
panoForward = np.array([-0.090604, 0.75476, -0.649714])
panoUp = np.array([0.223168, 0.651192, 0.725356])
R = compute_rotation_matrix(panoForward, panoUp)

##
pcd_dir = main_dir + "generated_pcd/"

subsample = True
if subsample :
    pcd_dir = main_dir + "generated_sub_pcd/"

subsample_with_normals = False
if subsample_with_normals :
    pcd_dir = main_dir + "generated_sub_pcd_normals/"

panorama_width = 2048//2# Output width
panorama_height = 1024//2 # Output height
use_z_test = 1  # Enable depth-based occlusion handling
save_s = False
# Initialize panorama and depth buffer
panorama_front = np.full((panorama_height, panorama_width, 3), np.nan)#,dtype=np.uint8)
panorama_back = np.full((panorama_height, panorama_width, 3), np.nan)#,dtype=np.uint8)
front_buffer = np.full((panorama_height, panorama_width),np.nan)  # Z-buffer
back_buffer = np.full((panorama_height, panorama_width),np.nan)
def cartesian_to_spherical(points_3d):
    X, Y, Z = points_3d.T
    r = np.linalg.norm(points_3d, axis=1)
    theta = np.arcsin(Z / r)  # Latitude [-π/2, π/2]
    phi = np.arctan2(Y, X)  # Longitude [-π, π]
    return r, theta, phi

def spherical_to_equirectangular(theta, phi, panorama_width, panorama_height):
    pano_x = (((phi) / (2 * np.pi) * panorama_width)%panorama_width).astype(int)
    pano_y = (((0.5 - theta / np.pi) * panorama_height)%panorama_height).astype(int)
    return pano_x, pano_y

def cartesian_to_equirectangular(points_3d, panorama_width, panorama_height):
    r, theta, phi = cartesian_to_spherical(points_3d)
    pano_x, pano_y = spherical_to_equirectangular(theta, phi, panorama_width, panorama_height)
    return pano_x, pano_y,r

def stretch_penalty(normals,view_dirs,tresh_angle = 1.66):
    dot_product = np.abs(np.sum(normals * view_dirs, axis=1))
    incidence_angle = np.arccos(dot_product)
    grazing_angle = 90 - np.degrees(incidence_angle)
    s = 1 - np.clip((grazing_angle/tresh_angle),a_min = 0,a_max=1)
    return s

def compute_z_prim(Z,s):
    return ((Z/trunc)+s)/2
# ==============================
# Load Precomputed 3D Points
# ==============================
# Assume 'points_3d' is a NumPy array of shape (N, 3)
# Assume 'colors' is a NumPy array of shape (N, 3) with values in [0,1]
# Example: Load from a PLY file (modify as needed)
rmax = 0

for k in tqdm(range(N)):
    #print(k)

    if subsample_with_normals :
        pcd = meshio.read(f"{pcd_dir}/pcd_{k}.vtk")
        #Compute S before transform
        view_dir = pcd.points-center
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
        s = stretch_penalty(pcd.point_data["Normals"],view_dir,tresh_angle=10)
        pcd.point_data["v"] = view_dir
        pcd.point_data["s"] = s
        if save_s :
            pcd.write(f"{pcd_dir}/pcd_{k}.vtk")
    elif subsample : 
        pcd = meshio.read(f"{pcd_dir}/img_to_pcd_{k}.vtk")

    #Transform to Equirectangle
    pcd.points -= center
    pcd.points = pcd.points@R#.T 
    points_3d = pcd.points  # Shape (N, 3)
    colors = (pcd.point_data["Colors"])#* 255).astype(np.uint8) # Shape (N, 3)
    X, Y, Z = points_3d.T
    pano_x,pano_y,r = cartesian_to_equirectangular(points_3d, panorama_width, panorama_height)
    r, theta, phi = cartesian_to_spherical(points_3d)
    # Compute Z for the test
    if use_z_test > 1 :
        Z_prim = compute_z_prim(r,s)
    elif use_z_test > 0 :
        Z_prim = r/trunc

    pano_coords = np.vstack([pano_x,pano_y])
    unique_coords,unique_map,unique_count = np.unique(pano_coords,axis=1,return_inverse=True,return_counts=True)
    # ==============================
    # Z-Buffer Depth Test (Optimized)
    # ==============================
    if use_z_test > 0 :
        for i in tqdm(range(unique_coords.shape[1])) :
            x,y = unique_coords[:,i]
            candidates = np.argwhere((pano_x==x)&(pano_y==y))
            best = np.argmin(Z_prim[candidates])
            if np.isnan(front_buffer[y,x]) or (Z_prim[candidates[best]]< front_buffer[y,x]) :
                front_buffer[y,x] = Z_prim[candidates[best]][0]
                panorama_front[y,x] = colors[candidates[best]]
                #if (y==50 and x==350) : 
                #    print(X[candidates[best]],Y[candidates[best]],Z[candidates[best]])
                #    print(r[candidates[best]],theta[candidates[best]],phi[candidates[best]])
            best = np.argmax(Z_prim[candidates])
            if np.isnan(back_buffer[y,x]) or (Z_prim[candidates[best]]>back_buffer[y,x]) :
                back_buffer[y,x] = Z_prim[candidates[best]][0]
                panorama_back[y,x] = colors[candidates[best]]
    else:
        panorama_front[pano_y, pano_x] = colors
    #plt.imsave(f"panorama_open3d_{k}.png", panorama)

# ==============================
# Save and Display Panorama
# ==============================
#print(rmax)
save_dir = main_dir + "pano_result/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

np.save(f"{save_dir}panorama_rgb_front.npy", panorama_front)
np.save(f"{save_dir}panorama_rgb_back.npy", panorama_back)
np.save(f"{save_dir}panorama_d_front.npy",front_buffer)
np.save(f"{save_dir}panorama_d_back.npy",back_buffer)

plt.imsave(f"{save_dir}panorama_rgb_front.png", panorama_front)
plt.imsave(f"{save_dir}panorama_rgb_back.png", panorama_back)
plt.imsave(f"{save_dir}panorama_d_front.png",np.nan_to_num(-np.log(front_buffer)))
plt.imsave(f"{save_dir}panorama_d_back.png",np.nan_to_num(-np.log(back_buffer+1)))
