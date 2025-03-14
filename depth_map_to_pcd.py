from parsers.parsers import *

import open3d as o3d
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os


def depth_to_point_cloud(color_image,depth_image,K,E,trunc,scale):
    """
    Converts a depth image to a point cloud and visualizes it using Open3D.
    
    Args:
        depth_image_path (str): Path to the depth image file.
        intrinsic_matrix (np.array): Camera intrinsic matrix (3x3).
    """
    
    # Convert depth image to an Open3D depth image (requires the depth to be in meters)
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_o3d = o3d.geometry.Image(color_image.astype(np.uint8))
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d,convert_rgb_to_intensity = False,depth_scale=1.0, depth_trunc=trunc)

    # Create the intrinsic camera parameters from the provided matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], K)

    # Generate the point cloud from rgbd image 
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic,E)
    pcd.scale(scale,center=(0, 0, 0))
    return pcd

def write_o3d_pcd(o3d_pcd,dir,file_name):
    o3d.io.write_point_cloud(f"{dir}/{file_name}.ply",o3d_pcd)

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('img', type=int, default=0, help='FOO!')
    #args = parser.parse_args()
    #image_int = args.img

    dataset = "boardgames_mobile"
    main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
    N =  218
    trunc = 12500.
    scale = 0.12

    full = True
    if full :
        pcd_dir = main_dir + "generated_pcd/"
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)
    
    traj_dir = main_dir + "trajectories/"
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    subsample = True
    SUBSAMPLE = 100
    if subsample :
        sub_pcd_dir = main_dir + "generated_sub_pcd/"
        if not os.path.exists(sub_pcd_dir):
            os.makedirs(sub_pcd_dir)
    
    trajectory_points = []
    trajectory_cubes = []
    lines = []

    for image_int in tqdm(range(N)):

        source_img = main_dir+"sparse/undistorted/%03d.jpg"%image_int
        depth_img_path = main_dir+"dense/depthmaps/%03d.png"%image_int
        camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
        proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int
        # Example depth map (synthetic)

        # Example intrinsic camera matrix (fx, fy, cx, cy)
        K,w,h = parse_pinhole_camera_params(camera_intrinsics_txt)

        color_image = load_image_to_numpy(source_img)
        depth_map = load_image_to_numpy(depth_img_path,resize=(w,h))
        # Example 3x4 transformation matrix (identity)
        E = parse_extrinsic_matrix(proj_mat_txt,K)


        # Call the function to convert depth to point cloud and visualize it
        pcd = depth_to_point_cloud(color_image,depth_map, K,E,trunc,scale)

        if subsample :
            sub_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd,SUBSAMPLE)
            write_o3d_pcd(sub_pcd,sub_pcd_dir,f"img_to_pcd_{image_int}")
        if full :
            write_o3d_pcd(pcd,pcd_dir,f"img_to_pcd_{image_int}")


        # Extract translation (camera position)
        position = E[:3, 3]
        trajectory_points.append(position)

    # Create a point cloud for visualization
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(trajectory_points)
    pcd_cam.paint_uniform_color([0, 1, 0])  # Green for trajectory points

    # Save as PLY file
    o3d.io.write_point_cloud(f"{traj_dir}trajectory_points.ply", pcd_cam,write_ascii = True)