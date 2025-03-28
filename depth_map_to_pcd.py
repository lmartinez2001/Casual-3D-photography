from utils.parsers import *

import open3d as o3d
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os

import meshio

def depth_to_point_cloud(color_image,depth_image,K,E,trunc,scale,subsample = 1):
    """
    Converts a depth image to a point cloud and visualizes it using Open3D.
    
    Args:
        depth_image_path (str): Path to the depth image file.
        intrinsic_matrix (np.array): Camera intrinsic matrix (3x3).
    """
    
    # Convert depth image to an Open3D depth image (requires the depth to be in meters)
    depth_o3d = o3d.t.geometry.Image(depth_image.astype(np.float32))
    color_o3d = o3d.t.geometry.Image(color_image.astype(np.uint8))
    rgbd_o3d = o3d.t.geometry.RGBDImage(color_o3d, depth_o3d)

    # Create the intrinsic camera parameters from the provided matrix
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], K)

    # Generate the point cloud from rgbd image 
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d,K,E,
                                                           depth_scale=1.0, depth_max=trunc,stride = subsample)

    #pcd.point.positions*=scale

    points = pcd.point.positions.numpy()  # Shape (N, 3)
    R, t = E[:3, :3].numpy(), E[:3, 3].numpy()
    camera_center = -R.T @ t

    # Compute view direction vectors (from each point to camera center)
    view_vectors = camera_center - points  # Vector pointing to the camera
    view_vectors /= np.linalg.norm(view_vectors, axis=1, keepdims=True)  # Normalize
    
    dict_pcd = {
        "pcd" : pcd,
        "view_vectors" : view_vectors
    }

    return dict_pcd


def write_o3d_pcd(o3d_pcd,dir,file_name):
    N = o3d_pcd["view_vectors"].shape[0]
    mesh = meshio.Mesh(
    points=o3d_pcd['pcd'].point.positions.numpy(),
    cells=[("vertex",np.array([[i,] for i in range(N)]))],  # No connectivity (pure point cloud)
    point_data={
        "View_dir": o3d_pcd["view_vectors"],
        "Colors": o3d_pcd['pcd'].point.colors.numpy()
    }
    )
    mesh.write(f"{dir}/{file_name}.vtk",file_format="vtk")

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('img', type=int, default=0, help='FOO!')
    #args = parser.parse_args()
    #image_int = args.img

    dataset = "creepyattic"
    main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
    N =  50
    minD = 3.3333333333333335
    trunc = 10000.0
    scale = 0.15
    factor = (2**16-1)

    full = False
    if full :
        pcd_dir = main_dir + "generated_pcd/"
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)
    
    traj_dir = main_dir + "trajectories/"
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    subsample = True
    SUBSAMPLE = 16
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
        depth_map = ((depth_map/factor)*(trunc-minD) )+ minD
        # Example 3x4 transformation matrix (identity)
        E = parse_extrinsic_matrix(proj_mat_txt,K)
        
        K,E = o3d.core.Tensor(K),o3d.core.Tensor(E)

        # Call the function to convert depth to point cloud and save it

        if subsample :
            sub_pcd = depth_to_point_cloud(color_image,depth_map, K,E,trunc,scale,SUBSAMPLE)
            write_o3d_pcd(sub_pcd,sub_pcd_dir,f"img_to_pcd_{image_int}")
        if full :
            pcd = depth_to_point_cloud(color_image,depth_map, K,E,trunc,scale)
            write_o3d_pcd(pcd,pcd_dir,f"img_to_pcd_{image_int}")


        # Extract translation (camera position)
        R, t = E[:3, :3].numpy(), E[:3, 3].numpy()
        camera_center = -R.T @ t
        trajectory_points.append(camera_center)
        if image_int ==0 :
            pcd_cam = o3d.geometry.PointCloud()
            pcd_cam.points = o3d.utility.Vector3dVector(trajectory_points)
            pcd_cam.paint_uniform_color([0, 1, 0])  # Green for trajectory points
            o3d.io.write_point_cloud(f"{traj_dir}camera_0.ply", pcd_cam,write_ascii = True)

    # Create a point cloud for visualization
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(trajectory_points)
    pcd_cam.paint_uniform_color([0, 1, 0])  # Green for trajectory points

    # Save as PLY file
    o3d.io.write_point_cloud(f"{traj_dir}trajectory_points.ply", pcd_cam,write_ascii = True)