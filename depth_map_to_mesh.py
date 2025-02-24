from parsers.parsers import *

import open3d as o3d
import numpy as np
from PIL import Image
import argparse

def depth_to_point_cloud(color_image,depth_image,K,E,frame = "random"):
    """
    Converts a depth image to a point cloud and visualizes it using Open3D.
    
    Args:
        depth_image_path (str): Path to the depth image file.
        intrinsic_matrix (np.array): Camera intrinsic matrix (3x3).
    """
    
    # Convert depth image to an Open3D depth image (requires the depth to be in meters)
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_o3d = o3d.geometry.Image(color_image.astype(np.uint8))
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d,convert_rgb_to_intensity = False,depth_scale=1.0, depth_trunc=1000.0)

    # Create the intrinsic camera parameters from the provided matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], K)

    # Generate the point cloud from the depth image 
    
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic,E)
    o3d.io.write_point_cloud(f"img_to_pcd_{frame}.ply",pcd)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic,E)
    o3d.io.write_point_cloud(f"img_to_pcd_colored_{frame}.ply",pcd)
    # Visualize the point cloud
    #o3d.visualization.draw_plotly([pcd])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=int, default=0, help='FOO!')
    args = parser.parse_args()

    dataset = "boardgames_mobile"
    main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
    image_int = args.img

    source_img = main_dir+"sparse/undistorted/%03d.jpg"%image_int
    depth_img_path = main_dir+"dense/depthmaps/%03d.png"%image_int
    camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
    proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int
    # Example depth map (synthetic)

    # Example intrinsic camera matrix (fx, fy, cx, cy)
    K,w,h = parse_pinhole_camera_params(camera_intrinsics_txt)

    color_image = load_image_to_numpy(source_img)
    print(color_image.shape)
    depth_map = load_image_to_numpy(depth_img_path,resize=(w,h))
    print(depth_map.shape)
    # Example 3x4 transformation matrix (identity)
    E = parse_extrinsic_matrix(proj_mat_txt,K)

    print(K)
    print(E)
    
    # Call the function to convert depth to point cloud and visualize it
    depth_to_point_cloud(color_image,depth_map, K,E,frame = dataset+"_"+str(image_int))



