from parsers.parsers import *

import open3d as o3d
import numpy as np
from PIL import Image

def depth_to_point_cloud(depth_image,K,T):
    """
    Converts a depth image to a point cloud and visualizes it using Open3D.
    
    Args:
        depth_image_path (str): Path to the depth image file.
        intrinsic_matrix (np.array): Camera intrinsic matrix (3x3).
    """
    
    # Convert depth image to an Open3D depth image (requires the depth to be in meters)
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # Create the intrinsic camera parameters from the provided matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[0], depth_image.shape[1], K)

    # Generate the point cloud from the depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic)

    o3d.io.write_point_cloud("img_to_pcd_bis.ply",pcd)
    # Visualize the point cloud
    #o3d.visualization.draw_plotly([pcd])

if __name__ == "__main__":

    main_dir = "library-mobile/Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/boardgames_mobile/"
    image_int = 10

    source_img = main_dir+"source/%03d.jpg"%image_int
    depth_img_path = main_dir+"dense/depthmaps/%03d.png"%image_int
    camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
    proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int

    # Example depth map (synthetic)

    # Example intrinsic camera matrix (fx, fy, cx, cy)
    K,w,h = parse_pinhole_camera_params(camera_intrinsics_txt)

    depth_map = load_image_to_numpy(depth_img_path,resize=(w,h))
    # Example 3x4 transformation matrix (identity)
    T = parse_extrinsic_matrix(proj_mat_txt)
    
    # Call the function to convert depth to point cloud and visualize it
    depth_to_point_cloud(depth_map*10, K,T)



