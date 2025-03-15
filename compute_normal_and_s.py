import meshio
import numpy as np
import os
from tqdm import tqdm
from utils.normals import compute_local_PCA

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('img', type=int, default=0, help='FOO!')
    #args = parser.parse_args()
    #image_int = args.img

    dataset = "boardgames_mobile"
    main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
    N = 1 # 218
    trunc = 12500.
    scale = 0.12

    full = False
    if full :
        pcd_dir = main_dir + "generated_pcd/"

    subsample = True
    if subsample :
        pcd_dir = main_dir + "generated_sub_pcd/"

    for image_int in tqdm(range(N)):

        source_img = main_dir+"sparse/undistorted/%03d.jpg"%image_int
        depth_img_path = main_dir+"dense/depthmaps/%03d.png"%image_int
        camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
        proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int

        pcd = meshio.read(f"{pcd_dir}/img_to_pcd_{image_int}.ply"   )
        vx,vy,vz = pcd.point_data['view_directions_x'],pcd.point_data['view_directions_y'],pcd.point_data['view_directions_z']

        all_eigenvalues, all_eigenvectors = compute_local_PCA(pcd.points, pcd.points, 0.50)
        normals = all_eigenvectors[:, :, 0]

        