import meshio
import numpy as np
import os
from tqdm import tqdm
from utils.normals import compute_local_PCA

def stretch_penalty(normals,view_dirs,tresh_angle = 1.66):

    dot_product = np.abs(np.sum(normals * view_dirs, axis=1))
    incidence_angle = np.arccos(dot_product)
    grazing_angle = 90 - np.degrees(incidence_angle)
    s = 1 - np.clip((grazing_angle/tresh_angle),a_min = 0,a_max=1)
    return s

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('img', type=int, default=0, help='FOO!')
    #args = parser.parse_args()
    #image_int = args.img

    dataset = "creepyattic"
    main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
    ### Values from param.txt
    N =  50
    minD = 3.3333333333333335
    trunc = 10000.0
    scale = 0.15
    factor = (2**16-1)

    full = False
    if full :
        pcd_dir = main_dir + "generated_pcd"
    subsample = True
    if subsample :
        pcd_dir = main_dir + "generated_sub_pcd"

    out_dir = pcd_dir+"_normals"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for image_int in tqdm(range(N)):

        source_img = main_dir+"sparse/undistorted/%03d.jpg"%image_int
        depth_img_path = main_dir+"dense/depthmaps/%03d.png"%image_int
        camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
        proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int

        pcd = meshio.read(f"{pcd_dir}/img_to_pcd_{image_int}.vtk",file_format="vtk")
        v = pcd.point_data['View_dir']

        all_eigenvalues, all_eigenvectors = compute_local_PCA(pcd.points, pcd.points, 0.50)
        normals = all_eigenvectors[:, :, 0]
        normals  /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize

        pcd.point_data['Normals'] = normals

        s = stretch_penalty(normals,v)
        pcd.point_data['s'] = s

        meshio.write(f"{out_dir}/pcd_{image_int}.vtk",pcd)
        

        