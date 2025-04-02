# Set of functions used to load camera intrinsic parameters from the textfile 
# Provided for each undistorted  image
# A more reliable and cleaner version is actually used, based on global parquet files gathering all the relevant information of a dataset
import numpy as np
import cv2
import os

# Rotation and translation matrices
def decompose_projection_matrix(P: np.ndarray, K: np.ndarray):
    # Ensure K is invertible
    K_inv = np.linalg.inv(K)
    
    # Extract rotation and translation
    M = P[:, :3]  # First 3 columns (K * R)
    p4 = P[:, 3]  # Last column (K * t)
    
    # Compute R and t
    R = np.dot(K_inv, M)
    t = np.dot(K_inv, p4)
    
    # Ensure R is a valid rotation matrix
    U, _, Vt = np.linalg.svd(R)  # Enforce orthonormality
    R = np.dot(U, Vt)

    return R, t

# Projection matrix
def get_P(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    P = np.array([list(map(float, line.split())) for line in lines])
    return P


def load_gt_depth_map(path: str, min_depth_for_scene: float, max_depth_for_scene: float, resize: tuple = None) -> np.ndarray:
    k_max_value_16 = 65535.0
    
    input_image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if resize is not None:
        input_image = cv2.resize(input_image, resize, interpolation=cv2.INTER_NEAREST)
    output = (input_image / k_max_value_16) * (max_depth_for_scene - min_depth_for_scene) + min_depth_for_scene
    
    return output

def to_normalized_coordinates(point: np.ndarray, config: dict):
    pano_right = np.cross(config["panoForward"], config["panoUp"])
    pano_right /= np.linalg.norm(pano_right)

    ortho_pano_up = np.cross(pano_right, config["panoForward"])
    ortho_pano_up /= np.linalg.norm(ortho_pano_up)

    rotation = np.concatenate((pano_right, ortho_pano_up, -config["panoForward"]))

    return (rotation @ (point - config["PanoCenter"])) * config["colmapToMeters"]


def get_all_extrinsic_params(root_path: str, K: np.ndarray) -> tuple:
    # ../res/sparse/undistorted
    all_P_files = []
    # Retrive every projection matrix file
    for f in os.listdir(root_path):
        if "proj_matrix" in f:
            all_P_files.append(f)

    all_P_files = sorted(all_P_files)
    all_R = np.zeros((len(all_P_files), 3, 3))
    all_t = np.zeros((len(all_P_files), 3))

    for i, proj_file in enumerate(all_P_files):
        P = get_P(os.path.join(root_path, proj_file))
        R, t = decompose_projection_matrix(P, K)
        all_R[i] = R
        all_t[i] = t    
    return all_R, all_t