import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

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

# Intrinsic matrix
def get_K(path: str, as_matrix=True):
    with open(path, "r") as f:
        f.readline()
        info = f.readline()
    fx, fy, cx, cy = list(map(float, info.split()[-4:]))
    if as_matrix:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K
    return fx, fy, cx, cy

# Global parameters
def get_global_params(path: str):
    data = {}
    with open(path , 'r') as file:
        for line in file:
            key, value = line.split('=')
            key = key.strip() 
            value = value.strip()

            if key in ['panoUp', 'panoForward', 'panoCenter']:
                # Convert the value into a tuple of floats
                data[key] = np.array(list(map(float, value.split())))
            elif key in ['colmapToMeters', 'minDepthForScene', 'maxDepthForScene']:
                data[key] = float(value)
    return data


def compute_depth_hypothesis(N: int, dmin: float, dmax: float):
    hypothesis = np.linspace(1/np.sqrt(dmin), 1/np.sqrt(dmax), N)
    hypothesis = 1/hypothesis**2
    return hypothesis


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
        

if __name__ == "__main__":
    rootdir = "../res/"
    im_file = os.path.join(rootdir, )

    # # TEST DEPTH HYPOTHESIS
    # N = 220
    # dmin = 0.5
    # dmax = 1500.0
    # hyp = compute_depth_hypothesis(N, dmin, dmax)