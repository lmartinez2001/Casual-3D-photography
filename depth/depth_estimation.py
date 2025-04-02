import cv2
import numpy as np
from utils import generate_coords_array
from tqdm import tqdm


def project_to_3d_world(depth: float, 
                        R: np.ndarray, 
                        t: np.ndarray, 
                        K: np.ndarray, 
                        n_rows: int, 
                        n_cols: int):
    n_pixels = n_rows * n_cols
    pixel_coords = generate_coords_array(n_cols, n_rows) # (n_pixels, (col, rowd))

    normalized_coords_2d = cv2.undistortPoints(pixel_coords, K, np.zeros((4,1), dtype=np.float32)) # (n_pixels, 2)

    points_3d_camera = np.concatenate([normalized_coords_2d * depth, np.ones((n_pixels, 1, 1), dtype=np.float32) * depth], axis=-1) # (n_pixels, 1, 3)

    points_3d_camera_minus_t = (points_3d_camera - t).reshape(-1,3) # (n_pixels, 3)

    points_3d_world = np.dot(R.T, points_3d_camera_minus_t.T).T # (n_pixels, 3) Coordinates of each pixel in the 3D world for a given depth hypothesis

    return points_3d_world, pixel_coords.astype(int)


def compute_coords_in_target_image(points_3d_world: np.ndarray,
                                   K: np.ndarray, 
                                   R_target: np.ndarray, 
                                   t_target: np.ndarray):
    rvec_target, _ = cv2.Rodrigues(R_target) # Rodrigues form of the target rotation
    points_3d_world = points_3d_world.reshape(-1, 1, 3)
    pixel_coords, _ = cv2.projectPoints(points_3d_world, rvec_target, t_target, K, np.zeros((5, 1), dtype=np.float32))

    return pixel_coords.reshape(-1, 2).round()


def in_bounds_indices(pixel_coords: np.ndarray, rows: int, cols: int):
    return (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < cols) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < rows)
    )


def transform_sad(sad: np.ndarray, sigma: float = 0.033):
    return 1 - np.exp(-sad/sigma)


# ==> Compute coords of pixels from anchor image to all neighbors given one depth
def compute_coords_in_neighbor_images(points_3d_world: np.ndarray, K: np.ndarray, neigh_R: np.ndarray, neigh_t: np.ndarray):
    transformed_coords = []
    for target_R, target_t in zip(neigh_R, neigh_t):
        transformed_coords.append(compute_coords_in_target_image(points_3d_world=points_3d_world,
                                                                K=K,
                                                                R_target=target_R,
                                                                t_target=target_t))
    return np.array(transformed_coords)

# ==> Compute coords of pixels from anchor image to all neighbors FOR ALL depths
def compute_projected_coords_all_depths(depths: np.ndarray, 
                                        anchor_R: np.ndarray, 
                                        anchor_t: np.ndarray, 
                                        neigh_R: np.ndarray,
                                        neigh_t: np.ndarray,
                                        K: np.ndarray, 
                                        n_rows: int, 
                                        n_cols: int):
    all_transformed_coords = []
    for depth in tqdm(depths):
        points_3d_world, raw_pixels_coords = project_to_3d_world(depth=depth, 
                                                            R=anchor_R,
                                                            t=anchor_t,
                                                            K=K,
                                                            n_rows=n_rows,
                                                            n_cols=n_cols)
        
        transformed_coords = compute_coords_in_neighbor_images(points_3d_world=points_3d_world,
                                                       K=K,
                                                       neigh_R=neigh_R,
                                                       neigh_t=neigh_t)
        
        all_transformed_coords.append(transformed_coords)
    return np.array(all_transformed_coords), raw_pixels_coords