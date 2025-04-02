import os
import cv2
import numpy as np
import polars as pl
from PIL import Image


# Intrinsic matrix
def get_K(path: str, as_matrix=True):
    with open(path, "r") as f:
        f.readline()
        info = f.readline()
    fx, fy, cx, cy = list(map(float, info.split()[-4:]))
    if as_matrix:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
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
    hypothesis = np.linspace(1/np.sqrt(dmin), 1/np.sqrt(dmax), N, dtype=np.float32)
    hypothesis = 1/hypothesis**2
    return hypothesis


def get_extrinsic_params(images_df: pl.DataFrame, im_index: int):
    im_params = images_df.filter(pl.col("IMAGE_ID") == im_index)
    im_Q = im_params[["QW", "QX","QY", "QZ"]].to_numpy()[0].astype(np.float32)
    im_R = quaternion_rotation_matrix(im_Q)
    im_t = im_params[["TX","TY", "TZ"]].to_numpy()[0].astype(np.float32)
    return im_R, im_t
        
def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


def generate_coords_array(cols, rows):
    row_indices = np.repeat(np.arange(rows), cols)
    col_indices = np.tile(np.arange(cols), rows)

    result = np.column_stack((col_indices, row_indices)).astype(np.float32)
    return result

# def get_image_gradient(im: np.ndarray, downscale_factor: int,normalized: bool=False):
#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     if normalized: gray = (gray-gray.min()) / (gray.max()-gray.min()) 
#     ksize = 3
#     grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
#     grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
#     return grad_x, grad_y

def get_image_gradient(im: np.ndarray, downscale_factor: int, normalized: bool=False):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    if normalized:
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    ksize = 3
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    if downscale_factor > 1:
        sigma = 0.5 * downscale_factor
        grad_x_blurred = cv2.GaussianBlur(grad_x, (0, 0), sigma)
        grad_y_blurred = cv2.GaussianBlur(grad_y, (0, 0), sigma)

        height, width = grad_x.shape
        new_height = height // downscale_factor
        new_width = width // downscale_factor
 
        grad_x = cv2.resize(grad_x_blurred, (new_width, new_height), interpolation=cv2.INTER_AREA)
        grad_y = cv2.resize(grad_y_blurred, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return grad_x, grad_y


# ==> Functions to load the parameters of the neighboring images of the anchor image
def load_image_by_index(root: str, index: int, downscale_factor: int, return_fullsize: bool = False):
    im_title = f"{index:03d}"
    im = Image.open(os.path.join(root, f"{im_title}.jpg"))
    im_down  = im.resize(np.array(im.size) // downscale_factor)
    im_down = np.array(im_down)

    if return_fullsize: return im_down, np.array(im) 
    return im_down


def load_neighbor_images(root: str, indices: np.ndarray, downscale_factor: int):
    images_down = []
    images_full = []
    for idx in indices:
        im_down, im_full = load_image_by_index(root, idx, downscale_factor, return_fullsize=True)
        images_down.append(im_down)
        images_full.append(im_full)
    return np.array(images_down), np.array(images_full)


def load_neighbor_extrinsic_params(df: pl.DataFrame, indices: np.ndarray):
    neigh_R = []
    neigh_t = []
    for idx in indices:
        R, t = get_extrinsic_params(df, idx)
        neigh_R.append(R)
        neigh_t.append(t)
    return np.array(neigh_R), np.array(neigh_t)


def get_neighbor_images_gradient(images: np.ndarray, downscale_factor: int,  normalized: bool = False):
    neighbor_gradients = []
    for im in images:
        grad_x, grad_y = get_image_gradient(im, normalized=normalized, downscale_factor=downscale_factor)
        neighbor_gradients.append(np.stack([grad_x, grad_y], axis=-1))
    return np.array(neighbor_gradients)
