import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def decompose_projection_matrix(P, K):
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

def construct_4x4_extrinsic(R, t):
    E = np.eye(4)  # Initialize as identity matrix
    E[:3, :3] = R  # Set rotation part
    E[:3, 3] = t.flatten()  # Set translation part
    return E.astype(np.float32)

def parse_extrinsic_matrix(file_path,K_intrinsic):

    # Convert the list into a 3x4 matrix
    P_matrix_3x4 = np.loadtxt(file_path)
    
    R, t = decompose_projection_matrix(P_matrix_3x4, K_intrinsic)
    
    return construct_4x4_extrinsic(R, t)

def parse_pinhole_camera_params(file_path):
    """
    Parses the camera parameters from a text file and returns the intrinsic matrix (K).

    Args:
        file_path (str): Path to the input text file containing camera parameters.

    Returns:
        K (np.array): 3x3 intrinsic matrix for the pinhole camera.
        width (int): Camera image width.
        height (int): Camera image height.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # Look for the line that starts with "PINHOLE"
        if line.startswith("PINHOLE"):
            # Split the line by whitespace
            params = line.split()
            
            # Extract the width, height, and camera parameters
            width = int(params[1])
            height = int(params[2])
            fx = float(params[3])  # Focal length in x
            fy = float(params[4])  # Focal length in y
            cx = float(params[5])  # Principal point in x
            cy = float(params[6])  # Principal point in y
            
            # Construct the intrinsic matrix K
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]],dtype=np.float32)
            
            return K, width, height
    
    # If no "PINHOLE" line was found, raise an error
    raise ValueError("No valid pinhole camera parameters found in the file.")


def load_image_to_numpy(image_path,resize = None):
    """
    Load an image from a given file path and return it as a NumPy array.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.array: Image represented as a NumPy array.
    """
    # Open the image using PIL
    image = Image.open(image_path)
    if resize :
        image = image.resize(resize, resample= Image.Resampling.BILINEAR)
    
    # Convert image to a NumPy array
    image_array = np.array(image)

    return image_array

def compute_rotation_matrix(forward, up):
    # Normalize input vectors
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)

    # Compute right vector
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # Recompute forward to ensure orthogonality
    forward = np.cross(right, up)
    forward /= np.linalg.norm(forward)

    # Construct rotation matrix
    R = np.stack([right, up, -forward], axis=1)  # 3x3 matrix
    return R
    
if __name__ == "__main__":

    main_dir = "library-mobile/Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/boardgames_mobile/"
    image_int = 0

    source_img = main_dir+"source/%03d.jpg"%image_int
    depth_img = main_dir+"dense/depthmaps/%03d.png"%image_int
    camera_intrinsics_txt = main_dir+"sparse/undistorted/%03d.jpg.camera.txt"%image_int
    proj_mat_txt = main_dir+"sparse/undistorted/%03d.jpg.proj_matrix.txt"%image_int

    print(parse_extrinsic_matrix(proj_mat_txt))
    print(parse_pinhole_camera_params(camera_intrinsics_txt))
    plt.imshow(load_image_to_numpy(depth_img))
    plt.show()
    plt.imshow(load_image_to_numpy(source_img))
    plt.show()
    