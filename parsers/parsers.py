import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_extrinsic_matrix(file_path):

    # Convert the list into a 3x4 matrix
    matrix_3x4 = np.loadtxt(file_path)
    
    # Add a 4th row [0, 0, 0, 1] to make it a 4x4 matrix
    matrix_4x4 = np.vstack([matrix_3x4, [0, 0, 0, 1]])
    
    return matrix_4x4

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
                          [0, 0, 1]])
            
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
    