import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Configuration
# ==============================
panorama_width = 2048  # Output width
panorama_height = 1024  # Output height
projection_type = "spherical"  # Choose "cylindrical" or "spherical"
use_z_test = True  # Enable depth-based occlusion handling

# Initialize panorama and depth buffer
panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
depth_buffer = np.full((panorama_height, panorama_width), np.inf)  # Z-buffer

# ==============================
# Load Precomputed 3D Points
# ==============================
# Assume 'points_3d' is a NumPy array of shape (N, 3)
# Assume 'colors' is a NumPy array of shape (N, 3) with values in [0,1]
# Example: Load from a PLY file (modify as needed)
for k in [0,7,10]:
    print(k)
    pcd = o3d.io.read_point_cloud("/home/infres/msrir-21/Casual-3D-photography/img_to_pcd_colored_boardgames_mobile_%d.ply"%k)  # Modify path as needed
    points_3d = np.asarray(pcd.points)  # Shape (N, 3)
    colors = np.asarray(pcd.colors)  # Shape (N, 3)

    # ==============================
    # Projection and Mapping
    # ==============================
    for i in range(points_3d.shape[0]):
        X, Y, Z = points_3d[i]
        
        if projection_type == "cylindrical":
            # Cylindrical projection
            theta = np.arctan2(X, Z)  # Longitude (horizontal angle)
            y = Y  # Keep vertical coordinate

            # Convert to 2D image coordinates
            pano_x = int((theta + np.pi) / (2 * np.pi) * panorama_width)
            pano_y = int((y + 2) / 4 * panorama_height)  # Scale Y appropriately
        
        elif projection_type == "spherical":
            # Spherical projection
            r = np.linalg.norm([X, Y, Z])  # Radius
            theta = np.arctan2(X, Z)  # Longitude
            phi = np.arcsin(Y / r)  # Latitude

            # Convert to 2D image coordinates
            pano_x = int((theta + np.pi) / (2 * np.pi) * panorama_width)
            pano_y = int((phi + (np.pi / 2)) / np.pi * panorama_height)  # Map phi to [0, π]

        else:
            raise ValueError("Invalid projection type. Choose 'cylindrical' or 'spherical'.")

        # Check if within bounds
        if 0 <= pano_x < panorama_width and 0 <= pano_y < panorama_height:
            if use_z_test:
                # Naive Z-test (keep closest depth)
                if Z < depth_buffer[pano_y, pano_x]:  
                    depth_buffer[pano_y, pano_x] = Z
                    panorama[pano_y, pano_x] = (colors[i] * 255).astype(np.uint8)  # Assign pixel color
            else:
                # Overwrite without depth testing
                panorama[pano_y, pano_x] = (colors[i] * 255).astype(np.uint8)

# ==============================
# Save and Display Panorama
# ==============================
plt.imsave("panorama_open3d.png", panorama)
