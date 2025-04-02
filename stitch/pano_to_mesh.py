import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import meshio,os

dataset = "creepyattic"
main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
save_dir = main_dir + "pano_result/"

pano_fc = np.load(f"{save_dir}panorama_rgb_front.npy")[100:440,700:] 
pano_bc = np.load(f"{save_dir}panorama_rgb_back.npy")[100:440,700:]
pano_fd = np.load(f"{save_dir}panorama_d_front.npy")[100:440,700:]
pano_bd = np.load(f"{save_dir}panorama_d_back.npy")[100:440,700:]

### Values from param.txt
trunc = 10000.0

def equirectangular_to_spherical(pano_x, pano_y, panorama_width, panorama_height):

    phi = ((pano_y-panorama_height)/panorama_height)*(2 * np.pi) 
    theta = (0.5-(pano_x/panorama_width))* np.pi

    return phi, theta

def spherical_to_cartesian(r,theta,phi):
    X = r * np.cos(theta) * np.cos(phi)
    Y = r * np.cos(theta) * np.sin(phi)
    Z = r*np.sin(theta)
    return np.stack((X, Y, Z),axis=2)

def points_to_mesh(points,colors):
    row,col = points.shape[:2]
    v_points = points.reshape((-1,3))
    v_colors = colors.reshape((-1,3))

    triangles = []
    R = np.arange((row-1))[:,None]
    C = np.arange(col-1)[None,:]

    p1 = (R*col+C).flatten()
    p2 = ((R+1)*col+C).flatten()
    p3 =(R*col+C+1).flatten()
    p4 = ((R+1)*col+C+1).flatten()

    s1 = np.stack((p1, p2, p3),axis=1)
    s2 = np.stack((p4, p2, p3),axis=1)

    triangles = np.vstack([s1,s2])

    mesh = meshio.Mesh(
    points=v_points,
    cells=[("vertex",np.array([[i,] for i in range(v_points.shape[0])])),
           #("triangle",np.array(triangles))
           ],  # No connectivity (pure point cloud)
    point_data={
        "Colors": v_colors
    }
    )
    
    return mesh
x = np.arange(100, 440)
y = np.arange(700,1024)
panorama_width, panorama_height = 512,1024
# full coordinate arrays

xx, yy = np.meshgrid(x, y)
pano = np.stack([xx.T,yy.T],axis=2)
phi, theta = equirectangular_to_spherical(xx.T, yy.T, panorama_width, panorama_height)

points = spherical_to_cartesian(pano_fd*trunc,theta,phi)
mesh = points_to_mesh(points,pano_fc)
mesh.write(f"{save_dir}reco_front.vtk",file_format="vtk")

points = spherical_to_cartesian(pano_bd*trunc,theta,phi)
mesh = points_to_mesh(points,pano_bc)
mesh.write(f"{save_dir}reco_back.vtk",file_format="vtk")