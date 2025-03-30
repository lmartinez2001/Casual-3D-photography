import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import meshio,os

dataset = "creepyattic"
main_dir = "Volumes/prn1_smb_computational_photo_001/projects/3DPhoto/Data/intermediate_data/%s/"%dataset
save_dir = main_dir + "pano_result/"

pano_fc = np.load(f"{save_dir}panorama_rgb_front.npy")[50*2:220*2,350*2:] 
pano_bc = np.load(f"{save_dir}panorama_rgb_back.npy")[50*2:220*2,350*2:]
pano_fd = np.load(f"{save_dir}panorama_d_front.npy")[50*2:220*2,350*2:]
pano_bd = np.load(f"{save_dir}panorama_d_back.npy")[50*2:220*2,350*2:]

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

x = np.arange(50*2, 220*2)
y = np.arange(350*2,512*2)
panorama_width, panorama_height = 256*2,512*2
# full coordinate arrays

xx, yy = np.meshgrid(x, y)
pano = np.stack([xx.T,yy.T],axis=2)
phi, theta = equirectangular_to_spherical(xx.T, yy.T, panorama_width, panorama_height)
points = spherical_to_cartesian(pano_bd*trunc,theta,phi)
plt.imsave("reco.png",pano_fc)
points = points.reshape((-1,3))
colors = pano_bc.reshape((-1,3))
mesh = meshio.Mesh(
    points=points,
    cells=[("vertex",np.array([[i,] for i in range(points.shape[0])]))],  # No connectivity (pure point cloud)
    point_data={
        "Colors": colors
    }
    )
#mesh.write(f"{dir}/{file_name}.vtk",file_format="vtk")
mesh.write("reco_b.vtk",file_format="vtk")