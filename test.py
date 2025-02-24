import numpy as np

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

# Example Intrinsic Matrix (K)
K = np.array([
 [2.77974e+03, 0.00000e+00, 1.60550e+03],
 [0.00000e+00, 2.78896e+03, 1.20400e+03],
 [0.00000e+00, 0.00000e+00, 1.00000e+00]
])


# Example Projection Matrix (P)
P = np.array([[ 2.76853e+03,  1.03928e+03,  1.24889e+03, -1.26343e+03],
 [-1.55611e+02, -1.11567e+03,  2.82117e+03, -6.58488e+03],
 [ 7.29678e-01, -6.50866e-01,  2.09627e-01, -5.17238e+00]])

# Decompose to get R and t
R, t = decompose_projection_matrix(P, K)

print("Rotation Matrix (R):\n", R)
print("Translation Vector (t):\n", t)