import numpy as np
import matplotlib.pyplot as plt

def cubed_sphere_mesh(n):
    """
    Constructs a cubed sphere mesh with n subdivisions.
    Returns an array of shape (6*n**2, 3) containing the nodal points.
    """
    # Define the cube
    x = np.linspace(-1, 1, n+1)
    X, Y, Z = np.meshgrid(x, x, x)

    # Inscribe a sphere inside the cube
    R = np.sqrt(X**2 + Y**2 + Z**2)
    X /= R
    Y /= R
    Z /= R

    # Project the vertices of the cube onto the sphere
    xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    r = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    theta = np.arccos(xyz[:,2]/r)
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    nodal_points = np.vstack((theta, phi)).T

    # Divide each square into smaller squares
    for i in range(6):
        for j in range(n):
            for k in range(n):
                idx = i*n**2 + j*n + k
                p1 = nodal_points[idx]
                p2 = nodal_points[idx+n]
                p3 = nodal_points[idx+n+1]
                p4 = nodal_points[idx+1]
                theta = np.linspace(p1[0], p2[0], n+1)
                phi = np.linspace(p1[1], p4[1], n+1)
                for l in range(1,n):
                    for m in range(1,n):
                        nodal_points = np.vstack((nodal_points, [theta[l], phi[m]]))

    return nodal_points

def partial_derivative(f, nodal_points, i, j, h):
    """
    Computes the partial derivative of f with respect to the i-th coordinate
    using a centered difference formula with spacing h at the nodal point j.
    """
    x = nodal_points[j]
    stencil = [j-h, j+h]
    x1 = nodal_points[stencil[0]]
    x2 = nodal_points[stencil[1]]
    f1 = f(stencil[0])
    f2 = f(stencil[1])
    return (f2[i] - f1[i]) / (x2[i] - x1[i])

# Define the test function
def f(theta, phi):
    return np.sin(theta) * np.cos(2*phi)

# Define the analytical derivatives of the test function
def df_dtheta(theta, phi):
    return np.cos(theta) * np.cos(2*phi)

def df_dphi(theta, phi):
    return -2 * np.sin(theta) * np.sin(2*phi)

# Construct the cubed sphere mesh
n = 32
nodal_points = cubed_sphere_mesh(n)

# Compute the partial derivatives using the centered difference formula
h = 2
df_dtheta_approx = np.array([partial_derivative(f, nodal_points, 0, i, h) for i in range(len(nodal_points))])
df_dphi_approx = np.array([partial_derivative(f, nodal_points, 1, i, h) for i in range(len(nodal_points))])

# Compute the analytical derivatives of the test function
df_dtheta_analytical = df_dtheta(nodal_points[:,0], nodal_points)
