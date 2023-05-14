import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate cubed sphere mesh
n = 10  # number of grid points along each face
m = 6*n**2  # total number of grid points
theta = np.linspace(0, np.pi, n+1)[1:-1]  # exclude poles
phi = np.linspace(0, 2*np.pi, n+1)[:-1]
x, y = np.meshgrid(phi, theta)
x = x.flatten()
y = y.flatten()
z = np.sqrt(1 - x**2 - y**2)
xyz = np.vstack([np.concatenate([x, -x, -y, y, -z, z]),
                 np.concatenate([y, -y, x, -x, -z, z]),
                 np.concatenate([z, z, z, z, -x, -x])]).T

# Compute finite difference stencil for partial derivative
h = np.sqrt(2/(n**2))  # mesh spacing
D = np.zeros((m, m))
for i in range(m):
    xi = xyz[i,:]
    for j in range(m):
        xj = xyz[j,:]
        if np.allclose(xi, xj):
            continue  # ignore diagonal
        xij = xj - xi
        r = np.linalg.norm(xij)
        if r > 2*h:
            continue  # ignore points too far away
        D[i,j] = np.dot(xij, np.array([1,2,3]))/(r**3)

# Define test function and analytical partial derivatives
f = lambda x,y,z: np.sin(2*x)*np.cos(y)*z
dfdx = lambda x,y,z: 2*np.cos(2*x)*np.cos(y)*z
dfdy = lambda x,y,z: -np.sin(2*x)*np.sin(y)*z
dfdz = lambda x,y,z: np.sin(2*x)*np.cos(y)

# Compute numerical partial derivatives for varying mesh spacing
nhs = np.logspace(-2, -6, 10)  # array of mesh spacings
errs = np.zeros_like(nhs)  # array of errors
for i, h in enumerate(nhs):
    Dh = D/h
    vals = f(xyz[:,0], xyz[:,1], xyz[:,2])
    dfdx_num = np.dot(Dh, vals)
    dfdy_num = np.dot(Dh, vals)
    dfdz_num = np.dot(Dh, vals)
    dfdx_analytic = dfdx(xyz[:,0], xyz[:,1], xyz[:,2])
    dfdy_analytic = dfdy(xyz[:,0], xyz[:,1], xyz[:,2])
    dfdz_analytic = dfdz(xyz[:,0], xyz[:,1], xyz[:,2])
    errs[i] = np.sqrt(np.mean((dfdx_num - dfdx_analytic)**2 +
                               (dfdy_num - dfdy_analytic)**2 +
                               (dfdz_num - dfdz_analytic)**2))

# Visualize convergence
fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(nhs, errs, '-o')
ax.set_xlabel('Mesh spacing')
ax.set_ylabel('Error')
ax.set_title('Convergence of finite difference stencil')
plt.show()
