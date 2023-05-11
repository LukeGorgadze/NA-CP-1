import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

VERT_AMOUNT = 20

# Cube vertices
vertices = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1]
]

verts = [vert for vert in vertices]
vertices = np.array(vertices)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([0,0,1]) / VERT_AMOUNT * i + np.array([0,1,0]) / VERT_AMOUNT * j)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([1,0,0]) / VERT_AMOUNT * i + np.array([0,0,1]) / VERT_AMOUNT * j)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([1,0,0]) + np.array([0,1,0]) / VERT_AMOUNT * i + np.array([0,0,1]) / VERT_AMOUNT * j)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([1,1,0]) - np.array([1,0,0]) / VERT_AMOUNT * i + np.array([0,0,1]) / VERT_AMOUNT * j)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([1,0,0]) / VERT_AMOUNT * i + np.array([0,1,0]) / VERT_AMOUNT * j)

for i in range(VERT_AMOUNT + 1):
    for j in range(VERT_AMOUNT + 1):
        verts.append(np.array([0,0,1]) + np.array([1,0,0]) / VERT_AMOUNT * i + np.array([0,1,0]) / VERT_AMOUNT * j)

verts = list(map(lambda vert: vert - np.array([0.5,0.5,0.5]),verts))
verts = list(map(lambda vert: vert / np.linalg.norm(vert),verts))
# Extract x, y, and z coordinates of vertices
x = [vertex[0] for vertex in verts]
y = [vertex[1] for vertex in verts]
z = [vertex[2] for vertex in verts]

# Plot vertices
ax.scatter(x, y, z, color='r')

# Set limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
