import numpy as np
import matplotlib.pyplot as plt

'''
To construct a cubed sphere mesh, we first start with a cube whose faces are divided into a 
regular grid of points. We then map each point on the cube to a point on the surface of a sphere 
inscribed inside the cube. This mapping preserves the regular grid structure and results in a mesh 
with six square faces that are each divided into a regular grid of points. This mesh is known as the cubed sphere mesh.

2.On a cubed sphere, we can compute the partial derivatives of f using the 
function values in nodal points only. These derivatives can be approximated using finite 
difference schemes that involve the function values at neighboring points on the mesh.

3.A stencil is a set of neighboring points on the mesh that are used to compute the derivative
of a function at a particular point. The convergence of a linear combination of 
function values depends on the order of the finite difference scheme being used. 
Higher-order schemes generally converge more quickly as the grid spacing vanishes.

4.To demonstrate convergence of a finite difference scheme on a cubed sphere mesh, 
we can compare the computed derivatives with the exact analytical solutions for simple 
functions. For example, we could compute the gradient of a function that varies linearly 
with x, y, and z and compare it to the exact analytical solution. We could also visualize 
the convergence by plotting the computed derivatives as a function of the grid spacing and 
comparing it to the expected rate of convergence.1.'''

def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) # latitude
        phi = np.arctan2(y, x) # longitude
        return theta, phi

def spherical_to_cartesian(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def CubedSphere(n, f, point):
    # Generate faces of the cube
    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    cubeOrigin = []

    # top face
    cubeOrigin.append([[[round(x, 2), round(y, 2), 1] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    # bottom face
    cubeOrigin.append([[[round(x, 2), round(y, 2), -1] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    # left face
    cubeOrigin.append([[[-1, round(x, 2), round(y, 2)] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    # right face
    cubeOrigin.append([[[1, round(x, 2), round(y, 2)] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    # front face
    cubeOrigin.append([[[round(x, 2), -1, round(y, 2)] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    # back face
    cubeOrigin.append([[[round(x, 2), 1, round(y, 2)] /
                      np.sqrt(x**2 + y**2 + 1) for y in Y] for x in X])
    

    '''
    This function calculates the distance between two points on the surface of 
    a sphere using the Haversine formula. The inputs to the function are the 
    latitude and longitude coordinates of the two points, where p1
    represents the coordinates of the first point and p2 represents 
    the coordinates of the second point.
    '''
    def distanceEstimator(p1, p2):
        return 2 * np.arcsin(np.sqrt(np.sin((p2[0] - p1[0]) / 2)**2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin((p2[1] - p1[1]) / 2)**2))


    def computeDerivative(f, pointOrigin, dir,n):
        distance = float('inf')
        closestPoint = None
        prevPointX = None
        prevPointY = None
        for face in cubeOrigin:
            # print(face,"\n")
            for i,row in enumerate(face):
                for j,point in enumerate(row):
                    # print(point)
                    lat, longt = cartesian_to_spherical(point[0], point[1], point[2])
                    dist = distanceEstimator(pointOrigin, (lat, longt))
                    if dist < distance:
                        distance = dist
                        closestPoint = point


        distance = float('inf')
        offsetX = np.array([1/ n,0,0])
        for face in cubeOrigin:
            for i,row in enumerate(face):
                for j,point in enumerate(row):
                    dist = np.linalg.norm(point - closestPoint + offsetX)
                    if dist < distance and (point != closestPoint).any():
                        distance = dist
                        prevPointX = point

        distance = float('inf')
        offsetY = np.array([0,1/ n,0])
        for face in cubeOrigin:
            for i,row in enumerate(face):
                for j,point in enumerate(row):
                    dist = np.linalg.norm(point - closestPoint + offsetY)
                    if dist < distance and (point != closestPoint).any() and (point != prevPointX).any():
                        distance = dist
                        prevPointY = point

        

        if dir == 'x':
            h = np.linalg.norm(closestPoint - prevPointX)
            print("Partial derivative along x axis on nodal point")
            return (closestPoint,prevPointX,prevPointY,(f(closestPoint[0], closestPoint[1],closestPoint[2]) - f(prevPointX[0],prevPointX[1],prevPointX[2])) / h)
        if dir == 'y':
            h = np.linalg.norm(closestPoint - prevPointY)
            print("Partial derivative along y axis on nodal point")
            return (closestPoint,prevPointX,prevPointY,(f(closestPoint[0], closestPoint[1],closestPoint[2]) - f(prevPointY[0],prevPointY[1],prevPointY[2])) / h)
        else:
            return closestPoint
        

    closestPoint,prevPointX,prevPointY,value = computeDerivative(f, point, 'y',n)
    print(value)
    # Draw the points of face in the cube
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x = closestPoint[0]
    y = closestPoint[1]
    z = closestPoint[2]

    ax.scatter(x, y, z, c='b', marker='o')
    ax.scatter(prevPointX[0],prevPointX[1],prevPointX[2], c='g', marker='o')
    ax.scatter(prevPointY[0],prevPointY[1],prevPointY[2], c='yellow', marker='o')
    for face in cubeOrigin:
        for row in face:
            # print(row,"row")
            X, Y, Z = zip(*row)
            # print(X)
            ax.plot(X, Y, Z, color='red')
        for column in range(len(face)):
            X, Y, Z = zip(*[face[row][column] for row in range(len(face))])
            ax.plot(X, Y, Z, color='red')
    plt.show()


def f(x, y,z): return x**2 + y**2 + z**2 + 60
print("---------")
x,y,z = 50,6,100
lat, longt = cartesian_to_spherical(x,y,z)
CubedSphere(50, f, (lat, longt))
