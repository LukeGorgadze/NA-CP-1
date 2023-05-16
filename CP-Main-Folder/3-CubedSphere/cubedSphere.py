import numpy as np  
import matplotlib.pyplot as plt

def CartToSpheric(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) 
        return theta, phi   

def CubeToSphere(n, f, point):
    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    origin = []
    mainOrigin = []

    for x in X:
        row = []
        for y in Y:
            row.append([x, y, 1] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    origin = []
    for x in X:
        row = []
        for y in Y:
            row.append([x, y, -1] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    origin = []
    for x in X:
        row = []
        for y in Y:
            row.append([-1, x, y] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    origin = []
    for x in X:
        row = []
        for y in Y:
            row.append([1, x, y] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    origin = []
    for x in X:
        row = []
        for y in Y:
            row.append([x, -1, y] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    origin = []
    for x in X:
        row = []
        for y in Y:
            row.append([x, 1, y] / np.sqrt(x**2 + y**2 + 1))
        origin.append(row)
    mainOrigin.append(origin)

    

    def sphereDistance(point1, point2):
        p1_lat, p1_lon = point1
        p2_lat, p2_lon = point2

        delta_lat = p2_lat - p1_lat
        delta_lon = p2_lon - p1_lon

        haversin_lat = np.sin(delta_lat / 2) ** 2
        haversin_lon = np.sin(delta_lon / 2) ** 2

        cos_p1_lat = np.cos(p1_lat)
        cos_p2_lat = np.cos(p2_lat)

        haversin_sum = haversin_lat + cos_p1_lat * cos_p2_lat * haversin_lon

        distance = 2 * np.arcsin(np.sqrt(haversin_sum))

        return distance 

    def calcDerivative(f, po, dir, h):
        minDistance = float('inf')
        closestPoint = None
        for face in mainOrigin:
            for row in face:
                for point in row:
                    lat, longt = CartToSpheric(point[0], point[1], point[2])
                    currDist = sphereDistance(po, (lat, longt))
                    if currDist < minDistance:
                        minDistance = currDist
                        closestPoint = point
        if dir == 'x':
            return (f(closestPoint[0] + h, closestPoint[1],closestPoint[2]) - f(closestPoint[0], closestPoint[1],closestPoint[2])) / h
        if dir == 'y':
            return (f(closestPoint[0], closestPoint[1] + h,closestPoint[2]) - f(closestPoint[0], closestPoint[1],closestPoint[2])) / h
        if dir == 'z':
            return (f(closestPoint[0], closestPoint[1],closestPoint[2] + h) - f(closestPoint[0], closestPoint[1],closestPoint[2])) / h
        else:
            return None


    print(calcDerivative(f, point, 'x', 0.0001))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xdata = []
    ydata = []
    zdata = []
    for face in mainOrigin:
        for row in face:
            for point in row:
                xdata.append(point[0])
                ydata.append(point[1])
                zdata.append(point[2])
    ax.scatter3D(xdata, ydata, zdata);
    plt.show()
    

def f(x, y, z): return np.exp(x) + np.exp(y) + np.cos(x) + np.sin(y) + np.sin(z)
x,z,y = 10,10,10
latitude,longtitude = CartToSpheric(x,y,z)
CubeToSphere(10, f, (latitude,longtitude))