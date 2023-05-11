import cv2
import numpy as np
# Load the image
image = cv2.imread("images\SuzannesWithBalls.png")
image = cv2.resize(image, (700,700), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def GaussianSmoothing(image):
    # Define the Gaussian kernel
    kernel_size = 10
    sigma = 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            # This formula was on Slide 4, page 41
            kernel[i, j] = np.exp(-(i**2 + j**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)

    # Pad the image and avoid overflows
    padding = kernel_size // 2
    padded_image = np.zeros(
        (image.shape[0] + 2*padding, image.shape[1] + 2*padding, image.shape[2]), dtype=image.dtype)
    padded_image[padding:-padding, padding:-padding, :] = image

    # Apply convolution with the kernel
    output = np.zeros_like(image)
    for c in range(image.shape[2]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):

                # Note: We want gaussian filter to work on colored images, so we may "fix" the color channels,
                # and use convolution for each channel separately,
                # this is similar to turning image to black and white

                # Note: * is elementwise multiplication, see Extra/elementwise for more
                output[i, j, c] = np.sum(
                    padded_image[i:i+kernel_size, j:j+kernel_size, c] * kernel)

    return image, output

def firstDeriv(image,h):
    
    width,height = image.shape[0],image.shape[1]
    redVec = []
    greenVec = []
    blueVec = []
    channels = [redVec,greenVec,blueVec]
    results = [[],[],[]]

    for c in range(image.shape[2]):
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                channels[c].append(image[row][col][c])

    for c in channels:
        c += [0] * (2*h)
        c = np.array(c)

    kernelVector = np.array([-1, 0, 1])
    for c in range(len(channels)):
        for i in range(len(channels[c])-2*h):
            vec = np.array([channels[c][i],channels[c][i + h],channels[c][i + 2*h]])
            # vec = channels[c][i:i+3]
            summ = np.sum(kernelVector * vec) / (2*h)
            # print(summ)
            results[c].append(summ)
        results[c] = np.array(results[c])

    for i,r in enumerate(results):
        r = r.reshape(width,height)
        results[i] = r
    
    rmat = results[0]
    gmat = results[1]
    bmat = results[2]
    output = np.zeros((width,height,3))
    for row in range(width):
        for col in range(height):
            output[row][col][0] = rmat[row][col]
            output[row][col][1] = gmat[row][col]
            output[row][col][2] = bmat[row][col]
    return output

def secondDeriv(image,h):
    
    width,height = image.shape[0],image.shape[1]
    redVec = []
    greenVec = []
    blueVec = []
    channels = [redVec,greenVec,blueVec]
    results = [[],[],[]]

    for c in range(image.shape[2]):
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                channels[c].append(image[row][col][c])

    for c in channels:
        c += [0] * (4*h)
        c = np.array(c)

    kernelVector = np.array([-1, 2,0,-2,1])
    for c in range(len(channels)):
        for i in range(len(channels[c])-4*h):
            vec = np.array([channels[c][i],channels[c][i + h],channels[c][i + 2*h],channels[c][i + 3*h],channels[c][i + 4*h]])
            summ = np.sum(kernelVector * vec) / (2*h**3)
            results[c].append(summ)
        results[c] = np.array(results[c])

    for i,r in enumerate(results):
        r = r.reshape(width,height)
        results[i] = r
    
    rmat = results[0]
    gmat = results[1]
    bmat = results[2]
    output = np.zeros((width,height,3))
    for row in range(width):
        for col in range(height):
            output[row][col][0] = rmat[row][col]
            output[row][col][1] = gmat[row][col]
            output[row][col][2] = bmat[row][col]
    return output

def fourthDeriv(image,h):
    
    width,height = image.shape[0],image.shape[1]
    redVec = []
    greenVec = []
    blueVec = []
    channels = [redVec,greenVec,blueVec]
    results = [[],[],[]]

    for c in range(image.shape[2]):
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                channels[c].append(image[row][col][c])

    for c in channels:
        c += [0] * (4*h)
        c = np.array(c)

    kernelVector = np.array([1,-4,6,-4,1])
    for c in range(len(channels)):
        for i in range(len(channels[c])-4*h):
            vec = np.array([channels[c][i],channels[c][i + h],channels[c][i + 2*h],channels[c][i + 3*h],channels[c][i + 4*h]])
            summ = np.sum(kernelVector * vec) / (h**4)
            results[c].append(summ)
        results[c] = np.array(results[c])

    for i,r in enumerate(results):
        r = r.reshape(width,height)
        results[i] = r
    
    rmat = results[0]
    gmat = results[1]
    bmat = results[2]
    output = np.zeros((width,height,3))
    for row in range(width):
        for col in range(height):
            output[row][col][0] = rmat[row][col]
            output[row][col][1] = gmat[row][col]
            output[row][col][2] = bmat[row][col]
    return output

# Display the result
# im = firstDeriv(image,1)
# im = secondDeriv(image,2)
im = fourthDeriv(image,1)
im,res = GaussianSmoothing(im)
cv2.imshow("Derivative Image",res)
cv2.waitKey(0)