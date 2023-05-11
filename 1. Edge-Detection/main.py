import cv2
import numpy as np
# Load the image
image = cv2.imread("images\SuzannesWithBalls.png")
# Rescale image to half its size
image = cv2.resize(image, (500,500), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def grayScaleSmoothing(imager):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothing Matrix
    smoothGauss = np.zeros(gray.shape, dtype=np.uint8)
    for row in range(1, len(gray)-1):
        for col in range(1, len(gray[0])-1):
            pix = 4 * gray[row][col] + 2 * gray[row][col+1] + 2 * gray[row][col-1] + 2 * gray[row-1][col] + 2 * \
                gray[row+1][col] + gray[row-1][col+1] + gray[row-1][col-1] + \
                gray[row+1][col-1] + gray[row+1][col+1]
            # pix = 4 * gray[row][col]
            # pix = int(pix / 4)
            smoothGauss[row][col] = pix / 16
    return gray, smoothGauss


def GaussianSmoothing(image):
    # Define the Gaussian kernel
    kernel_size = 5
    sigma = 5
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

                # Note: We want gaussian filter to work on colored images, so we may "fix" the color channels xDDD,
                # and use convolution for each channel separately,
                # this is similar to turning image to black and white

                # Note: * is elementwise multiplication, see Extra/elementwise for more
                output[i, j, c] = np.sum(
                    padded_image[i:i+kernel_size, j:j+kernel_size, c] * kernel)

    return image, output


def SobelEdgeDetection(image):
    # Turn image into black and white
    gray = cv2.cvtColor(smoothColoredGauss, cv2.COLOR_BGR2GRAY)

    kernelY = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    kernelX = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    # Pad the image and avoid overflows
    output = np.zeros_like(gray)
    padding = kernelX.shape[0] // 2
    padded_image = np.zeros(
        (gray.shape[0] + 2*padding, gray.shape[1] + 2*padding), dtype=np.uint8)
    padded_image[padding:-padding, padding:-padding] = gray
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            summX = np.sum(
                padded_image[i:i+kernelX.shape[0], j:j+kernelX.shape[0]] * kernelX)
            summY = np.sum(
                padded_image[i:i+kernelX.shape[0], j:j+kernelX.shape[0]] * kernelY)
            mx = max(summX, summY)
            # output[i, j] = abs(mx) + 255 * (abs(mx))/255
            output[i,j] = (summX ** 2 + summY ** 2) ** (1/2)
    return output

def LaplacianEdgeDetection(image):
    # Turn image into black and white
    gray = cv2.cvtColor(smoothColoredGauss, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])

    # Pad the image and avoid overflows
    output = np.zeros_like(gray)
    padding = kernel.shape[0] // 2
    padded_image = np.zeros(
        (gray.shape[0] + 2*padding, gray.shape[1] + 2*padding), dtype=np.uint8)
    padded_image[padding:-padding, padding:-padding] = gray
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            summ = np.sum(
                padded_image[i:i+kernel.shape[0], j:j+kernel.shape[0]] * kernel)
            # print(summ)
            # output[i, j] = abs(summ) + 255 * (abs(summ))/255
            output[i, j] = summ * summ

    return output


colored, smoothColoredGauss = GaussianSmoothing(image)
edged = SobelEdgeDetection(smoothColoredGauss)
# edged = LaplacianEdgeDetection(smoothColoredGauss)
edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)

# Display the result
result = cv2.hconcat([colored,smoothColoredGauss, edged])
cv2.imshow("Blurred Image + Edge detection", result)
cv2.waitKey(0)
