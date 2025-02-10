import numpy as np
import cv2

# First convolve the input image with vertical 1-D Gaussian
def Convolve(input_image, kernel_matrix):
    # Get dimensions of the input image and the kernel matrix
    img_height, img_width = input_image.shape
    kernel_height, kernel_width = kernel_matrix.shape

    # Initialize the result image with zeros, using the same dimensions as the input image
    result_image = np.zeros((img_height, img_width), dtype=float)

    for row in range(img_height): # Iterate over each pixel in the input image
        for col in range(img_width):
            accum_value = 0  # Initialize accumulator for the convolution result

            # Iterate over each element in the kernel
            for k_row in range(kernel_height):
                for k_col in range(kernel_width):
                    # Calculate the offset from the center of the kernel
                    offset_row = k_row - kernel_height // 2
                    offset_col = k_col - kernel_width // 2

                    # Check if the corresponding pixel in the input image is within bounds
                    if 0 <= row + offset_row < img_height and 0 <= col + offset_col < img_width:
                        # Accumulate the weighted sum for the pixel at (row, col)
                        accum_value += input_image[row + offset_row, col + offset_col] * kernel_matrix[k_row, k_col]
            result_image[row, col] = accum_value  # Store the convolution result at (row, col)
    return result_image  # Return the final convolved image


def Gaussian(sigma_value):
    # Calculate the size of the Gaussian kernel based on the sigma value (fixed value I have set is 0.6)
    kernel_radius = int(2.5 * sigma_value)  # Radius of the kernel
    kernel_size = 2 * kernel_radius + 1  # Total size of the kernel
    kernel = np.zeros((kernel_size, 1), dtype=float)  # Initialize the kernel

    total_sum = 0  # This variable normalizes the kernel

    # Calculate Gaussian values for each position in the kernel
    for i in range(kernel_size):
        kernel[i] = np.exp(-((i - kernel_radius) ** 2) / (2 * sigma_value ** 2))  # Gaussian formula
        total_sum += kernel[i]  # Sum for normalization

    # Normalize the kernel values so they sum to 1
    kernel /= total_sum
    return kernel  # Return the Gaussian kernel

def Gaussian_Deriv(sigma_value):
    kernel_radius = int(2.5 * sigma_value)   # Calculate the size of the Gaussian derivative kernel based on the sigma value
    kernel_size = 2 * kernel_radius + 1
    derivative_kernel = np.zeros((kernel_size, 1), dtype=float)  # Initialize derivative kernel

    total_sum = 0  # Variable to normalize the kernel
    for i in range(kernel_size): # Calculate Gaussian derivative values for each position in the kernel
        derivative_kernel[i] = -1 * (i - kernel_radius) * np.exp(
            -((i - kernel_radius) ** 2) / (2 * sigma_value ** 2))  # Derivative formula
        total_sum -= i * derivative_kernel[i]  # Sum for normalization

    # Normalize the kernel values so they sum to 1
    derivative_kernel /= total_sum
    return derivative_kernel  # Return the Gaussian derivative kernel


def calculate_intensity_changes(gray_image, sigma_value):
    # Compute the intensity changes through convolution, Gaussian derivative, and transpose
    # Create vertical and horizontal Gaussian kernels
    vertical_kernel = create_gaussian_kernel(sigma_value)  # Vertical Gaussian kernel
    horizontal_kernel = vertical_kernel.T  # Horizontal Gaussian kernel (transpose of vertical 1-D Gaussian used for Horizontal)

    # Create vertical and horizontal Gaussian derivative kernels
    vertical_derivative = create_gaussian_derivative(sigma_value)  # Vertical Gaussian derivative
    horizontal_derivative = vertical_derivative.T  # Horizontal Gaussian derivative (transpose)

    # Smooth the image first in both horizontal and vertical directions because it reduces noise
    temp_horizontal = apply_convolution(gray_image, vertical_kernel)  # Apply vertical Gaussian convolution
    temp_vertical = apply_convolution(gray_image, horizontal_kernel)  # Apply horizontal Gaussian convolution

    # Apply Gaussian derivatives in each direction
    horizontal_gradient = apply_convolution(temp_horizontal,
                                            horizontal_derivative)  # Gaussian Derivative in horizontal direction
    vertical_gradient = apply_convolution(temp_vertical, vertical_derivative)  # Gaussian Derivative in vertical direction
    return horizontal_gradient, vertical_gradient  # Return both vertical and horizontal gradients

# Use the vertical and horizontal components to get the magnitude(Gxy) and direction(Iangle) images
def MagnitudeGradient(horizontal_grad, vertical_grad):
    # Calculate the magnitude of the gradient using the horizontal and vertical components
    magnitude = np.sqrt(horizontal_grad ** 2 + vertical_grad ** 2)  # Compute the gradient magnitude
    angle = np.arctan2(vertical_grad, horizontal_grad)  # Compute the angle of the gradient

    return magnitude, angle  # Return the magnitude and angle
# Both Gxy and Iangle are double images

# Use the Gxy and Iangle calculated above the perform Step 3: non maximal suppression
def NonMaxSuppression(magnitude_img, angle_img):
    # Get the dimensions of the magnitude image
    img_height, img_width = magnitude_img.shape
    suppressed_img = magnitude_img.copy()  # Initialize suppressed image to the copy of the magnitude image

    # Iterate over each pixel in the magnitude image
    for row in range(img_height):
        for col in range(img_width):
            theta = angle_img[row, col]  # Get the angle for the current pixel

            # Adjust theta to be positive, then convert to degrees
            if theta < 0:
                theta += np.pi
            theta = theta * (180 / np.pi)  # Convert radians to degrees

            # Check gradient direction and apply non-maximum suppression
            if (theta <= 22.5 or theta > 157.5):  # North-South direction
                if (row > 0 and magnitude_img[row, col] < magnitude_img[row - 1, col]) or (
                        row < img_height - 1 and magnitude_img[row, col] < magnitude_img[row + 1, col]):
                    suppressed_img[row, col] = 0  # Suppress non-maximum
            elif (22.5 < theta <= 67.5):  # Northwest-Southeast direction
                if (row > 0 and col > 0 and magnitude_img[row, col] < magnitude_img[row - 1, col - 1]) or (
                        row < img_height - 1 and col < img_width - 1 and magnitude_img[row, col] < magnitude_img[
                    row + 1, col + 1]):
                    suppressed_img[row, col] = 0  # Suppress non-maximum
            elif (67.5 < theta <= 112.5):  # East-West direction
                if (col > 0 and magnitude_img[row, col] < magnitude_img[row, col - 1]) or (
                        col < img_width - 1 and magnitude_img[row, col] < magnitude_img[row, col + 1]):
                    suppressed_img[row, col] = 0  # Suppress non-maximum
            elif (112.5 < theta <= 157.5):  # Northeast-Southwest direction
                if (row > 0 and col < img_width - 1 and magnitude_img[row, col] < magnitude_img[row - 1, col + 1]) or (
                        row < img_height - 1 and col > 0 and magnitude_img[row, col] < magnitude_img[row + 1, col - 1]):
                    suppressed_img[row, col] = 0  # Suppress non-maximum

    return suppressed_img  # Return the image after non-maximum suppression

# Use the non-maximal image to perform edge-linking with hysteresis
def Hysteresis(suppressed_img):
    # Get dimensions of the suppressed image
    img_height, img_width = suppressed_img.shape
    edge_image = suppressed_img.copy()  # Initialize edge image as a copy of the suppressed image

    # Determine high and low thresholds based on the suppressed image
    high_threshold = np.percentile(suppressed_img, 90)  # High threshold is the 90th percentile
    low_threshold = 0.2 * high_threshold  # Low threshold is 20% of the high threshold

    # Iterate over each pixel in the suppressed image
    for row in range(img_height):
        for col in range(img_width):
            if suppressed_img[row, col] > high_threshold:
                edge_image[row, col] = 255  # Strong edge
            elif suppressed_img[row, col] > low_threshold:
                edge_image[row, col] = 115  # Weak edge
            else:
                edge_image[row, col] = 0  # Non-edge

    final_edges = edge_image.copy()  # Create a copy for final edge determination
    # Iterate over each pixel to refine weak edges
    for row in range(img_height):
        for col in range(img_width):
            # Check if the current pixel is a weak edge
            if edge_image[row, col] == 115:
                # Check all 8 neighbors for any strong edges
                if (0 < row < img_height - 1 and 0 < col < img_width - 1):  # Ensure we don't go out of bounds
                    neighbors = [
                        edge_image[row - 1, col], edge_image[row, col + 1], edge_image[row + 1, col],
                        edge_image[row, col - 1], edge_image[row - 1, col + 1], edge_image[row + 1, col + 1],
                        edge_image[row + 1, col - 1], edge_image[row - 1, col - 1]
                    ]
                    # If any neighboring pixel is strong, promote weak edge to strong edge
                    if any(neighbor == 255 for neighbor in neighbors):
                        final_edges[row, col] = 255  # Promote to strong edge
                    else:
                        final_edges[row, col] = 0  # Suppress to non-edge

    return final_edges  # Return the final image after hysteresis (Edges)


