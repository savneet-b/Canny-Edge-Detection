import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from SKBlib import Convolve, Gaussian, Gaussian_Deriv, MagnitudeGradient, NonMaxSuppression, Hysteresis

def main():
    # Create a Tkinter root window and withdraw it to avoid showing the main window
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select an image file using a file dialog
    img_path = filedialog.askopenfilename(
        title="Select the Image",  # Window title
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png *.pgm")]  # Supported image file types
    )

    # If no file is selected, print a message and exit the function
    if not img_path:
        print("No file selected. Exiting...")
        return

    # Load the original image from the selected path for later visualization
    original_img = cv2.imread(img_path)
    cv2.imshow("Original Image", original_img)  # Display the original image

    # Convert the original image to grayscale since we are working exclusively with grayscale images
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    sigma = 0.6 # Set sigma value for the Gaussian filter, which controls the amount of smoothing

    # Generate Gaussian and Gaussian Derivative kernels for edge detection
    gaussian_kernel = Gaussian(sigma)  # Gaussian kernel for smoothing
    gaussian_deriv_kernel = Gaussian_Deriv(sigma)  # Derivative of Gaussian for edge detection

    # STEP 1: Compute Horizontal and Vertical Intensity Changes
    temp_horizontal = Convolve(grayscale_img, gaussian_kernel.T)  # Convolve with the transposed Gaussian kernel for horizontal changes
    horizontal_change = Convolve(temp_horizontal, gaussian_deriv_kernel)  # Convolve with the Gaussian derivative for horizontal gradient

    temp_vertical = Convolve(grayscale_img, gaussian_kernel)  # Convolve with the Gaussian kernel for vertical changes
    vertical_change = Convolve(temp_vertical, gaussian_deriv_kernel.T)  # Convolve with the transposed Gaussian derivative for vertical gradient

    # Display vertical and horizontal gradient images after convolution (switched)
    cv2.imshow('Vertical Image', horizontal_change.astype(np.uint8))  # Display horizontal gradient as vertical
    cv2.imshow('Horizontal Image', vertical_change.astype(np.uint8))  # Display vertical gradient as horizontal

    # STEP 2: Compute Magnitude and Gradient
    Gxy, Iangle = MagnitudeGradient(horizontal_change, vertical_change)  # Calculate the magnitude and angle of the gradient

    # Display Magnitude and Gradient
    cv2.imshow('Magnitude', np.uint8(np.clip(Gxy, 0, 255)))  # Show the magnitude of the gradient, clipped to valid pixel range
    cv2.imshow('Gradient Image', np.uint8(np.clip(Iangle * 255 / np.pi, 0, 255)))  # Display the angle as a gradient image, scaled to pixel values

    # STEP 3: Non-Maximal Suppression
    non_max_suppression = NonMaxSuppression(Gxy, Iangle)  # Apply non-maximal suppression to thin the edges
    cv2.imshow('Non-Maximal Suppression Image', np.uint8(non_max_suppression))  # Display the result of non-maximal suppression

    # STEP 4: Hysteresis and Edge-Linking
    edges = Hysteresis(non_max_suppression)  # Perform hysteresis thresholding to link edges
    cv2.imshow('Edges', np.uint8(edges))  # Display the final edge-detected image

    # Template Matching
    # Ask the user to select a template image for matching against the detected edges
    template_image_path = filedialog.askopenfilename(
        title="Select Template Image",  # Window title for template selection
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png *.pgm")]  # Supported image file types for template
    )

    # If a template image is selected, perform template matching
    if template_image_path:
        # Load the template image in grayscale
        template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        if template_image is not None:  # Check if the template image was loaded successfully
            # Convolve the template image with Gaussian kernels
            template_horizontal = Convolve(template_image, gaussian_kernel.T)  # Horizontal convolution
            template_vertical = Convolve(template_image, gaussian_kernel)  # Vertical convolution
            template_magnitude, template_gradient = MagnitudeGradient(template_horizontal, template_vertical)  # Calculate gradient of the template
            template_non_max_suppression = NonMaxSuppression(template_magnitude, template_gradient)  # Apply non-maximal suppression
            template_edges = Hysteresis(template_non_max_suppression)  # Perform hysteresis on the template edges

            # Template matching against the detected edges using normalized cross-correlation
            res = cv2.matchTemplate(edges, template_edges, cv2.TM_CCOEFF_NORMED)
            cv2.imshow('Template Matching Result', res)  # Display the result of template matching

    # Wait for user input to close all windows
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
