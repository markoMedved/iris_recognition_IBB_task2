
import cv2
import numpy as np


def calculate_lbp(image, radius=1, num_points=8):
    """
    Calculate the LBP (Local Binary Pattern) for each pixel in the image.

    Parameters:
    - image: 2D numpy array of the grayscale image.
    - radius: Radius of the circle used for LBP.
    - num_points: Number of surrounding points in the circle.

    Returns:
    - lbp_image: 2D numpy array of the same size as image, containing LBP codes.
    """
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    for row in range(radius, image.shape[0] - radius):
        for col in range(radius, image.shape[1] - radius):
            center_pixel = image[row, col]
            binary_string = ''
            for n in range(num_points):
                theta = 2 * np.pi * n / num_points
                x = int(row + radius * np.sin(theta))
                y = int(col + radius * np.cos(theta))
                neighbor_pixel = image[x, y]
                binary_string += '1' if neighbor_pixel >= center_pixel else '0'
            lbp_image[row, col] = int(binary_string, 2)
    return lbp_image

def lbp_histogram(lbp_image, mask, num_bins=256):
    """
    Create a histogram of the LBP image, only considering pixels where mask == 1.

    Parameters:
    - lbp_image: 2D numpy array containing LBP codes for each pixel.
    - mask: 2D numpy array of the same size as lbp_image; binary mask.
    - num_bins: Number of bins for the histogram.

    Returns:
    - hist: LBP histogram of the masked region.
    """
    # Apply mask to the LBP image
    masked_lbp = lbp_image[mask == 1]
    
    # Check if masked_lbp contains any values to avoid empty histograms
    if masked_lbp.size == 0:
        # Return an empty histogram if no valid data is available
        hist = np.zeros(num_bins)
    else:
        # Compute histogram for valid masked region
        hist, _ = np.histogram(masked_lbp, bins=num_bins, range=(0, num_bins))
        hist = hist / (hist.sum() + 1e-6)  # Normalize and handle zero sum safely

    return hist

def chi_square_distance(hist1, hist2):
    """
    Compute the Chi-Square distance between two histograms.

    Parameters:
    - hist1: LBP histogram for the first image.
    - hist2: LBP histogram for the second image.

    Returns:
    - distance: Chi-square distance between histograms.
    """
    chi_sq = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-6))
    return chi_sq







filename="images_polar/1_1L_s_1_im_polar.png"
mask_filename = "masks_polar/1_1L_s_1_mask_polar.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
#polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

im1 = calculate_lbp(polar, radius=1, num_points=8)
h1 = lbp_histogram(im1,mask)


filename="images_polar/2_2L_s_2_im_polar.png"
mask_filename = "masks_polar/2_2L_s_2_mask_polar.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
#polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)


im2 = calculate_lbp(polar, radius=1, num_points=8)
h2 = lbp_histogram(im2,mask)


print(chi_square_distance(h1,h2))