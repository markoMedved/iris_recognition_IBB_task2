from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_feature_vector(polar,mask):
    codeBinaries = []
    bin_row = []
    height, width = np.array(polar).shape

    #LPB 8,1
    R = 1
    P =8

    codeBinaries = []
    bin_row = []
    height, width = np.array(polar).shape

    steps = []
   
    for i in range(P):
        steps.append((int(round(R*np.cos(2*np.pi*(i+3)/P),0)),int(round(R*np.sin(2*np.pi*(i+3)/P),0))))
    print(steps)
   

    for i in range(height):
        bin_row = []
        for j in range(width):
            if mask[i][j] == 0:
                bin_row.append(0)
                
                continue
        
            binary = []

            center_px = polar[i][j]
            
            for s in steps:
                if i+s[0] > 0 and i+s[0] < height and j+s[1] > 0 and j+s[1] < width:
                    pixel = polar[i+s[0]][j+s[1]]

                    if pixel > center_px:
                        binary.append(1)
                    else:
                        binary.append(0)
            
          
           
            if sum((binary[i] != binary[i-1] for i in range(1, len(binary)))) <=2:

                rotations = [binary[i:] + binary[:i] for i in range(len(binary))]
                
                min_rotation = min(rotations)
                dec = sum(val * (2 ** b) for b, val in enumerate(binary[::-1]))
                bin_row.append(dec)
           
            else:
                bin_row.append(P+1)
              

        codeBinaries.append(bin_row)

    codeBinaries = np.array(codeBinaries)

    num_windows = (4,4)
    window_height = height // num_windows[0]
    window_width = width // num_windows[1]

    num_bins =P +2


    histograms = []


    for i in range(num_windows[0]):
        for j in range(num_windows[1]):
            window = codeBinaries[i*window_height:(i+1)*window_height, j*window_width:(j+1)*window_width]
            histogram,_ = np.histogram(window, bins=num_bins)
            if np.count_nonzero(histogram) > 1:
                histograms.extend(histogram)
            else:
                histograms.extend(np.zeros(len(histogram)))



    feature_vector = np.array(histograms)
    return feature_vector,np.array(codeBinaries,dtype=np.uint8)


def calculate_distance(feature_vector1,feature_vector2):
    score = 0.0

    for i in range(len(feature_vector1)):
    
        score += (feature_vector1[i] - feature_vector2[i]) ** 2 / (feature_vector1[i] + feature_vector2[i] + 1e-8) 
    return score


filename="data/1_1R_s_1.jpg"
mask_filename = "masks/1_1R_s_1_seg_mask.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector1,im1 = get_feature_vector(polar,mask)



filename="data/1_1L_s_3.jpg"
mask_filename = "masks/1_1L_s_3_seg_mask.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector2,im2 = get_feature_vector(polar,mask)


print(calculate_distance(feature_vector1,feature_vector2))

"""
filename="data/2_2L_s_2.jpg"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)



feature_vector2,im2 = get_feature_vector(polar)


print(calculate_distance(feature_vector1,feature_vector2))

plt.plot(feature_vector1, label="feature_vector1")
plt.plot(feature_vector2, label="feature_vector2")
plt.legend()
#plt.show()
"""


def calculate_lbpriu2(image, mask, P=8, R=1):
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=int)
    steps = [(int(round(R * np.cos(2 * np.pi * i / P))),
              int(round(R * np.sin(2 * np.pi * i / P)))) for i in range(P)]
    
    for i in range(R, height - R):
        for j in range(R, width - R):
            # Skip masked pixels
            if mask[i, j] == 0:
                lbp_image[i, j] = 0
                continue

            center_px = image[i, j]
            binary_pattern = []
            
            for dy, dx in steps:
                neighbor_y, neighbor_x = i + dy, j + dx
                # Check if the neighbor is within bounds and is unmasked
                if 0 <= neighbor_y < height and 0 <= neighbor_x < width and mask[neighbor_y, neighbor_x] != 0:
                    neighbor_px = image[neighbor_y, neighbor_x]
                    binary_pattern.append(1 if neighbor_px >= center_px else 0)
                else:
                    binary_pattern.append(0)  # Consider masked or out-of-bounds pixels as 0
            
            # Rotate to minimum binary pattern for rotation invariance
            min_pattern = min(int(''.join(map(str, binary_pattern[k:] + binary_pattern[:k])), 2)
                              for k in range(P))
            
            # Count transitions (check if it's uniform)
            transitions = sum(binary_pattern[m] != binary_pattern[(m + 1) % P] for m in range(P))
            lbp_image[i, j] = min_pattern if transitions <= 2 else P + 1  # P+1 is the non-uniform bin
    
    return lbp_image


im = calculate_lbpriu2(polar, mask, P=8, R=1)
cv2.imshow("img",im1)
cv2.waitKey(0)
