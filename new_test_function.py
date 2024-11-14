import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import *
from matplotlib import pyplot as plt


def extractIBBCode(polar, mask, R=3, P=24, W=32):
    height, width = np.array(polar).shape

    steps = [(int(round(R * np.cos(2 * np.pi * (i + 3) / P), 0)),
            int(round(R * np.sin(2 * np.pi * (i + 3) / P), 0))) for i in range(P)]
    
    lbp = np.zeros((height, width), np.int32)
    feature_vector = []

    # Divide the image into W windows
    window_size_x = height // int(np.sqrt(W))
    window_size_y = width // int(np.sqrt(W))

    for wx in range(0, height, window_size_x):
        for wy in range(0, width, window_size_y):
            # Define the window boundaries
            end_x = min(wx + window_size_x, height)
            end_y = min(wy + window_size_y, width)
            
            histogram = np.zeros(2 ** P, np.int32)

            for i in range(wx, end_x):
                for j in range(wy, end_y):
                    if mask[i][j] == 0:
                        continue
                    binary = []

                    # Compute LBP pattern
                    for k in range(P):
                        ni, nj = i + steps[k][0], j + steps[k][1]
                        if 0 <= ni < height and 0 <= nj < width and polar[ni][nj] >= polar[i][j]:
                            binary.append(1)
                        else:
                            binary.append(0)

                    # Rotate to minimum binary pattern for rotation invariance
                    minVal = 255
                    min_binary = binary
                    for m in range(P):
                        val = sum(2 ** k * binary[(k + m) % P] for k in range(P))
                        if val < minVal:
                            minVal = val
                            min_binary = [binary[(k + m) % P] for k in range(P)]

                    # Update the LBP and histogram
                    lbp[i][j] = sum(2 ** k * binary[k] for k in range(P))
                    histogram[sum(2 ** k * min_binary[k] for k in range(P))] += 1

            # Append this window's histogram to the feature vector
            feature_vector.extend(histogram)

    return feature_vector



def calculate_distance(feature_vector1, feature_vector2):
    score = 0.0

    for i in range(len(feature_vector1)):
        score += (feature_vector1[i] - feature_vector2[i]) ** 2 / (feature_vector1[i] + feature_vector2[i] + 1e-8)

    return score


filename = "images_polar/1_1L_s_4_im_polar.png"
#filename = "data/1_1L_s_1.jpg"
mask_filename = "masks_polar/1_1L_s_4_mask_polar.png"
#mask_filename = "masks/1_1L_s_1_seg_mask.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
#polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector1, jh1 = extractIBBCode(polar, mask)


filename = "images_polar/1_1L_s_1_im_polar.png"
mask_filename = "masks_polar/1_1L_s_1_mask_polar.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
# polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector2, jh2 = extractIBBCode(polar, mask)


print(calculate_distance(feature_vector1, feature_vector2))


#plt.imshow(jh1)
#plt.colorbar()  # Add color bar to indicate the counts
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('2D Histogram (imshow)')
#plt.show()
