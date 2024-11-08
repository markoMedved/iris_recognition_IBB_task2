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


@torch.inference_mode()
def extractIBBCode( polar,mask,R=1,P=8,W=8): #, mask):

    if polar is None:
        return None
        
    #image dimensions
    height, width = np.array(polar).shape
    #matrix with all the end decimal codes
    lbp_riu2 = np.zeros((height,width), np.int8)
    #variance matrix
    var_mtx = np.zeros((height, width), dtype=np.float32)

    #which pixels around center px
    steps = []
    for i in range(P):
        steps.append((int(round(R*np.cos(2*np.pi*(i+1)/P),0)),int(round(R*np.sin(2*np.pi*(i+1)/P),0))))
    
        
    #through image
    for i in range(height):
        for j in range(width):
            #if mask = 0, then just append 0
            if mask[i][j] == 0:
                continue
                
            binary = []

            center_px = polar[i][j]
            
            mew = 0
            #for all neighbour pixels
            pixels = []
            for s in steps:
                #if center px is on the border
                if i+s[0] >= 0 and i+s[0] < height and j+s[1] >= 0 and j+s[1] < width:
                    pixel = polar[i+s[0]][j+s[1]]
                    pixels.append(pixel)
                    
                    if pixel > center_px:
                        binary.append(1)
                    else:
                        binary.append(0)
                else:
                    binary.append(0)
                    pixels.append(0)
            

            #riu2 - smallest rotation, if uniform, else P+1
            #get all 0/1 changes in pattern (to check uniformity)
            if sum((binary[i] != binary[(i+1) % P] for i in range(P))) <=2:

                lbp_riu2[i,j] = sum(binary)
                
            else:
                lbp_riu2[i,j] = P+1

            #append var to var mtx
            pixels = np.array(pixels)
            mew = np.mean(pixels)
            var = np.mean((pixels - mew)**2)
            var_mtx[i,j] = var         

    
    #amount of bins in the histogram
    lbp_bins = P+2
    var_bins = 50

    #assign values to bins for the variance matrix
    var_mtx = np.clip(var_mtx,None,np.percentile(var_mtx, 99))
    
    var_mtx = (var_mtx * (var_bins - 1) / np.max(var_mtx)).astype(int)
    var_mtx = var_mtx.astype(int)

    window_size_x = height // int(np.sqrt(W))
    window_size_y = width // int(np.sqrt(W))
    joint_histogram = []
    for wx in range(0, height, window_size_x):
        for wy in range(0, width, window_size_y):
            # Define the window boundaries
            end_x = min(wx + window_size_x, height)
            end_y = min(wy + window_size_y, width)

            histogram = np.zeros((lbp_bins, var_bins), np.int32)

            #make a 2D histogram from lbp_riu2 and var_mtx
            for i in range(wx, end_x):
                for j in range(wy, end_y):
                    if mask[i][j] == 0:
                        continue
                    lbp_val = lbp_riu2[i, j]
                    var_val = var_mtx[i, j]
                    histogram[lbp_val, var_val] += 1
            histogram = histogram.flatten()
            joint_histogram.extend(histogram)
    
    return joint_histogram

def calculate_distance(feature_vector1, feature_vector2):
    score = 0.0

    for i in range(len(feature_vector1)):
        score += (feature_vector1[i] - feature_vector2[i]) ** 2 / (
            feature_vector1[i] + feature_vector2[i] + 1e-8
        )
    return score


filename = "images_polar/1_1L_s_1_im_polar.png"
mask_filename = "masks_polar/1_1L_s_1_mask_polar.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
# polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector1 = extractIBBCode(polar, mask)


filename = "images_polar/2_2L_s_2_im_polar.png"
mask_filename = "masks_polar/2_2L_s_2_mask_polar.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
# polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector2 = extractIBBCode(polar, mask)


print(calculate_distance(feature_vector1, feature_vector2))

