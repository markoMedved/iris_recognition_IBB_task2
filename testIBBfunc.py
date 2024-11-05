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


@torch.inference_mode()
def extractIBBCode( polar,mask,R=1,P=8,W=8): #, mask):

    if polar is None:
        return None
        
    #image dims
    height, width = np.array(polar).shape
    #matrix with at the end decimal coddes
    lbp_riu2 = np.zeros((height,width), np.int8)
    #var mtx
    var_mtx = np.zeros((height, width), dtype=np.float32)



    #which pixels around center px
    steps = []


    #LPB 8,1
    for i in range(P):
        steps.append((int(round(R*np.cos(2*np.pi*(i+1)/P),0)),int(round(R*np.sin(2*np.pi*(i+1)/P),0))))
        
    #amout of bins in the histogram
    lbp_bins = P+2
    var_bins = 10

    #histogram
    joint_histogram = np.zeros((lbp_bins, var_bins), dtype=int)

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
                if i+s[0] > 0 and i+s[0] < height and j+s[1] > 0 and j+s[1] < width:
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
            if sum((binary[i] != binary[(i+i) % P] for i in range(P))) <=2:
                
                val = sum(binary)
                lbp_riu2[i,j] = sum(binary)
                
            else:
                val = P+1
                lbp_riu2[i,j] = P+1

            pixels = np.array(pixels)
            mew = np.mean(pixels)
            var = np.mean((pixels - mew)**2)
            var_mtx[i,j] = var         
            
            var = (var * (var_bins - 1) / np.max(var_mtx)).astype(int)
            #joint_histogram[val, var]+=1

    

    lbp_riu2 = np.array(lbp_riu2)

    var_mtx = np.array(var_mtx)
    var_mtx = (var_mtx * (var_bins - 1) / np.max(var_mtx)).astype(int)
    var_mtx = var_mtx.astype(int)



    joint_histogram = np.zeros((lbp_bins, var_bins), dtype=int)
    for i in range(height):
        for j in range(width):
            if mask[i][j] == 0:
                continue
            lbp_val = lbp_riu2[i, j]
            var_val = var_mtx[i, j]
            joint_histogram[lbp_val, var_val] += 1

    """"
    num_windows = (W,W)
    window_height = height // num_windows[0]
    window_width = width // num_windows[1]

    

    histograms = []


    for i in range(num_windows[0]):
        for j in range(num_windows[1]):
            window = lbp_riu2[i*window_height:(i+1)*window_height, j*window_width:(j+1)*window_width]
            histogram,_ = np.histogram(window.ravel(), bins=num_bins-1)
            if np.count_nonzero(histogram) > 1:
                histograms.extend(histogram)
            else:
                histograms.extend(np.zeros(len(histogram)))
    """


    feature_vector = joint_histogram.flatten()
    feature_vector = feature_vector 

    return feature_vector


@torch.inference_mode()
def matchIBBCodes( codes1, codes2): #, mask1, mask2):
    score = 0.0

    for i in range(len(codes1)):

        score += (codes1[i] - codes2[i]) ** 2 / (codes1[i] + codes2[i] + 1e-8) 

    
    return score




def calculate_distance(feature_vector1,feature_vector2):
    score = 0.0

    for i in range(len(feature_vector1)):
    
        score += (feature_vector1[i] - feature_vector2[i]) ** 2 / (feature_vector1[i] + feature_vector2[i] + 1e-8) 
    return score


filename="data/1_1L_s_1.jpg"
mask_filename = "masks/1_1L_s_1_seg_mask.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector1 = extractIBBCode(polar,mask)



filename="data/1_1L_s_4.jpg"
mask_filename = "masks/1_1L_s_4_seg_mask.png"
polar = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
polar = cv2.resize(polar, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

feature_vector2 = extractIBBCode(polar,mask)


print(calculate_distance(feature_vector1,feature_vector2))