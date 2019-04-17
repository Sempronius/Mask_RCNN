#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:47:43 2019

@author: Home
"""
import random
import os
import cv2
import numpy as np

def windowing(im, win):
        # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

# of the images you have (in a list named files_key) , 
#find the info from deep_lesion csv which has been imported as DATAFRAME
def load_random_example(DL_df,n):
    #n is random
    #files_key is a list of key images you have (any number of deep lesion database)
    # DL_df is deep lesion csv turned into a dataframe.
    foldername = os.path.basename(os.path.dirname(n))
    filename = os.path.basename(n)
    comb = foldername + '_' + filename
    x=DL_df.loc[DL_df['File_name'] == comb]
    print('Filename')
    print(filename)
    return x

    
def load_im_with_default_windowing(win,image_path):
    

    ########################
    im = cv2.imread(image_path, -1) # -1 is needed for 16-bit image
    im = im.astype(np.float32, copy=False)-32768  
    im = windowing(im, win).astype(np.uint8)  # soft tissue window
    image = np.stack([im,im,im],axis=2) # THIS IS NECESSARY TO GET INTO RIGHT FORMAT FOR BELOW
    return image


def image_default_window(image, win):
        # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = image.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


        

    
    