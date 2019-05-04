#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:24:06 2019

@author: sempronius
"""

#NEED TO SET MODEL_DIR
MODEL_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/'
#NEED TO SET WEIGHT DIRECTORY

#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/balloon20190428T1350/mask_rcnn_balloon_0030.h5'


#################
WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190430T2209/mask_rcnn_deep_lesion_0100.h5'
###############

from random import randint
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import utilities as util

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

### Path to DEXTR
BASE_DIR = os.path.abspath("../../../")

DEXTR_DIR = os.path.join(BASE_DIR,"DEXTR-PyTorch_p")


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Deep_Lesion"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 #seems highest is 512 otherwise memory problems. 
    GPU_COUNT = 1
    BATCH_SIZE = IMAGES_PER_GPU*GPU_COUNT
    #According to the Keras documentation recommendation:
#STEPS_PER_EPOCH = NUMBER_OF_SAMPLES/BATCH_SIZE
#According to the MaskRCNN code:
#BATCH_SIZE = IMAGES_PER_GPU*GPU_COUNT
#That means that:
#STEPS_PER_EPOCH = NUMBER_OF_SAMPLES/(IMAGES_PER_GPU*GPU_COUNT)
    VALIDATION_STEPS = 250/BATCH_SIZE 
    STEPS_PER_EPOCH = 1000/BATCH_SIZE 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # Background + 9 classes (see below)
    # Number of training steps per epoch
    # Skip detections with < 90% confidence
    ######################## ADDED THIS , default was 800, 1024. But this was crashing the system. 
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence

    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3


class InferenceConfig(BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
print(config)


model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    
#model.load_weights(WEIGHT_DIR, by_name=True)
model.load_weights(WEIGHT_DIR, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"]) 

   
    
annotations = json.load(open(os.path.join(DEXTR_DIR, "data.json")))
annotations_seg = annotations['annotations']
b = randint(0,len(annotations_seg))
image_info = annotations['images'][b]

print("Running on:")
print(image_info['File_name'])
print(image_info['File_path'])

image_path = os.path.join(DEXTR_DIR,image_info['File_path'])
            
win = image_info['DICOM_windows']
win = win.split(",") # turn it into a list. 
        
win = list(map(float, win)) # turn the list of str, into a list of float (in case of decimals)
win = list(map(int, win)) # turn the list of str, into a list of int
        
        
################### image format should be: unit8 rgb 
#image = skimage.io.imread(image_path)
image = util.load_im_with_default_windowing(win,image_path)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


r = model.detect([image], verbose=1)[0]

mask = r['masks']
mask1 = mask*255
red = mask*0
red = red.astype(np.uint8)
red = red[...,0]
mask1 = mask1[...,0]
mask2 = np.stack([mask1,red,red],axis=2)

plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window
#plt.imshow(mask2)

#plt.imshow(image)

########## CREATE a plot of 3 images, original, labeled, and mask. 
fig, ax = plt.subplots(1,3,figsize=(75, 25))
plt.set_cmap('gray')
########## moved out of the loop
plt.ioff() ### TURN OFF INTERACTIVE MODE.   

masked_img = np.where(mask[...,None]==0, image,[0,0,255])
ax[0].imshow(masked_img) # original
ax[0].axis('off')
ax[1].imshow(mask2)# Labeled
ax[1].axis('off')

####
ax[2].imshow(image) #mask
ax[2].axis('off')
####




#plt.savefig(dir_comb,bbox_inches='tight')    
#print('Saving File')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    mask1 = mask*255
    red = mask*0
    red = red.astype(np.uint8)
    red = red[...,0]
    mask1 = mask1[...,0]
    mask2 = np.stack([mask1,red,red],axis=2)
    imgplot = plt.imshow(mask2)
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
        
    else:
        splash = gray.astype(np.uint8)
    return splash



# Color splash

#splash = color_splash(image, r['masks'])

# Save output

#file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#skimage.io.imsave(file_name, splash)



############################ NEED TO LOAD ACTUAL IMAGE AND SHOW CALCULATED LESION VERSUS ACTUAL LESION. 