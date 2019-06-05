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
#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190508T0729/mask_rcnn_deep_lesion_0100.h5'
WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190604T1631/mask_rcnn_deep_lesion_0010.h5'
###############

########################################## NEED TO CHAnge DEPENDING ON HOW YOU TRAINED
image_size = 512

from random import randint
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import numpy
from PIL import Image, ImageDraw
import cv2

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
    NUM_CLASSES = 1 + 8  # Background + 9 classes (see below)
    # Number of training steps per epoch
    # Skip detections with < 90% confidence
    ######################## ADDED THIS , default was 800, 1024. But this was crashing the system. 
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = image_size
    IMAGE_MAX_DIM = image_size




    # Skip detections with < 90% confidence

    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.8
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




###########################################################
model.load_weights(WEIGHT_DIR, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"]) 
##############################################################
   
    
annotations = json.load(open(os.path.join(DEXTR_DIR, "data.json")))
annotations_seg = annotations['annotations']
################ Create a loop to go through all the test images.
#
#for x in range(0,len(annotations['images'])):
#    train_valid_test = annotations['images'][x]['Train_Val_Test']
#    if train_valid_test == 3:

#challenge = False
#while challenge == False:
#     
#    b = randint(0,len(annotations_seg))
#    image_info = annotations['images'][b]
#    #train_valid_test = annotations['images'][b]['Train_Val_Test']
#    train_valid_test = annotations['images'][b]['Train_Val_Test']
#    area = annotations_seg[b]['area']
#    print('Searching for test file with segmentation area 500 or more')
#    if train_valid_test == 1 and area >= 250:
#        challenge = True
        
b = randint(0,len(annotations_seg))
image_info = annotations['images'][b]
train_valid_test = annotations['images'][b]['Train_Val_Test']

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
############ default windowing:
image_o = util.load_im_with_default_windowing(win,image_path)

#im = cv2.imread(image_path, -1)
#image = np.stack([im,im,im],axis=2)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


r = model.detect([image_o], verbose=1)[0]

############################################################################### 
################# Need to deal with situation of MULTIPLE DETECTIONS. if r['masks'] = 2 or more. 
mask = r['masks']
print(r['masks'].shape)
assert r['masks'].size >= 1

############################################## Allows multiple detections...
if mask.shape[-1] > 0:
    # We're treating all instances as one, so collapse the mask into one layer

    mask = (np.sum(mask, -1, keepdims=True) >= 1)
#############################################

########################## This crashes my computer:
########## CREATE a plot of 3 images, original, labeled, and mask. 

fig, ax = plt.subplots(1,3,figsize=(75, 25))
plt.set_cmap('gray')

########## moved out of the loop
plt.ioff() ### TURN OFF INTERACTIVE MODE.   

mask_stack = np.stack([mask[...,0]==0,mask[...,0]==0,mask[...,0]==0],axis=2)
masked_img = np.where(mask_stack, image_o,[0,0,255])
masked_img = masked_img.astype(np.uint8)
ax[0].imshow(image_o) # original
ax[0].axis('off')


##################################### Ground truth.
polygon = annotations_seg[b]['segmentation']

img = Image.new('L', (512, 512), 0)
ImageDraw.Draw(img).polygon(polygon[0], outline=1, fill=1)
mask_original = numpy.array(img)

mask_original_stack = np.stack([mask_original==0,mask_original==0,mask_original==0],axis=2)

masked_original_img = np.where(mask_original_stack, image_o,[0,0,255])
masked_original_img = masked_original_img.astype(np.uint8)
################################# WILL RUN INTO TROUBLE WITH MULTIPLE MASK (polygon being a list greater than one.)

ax[1].imshow(masked_original_img)# Labeled
ax[1].axis('off')
###################################### Need to make ground truth image.

####
ax[2].imshow(masked_img) #mask
ax[2].axis('off')
####




plt.savefig(image_info['File_name'],bbox_inches='tight')    
print('Saving File')
print(r)






# Color splash

#splash = color_splash(image, r['masks'])

# Save output

#file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#skimage.io.imsave(file_name, splash)



############################ NEED TO LOAD ACTUAL IMAGE AND SHOW CALCULATED LESION VERSUS ACTUAL LESION. 