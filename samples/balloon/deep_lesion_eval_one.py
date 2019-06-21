#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:32:13 2019

@author: sempronius
"""

#https://github.com/CrookedNoob/Mask_RCNN-Multi-Class-Detection/blob/master/inspect_food_model.ipynb
# Create new deep_lesion_eval based off of this:

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn import utils_DL
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


MODEL_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/'
#NEED TO SET WEIGHT DIRECTORY

#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/balloon20190428T1350/mask_rcnn_balloon_0030.h5'


#################
#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190508T0729/mask_rcnn_deep_lesion_0100.h5'
WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190605T0954/mask_rcnn_deep_lesion_0020.h5'
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
from mrcnn import model as modellib, utils_DL
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
    NUM_CLASSES = 1 + 1  # Background + 9 classes (see below)
    # Number of training steps per epoch
    # Skip detections with < 90% confidence
    ######################## ADDED THIS , default was 800, 1024. But this was crashing the system. 
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = image_size
    IMAGE_MAX_DIM = image_size




    # Skip detections with < 90% confidence

    DETECTION_MAX_INSTANCES = 2
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3


class InferenceConfig(BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
print(config)

TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class Deep_Lesion_Dataset(utils_DL.Dataset):
    
    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Abnormality", 1, "Lesion") # "Bone"

        
        
        
        
        ##################### UNKNOWN CASES, WE WILL LEAVE THESE OUT. 
        #self.add_class("-1", 9, "-1") #Can i have a negative number here? This is straight from Deep Lesion format. 
     
#########################
        # Train or validation dataset?
        
        #assert subset in ["train", "val"]
        #dataset_dir = os.path.join(dataset_dir, subset)

        ###### ---> SAVED under "Image" in data.json, there is variable called "train_val_test" 
        #which has a int value (0-2 or 1-3) indicating it is training, test or validation. 
        # NEED TO FIGURE OUT HOW TO USE THIS TO ASSIGN TRAIN, VALIDATION, TEST. 
##########################


        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        #annotations = json.load(open(os.path.join(DEXTR_DIR, "data_including_unknown_one_lesion.json")))
        annotations = json.load(open(os.path.join(DEXTR_DIR, "data_including_unknown.json")))
        #annotations_seg = list(annotations.values())  # don't need the dict keys

        annotations_seg = annotations['annotations']
        
        #annotations_seg = segmentation[2]

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        #annotations_seg = [a for a in annotations_seg if a['segmentation']]


        
        b=0
        for a in annotations_seg:
            image_info = annotations['images'][b]
            win = annotations_seg[b]['Windowing']
            image_id = annotations['images'][b]['id']
            image_cat = annotations['categories'][b]['category_id']

            
            
            
            ############
            ############
            ##########  Copy Food.py
            polygons=[]
            objects=[]
            #for r in a['regions'].values():
            for r in a['regions']:
                polygons.append(r['shape_attributes'])
            # print("polygons=", polygons)
                #objects.append(r['region_attributes'])
                objects.append({"Abnormality":1}) 
            
            #class_ids = [int(n['Lesion']) for n in objects]
            # Since we are only considering one type of abnormality, "Lesion", 1
            class_ids = [int(n['Abnormality']) for n in objects]

                
        
            
            train_valid_test = annotations['images'][b]['Train_Val_Test']
            #### Must use index 'b' before here, because after this point it
            # will point to next image/index/
            b=b+1
            
            
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            ######## polygons 
            # needs to be a list of dictionaries for each lesion
            # so if there is one lesion.
            # polygons = list of size 1
            # dict of size 3. 
            # all_point_x is list of variable size(depends on number of points)
            # same for y
            # name is str 'polygon'

            
            
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            
            ###### LETS STORE PATH TO IMAGE INTO JSON FILE. 
            
            ## --- RERUN. added to images under annotations. current_file_path
            #image_path = image_info['File_path']
            #image_path = os.path.join(dataset_dir, a['filename'])
            image_path = os.path.join(DEXTR_DIR,image_info['File_path'])
            
            
            #***************************************************************
            ########################################################### use this to import all png files in this directory and load them as blanks. NaN for segmentation. 
            #############################################################
            #********************************************************************
            
            
            # use files_bg to add in non-segmentated images to the dataset. 
            ##
            #
            #
            # polygons = [] ? or polygons = NaN?
            # Need to figure out what format will work so it will train on these background images and not throw an error. 
            #
            #
            ##
            
            

    
    
    
            ################### image format should be: unit8 rgb 
            #image = skimage.io.imread(image_path)
            ############################### WORKS! LOAD WITH DEFAULT WINDOWING.
            #image = util.load_im_with_default_windowing(win,image_path)
            
            ############################### LOAD WITH default 16bit format?
            image = cv2.imread(image_path, -1)

            
            ###############
            
            height, width = image.shape[:2]

            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFOb = randint(0,len(annotations_seg)) BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
 
            print(subset)
            print(int(image_cat))
            print(train_valid_test)
            #### PROBLEM: Most of the "1" training images are "unknown" and therefore worthless. There is 4831 in 3 and 4793 or so in 2. ZERO LABELS IMAGES IN 1. How lame is that?
            if subset == 'train' and train_valid_test==1: #Last checks that there are no unknowns.
                print("Image Added for training")
                print("Image Added for training")
                print("Image Added for training")
                print("Image Added for training")
                print("Image Added for training")
                

                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Abnormality",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)


            elif subset == 'val' and train_valid_test==2: #Last checks that there are no unknowns.
                print("Image added for validation")

                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Abnormality",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)

            elif subset == 'test' and train_valid_test==3: #Last checks that there are no unknowns.
                print("Image added for validation")

                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Abnormality",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)

                
            else:
                print("No image added...")
                print("Unknown, should say -1 below")
                print(int(image_cat)) # 3 is saved for validation since we are using both training and validation (1&2) for training.
                print("train_valid_test")
                print(train_valid_test)
                
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        #image = skimage.io.imread(self.image_info[image_id]['path'])
        im = cv2.imread(self.image_info[image_id]['path'], -1)
        im1 = im.astype(np.float32, copy=False)-32768
        info = self.image_info[image_id]
        win = info['win']

        im1 -= win[0]
        im1 /= win[1] - win[0]
        im1[im1 > 1] = 1
        im1[im1 < 0] = 0
        im1 *= 255
        #image = im.astype(np.uint8)
        im2 = np.stack([im1,im1,im1],axis=2)
        im3=im2.astype(np.uint8)
        return im3

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]


        
        ######### This isn't working....
        #if image_info["source"] != "Lesion":
        #    return super(self.__class__, self).load_mask(image_id)
        if image_info["source"] != "Abnormality":
            return super(self.__class__, self).load_mask(image_id)
        
        
        
        
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr, cc = skimage.draw.polygon(p['all_points_x'],p['all_points_y'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #class_ids=np.array([self.class_names.index(shapes[0])])
        #print("info['class_ids']=", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        
        
        ########################## OLD CODE #####################################################
        #image_info = self.image_info[image_id]
        #info = self.image_info[image_id]
        #mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                dtype=np.uint8)

        #for i, p in enumerate(info["polygons"]):

            #p['all_points_y'] = [int(i) for i in p['all_points_y']]
            #p['all_points_x'] = [int(i) for i in p['all_points_x']]

            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            #mask[rr, cc, i] = 1
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        ############################ OLD CODE #######################################################
        
        return mask, class_ids#[mask.shape[-1]] #np.ones([mask.shape[-1]], dtype=np.int32)#class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        
        #################### NEED TO SET THIS equal to something else. or jsut return all. 
        
        return info["path"]
        
        #if info["source"] == "balloon":
        #    return info["path"]
        #else:
        #    super(self.__class__, self).image_reference(image_id)

         ################## Not sure about this... 


dataset_train = Deep_Lesion_Dataset()
    
############################## We need to separate the training and validation
### There is a variable in the dict, which is training_val_testing
### It's in image_info file, 
# under annotations, image
#annotations['images'][0]['Train_Val_Test']
    
    ### TRAIN = 1
    ### VALIDATION = 2
    ### TEST = 3
    
    ###### NEED TO EXRACT THREE DIFFERENT DATASETS based on this. 
    ## look at how "train"/"valid" is fed in below
    
    
dataset_train.load_deep_lesion('doesntmatter', "train")
    
dataset_train.prepare()

    # Validation dataset
dataset_val = Deep_Lesion_Dataset()
    
dataset_val.load_deep_lesion('doesntmatter', "val")
    
dataset_val.prepare()

dataset_test = Deep_Lesion_Dataset()
    
dataset_test.load_deep_lesion('doesntmatter', "test")
    
    
    
dataset_test.prepare()

print("Training Images: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

print("Validations Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))

#model = modellib.MaskRCNN(mode="inference", config=config,
#                          model_dir=MODEL_DIR)
DEVICE = "/gpu:0"
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config) 

   
########################################################
#model.load_weights(WEIGHT_DIR, by_name=True, exclude=[
#    "mrcnn_class_logits", "mrcnn_bbox_fc",
#   "mrcnn_bbox", "mrcnn_mask"]) 
########################################################

##################################################################################### results wont predict with this code
weights_path = model.find_last()
########## Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
########################################################################################

print("Validation image")
image_id = random.choice(dataset_val.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
info = dataset_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_val.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

image = dataset_val.load_image(image_id)
mask, class_ids = dataset_val.load_mask(image_id)
# Compute Bounding box
bbox = utils_DL.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_val.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_val.class_names)
#path = dataset_val.load_image_path(image_id)
#win = dataset_val.load_image_win(image_id)

print("TEST IMAGE- never seen")
image_id = random.choice(dataset_test.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
info = dataset_test.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_test.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_test.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

image = dataset_test.load_image(image_id)
mask, class_ids = dataset_test.load_mask(image_id)
# Compute Bounding box
bbox = utils_DL.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset_test.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_test.class_names)
#path = dataset_val.load_image_path(image_id)
#win = dataset_val.load_image_win(image_id)
mask_pred = r['masks']
#mask_pred = mask_pred.astype(np.uint8)
#mask_pred = mask_pred[:,:,0:1]
utils_DL.compute_overlaps_masks(mask_pred,mask)

print("Mean Average Precision")
pred_box = r['rois']
pred_class_id = r['class_ids']
pred_score = r['scores']
pred_mask = r['masks']
mAP = utils_DL.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1)
print(mAP)

