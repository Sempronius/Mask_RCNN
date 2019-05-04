#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:07:09 2019

@author: sempronius
"""

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
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.balloon import balloon

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
    DETECTION_MIN_CONFIDENCE = 0.9
    
config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "samples/balloon/balloon")


class Deep_Lesion_Dataset(utils.Dataset):

    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("1", 1, "1") # or bone bone? or bone lesion?
        self.add_class("2", 2, "2")
        self.add_class("3", 3, "3")
        self.add_class("4", 4, "4")
        self.add_class("5", 5, "5")
        self.add_class("6", 6, "6")
        self.add_class("7", 7, "7") #Soft tissue: miscellaneous lesions in the body wall, muscle, skin, fat, limbs, head, and neck
        self.add_class("8", 8, "8")
        self.add_class("-1", 9, "-1") #Can i have a negative number here? This is straight from Deep Lesion format. 
     
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
        annotations = json.load(open(os.path.join(DEXTR_DIR, "data.json")))
        
        #annotations = list(annotations.values())  # don't need the dict keys
        annotations_seg = annotations['annotations']
        
        #annotations_seg = segmentation[2]

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        #annotations_seg = [a for a in annotations_seg if a['segmentation']]

        # Add images
        
        b=0
        for a in annotations_seg:
            image_info = annotations['images'][b]
            win = annotations_seg[b]['Windowing']
            image_id = annotations['images'][b]['id']
            image_cat = annotations['categories'][b]['category_id']
            train_valid_test = annotations['images'][b]['Train_Val_Test']
            #### Must use index 'b' before here, because after this point it
            # will point to next image/index/
            b=b+1
            
            
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = a['regions']
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
            
   
            ################### image format should be: unit8 rgb 
            #image = skimage.io.imread(image_path)
            image = util.load_im_with_default_windowing(win,image_path)
            

            
            ###############
            
            height, width = image.shape[:2]

            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFOb = randint(0,len(annotations_seg)) BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
            #### SEE IMAGE_INFO, INFO BELOW: I think it gets this from here
 
            if subset == 'train' and train_valid_test==1:
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        image_cat,
                        ############### Replace balloon with CLASSES ABOVE,take from category.
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons)

            if subset == 'val' and train_valid_test==2:
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        image_cat,
                        ############### Replace balloon with CLASSES ABOVE,take from category.
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons)             
                
                

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        
        
        
        ###########
        ##########
        ######## We want to evaluate every image. We could re-write this to include "1-10,-1" for categoies. 
        #if image_info["source"] != "balloon":
        #    return super(self.__class__, self).load_mask(image_id)
##################
        #############

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1

            ####
            ### p needs to be a dictionary that contains three things: 
            #polygon/name(str), all_points_x(list), all_points_y(list)
            ### my all_points_x seems to be a list within a list? not sure. 
            ###
            
            #### MY all_points are lists of floats. We need them to be lists of int
            p['all_points_y'] = [int(i) for i in p['all_points_y']]
            p['all_points_x'] = [int(i) for i in p['all_points_x']]
            #############
            ################
            #####################
            
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

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

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190429T2251/mask_rcnn_deep_lesion_0100.h5'

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

dataset = Deep_Lesion_Dataset()
dataset.load_deep_lesion(BALLOON_DIR, "val")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
    