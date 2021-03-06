#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:48:41 2019

@author: sempronius
"""
import os
import sys
import json
import random
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.coco import maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils_DL

from mrcnn import visualize

from mrcnn.model import log
import cv2

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
    NAME = "Lesion"
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
    VALIDATION_STEPS = 100/BATCH_SIZE 
    STEPS_PER_EPOCH = 1000/BATCH_SIZE 
    # Number of classes (including background)
    
    
    ##############################################
    NUM_CLASSES = 1 + 8  # Background + 8 classes (see below)
    ######## NEED TO CHANGE IF YOU ADD UNKNOWNS BACK IN
    
    # Number of training steps per epoch
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 128
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.93
    DETECTION_NMS_THRESHOLD = 0.3
    
    LEARNING_RATE = 0.00001
    LEARNING_MOMENTUM = 0.9
    

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    USE_MINI_MASK = False
    ########################## make this false?
    #USE_MINI_MASK = True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    ##########################

config = BalloonConfig()


class Deep_Lesion_Dataset(utils_DL.Dataset):

    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Lesion", 1, "Bone") # "Bone"
        self.add_class("Lesion", 2, "Abdomen_notLiver_notKidney") # "Abdomen_notLiver_notKidney"
        self.add_class("Lesion", 3, "Mediastinum") # "Mediastinum"
        self.add_class("Lesion", 4, "Liver") # "Liver"
        self.add_class("Lesion", 5, "Lung") #"Lung"
        self.add_class("Lesion", 6, "Kidney") #"Kidney"
        self.add_class("Lesion", 7, "Soft_tissue") #Soft tissue: miscellaneous lesions in the body wall, muscle, skin, fat, limbs, head, and neck
        self.add_class("Lesion", 8, "Pelvis") #"Pelvis"
        
        
        
        
        
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
        annotations = json.load(open(os.path.join(DEXTR_DIR, "data.json")))
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
                objects.append(r['region_attributes'])
            
            class_ids = [int(n['Lesion']) for n in objects]
                
        
            
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
            #### PROBLEM: Most of the "1" training images are "unknown" and therefore worthless. So we are using 1 and 2 for training. Save 3 for validation.
            if subset == 'train' and (train_valid_test==1 or train_valid_test==2) and int(image_cat) >= 0: #Last checks that there are no unknowns.
                print("Image Added for training")
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Lesion",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)
                

            elif subset == 'val' and train_valid_test==2 and int(image_cat) >= 0: #Last checks that there are no unknowns.
                print("Image added for validation")
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Lesion",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)
                
            else:
                print("No image added...")
                print("test image, should say 3 below")
                print(train_valid_test) # 3 is saved for validation since we are using both training and validation (1&2) for training.



        
                
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
        print("im1 dtype and shape")
        print(im1.dtype)
        print(im1.shape)
        #image = im.astype(np.uint8)
        im2 = np.stack([im1,im1,im1],axis=2)
        im3=im2.astype(np.uint8)
        print("inspect_deep_lesion")
        print("image.dtype")
        print(im3.dtype)
        print("image.dtype")
        print('load_image plt.imshow')
        plt.imshow(im3)
        return im3
    
    def load_image_path(self, image_id):
        path = self.image_info[image_id]['path']
        return path
    def load_image_win(self, image_id):
        info = self.image_info[image_id]
        win = info['win']
        return win
    
    
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
        if image_info["source"] != "Lesion":
            return super(self.__class__, self).load_mask(image_id)
        
        
        
        
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            print(info["polygons"])
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
        
        #return info["path"]
        
        if info["source"] == "Lesion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

         ################## Not sure about this... 
dataset = Deep_Lesion_Dataset()
dataset.load_deep_lesion("doesnt_matter_what_i_type_here", "train")
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils_DL.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
path = dataset.load_image_path(image_id)
win = dataset.load_image_win(image_id)
## Need to pull out path from somewhere above. 

ima = cv2.imread(path, -1)
ima = ima.astype(np.float32, copy=False)-32768
ima1 = ima.astype(float)
ima1 -= win[0]
ima1 /= win[1] - win[0]
ima1[ima1 > 1] = 1
ima1[ima1 < 0] = 0
ima1 *= 255
ima2 = np.stack([ima1,ima1,ima1],axis=2)
ima3=ima2.astype(np.uint8)
plt.imshow(ima3)