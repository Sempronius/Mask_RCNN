#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:32:13 2019

@author: sempronius
Train with last  weights
LOOK UP:
    http://cocodataset.org/#detection-eval
"""

json_file = "data_coco_format_one_lesion.json"

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
from skimage import measure
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn import utils_DL
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import copy

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


MODEL_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/'
#NEED TO SET WEIGHT DIRECTORY

#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/balloon20190428T1350/mask_rcnn_balloon_0030.h5'


#################
#WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190617T0030_trained100/mask_rcnn_deep_lesion_0100.h5'
WEIGHT_DIR = '/home/sempronius/deep_learning/Mask_RCNN/logs/deep_lesion20190606T0805_good/mask_rcnn_deep_lesion_0020.h5'
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
from mrcnn import model as modellib, utils_DL_map
import utilities as util

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


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

TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class Deep_Lesion_Dataset(utils_DL_map.Dataset):
    
    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Abnormality", 1, "Lesion") # "Bone"

        #annotations = json.load(open(os.path.join(DEXTR_DIR, "data_including_unknown.json")))
        #annotations = json.load(open(os.path.join(DEXTR_DIR, "data_coco_format.json")))
        #annotations_seg = annotations['annotations']
        #coco = COCO(os.path.join(DEXTR_DIR, "data_coco_format.json"))

        coco = COCO(os.path.join(DEXTR_DIR, json_file))
        class_ids = sorted(coco.getCatIds())
        # All images
        image_ids = list(coco.imgs.keys())
                # Add classes
        for i in class_ids:
            # USE THIS CODE When you want to train individual (-1,1,2,3,4,5,6,7,8). unknown, abdomen, lung, etc
            #self.add_class("coco", i, coco.loadCats(i)[0]["name"])
            # Here we are just manually making one category. Everything is a "Lesion". 
            self.add_class("Abnormality", 1, "Lesion")

                
            
        for i in image_ids:
            ################################## This removes the individual categories and just makes everthing equal to generic "lesion. Remove these lines if you want it to train on individual categories. 
            if coco.anns[i]['Coarse_lesion_type'] == -1 or coco.anns[i]['Coarse_lesion_type'] == 1 or coco.anns[i]['Coarse_lesion_type'] == 2 or coco.anns[i]['Coarse_lesion_type'] == 3 or coco.anns[i]['Coarse_lesion_type'] == 4 or coco.anns[i]['Coarse_lesion_type'] == 5 or coco.anns[i]['Coarse_lesion_type'] == 6 or coco.anns[i]['Coarse_lesion_type'] == 7 or coco.anns[i]['Coarse_lesion_type'] == 8:
                class_id=1
            if subset == "val" and coco.imgs[i]['Train_Val_Test'] == 2:

                            self.add_image(
                                    "Abnormality", image_id=i,
                                    path=coco.imgs[i]['File_path'],
                                    width=coco.imgs[i]["width"],
                                    height=coco.imgs[i]["height"],
                                    annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)),
                                    #annotations=coco.anns[i]['segmentation'],
                                    win=coco.anns[i]['Windowing'],
                                    class_id=class_id
                                    )
            if subset == "test" and coco.imgs[i]['Train_Val_Test'] == 3:

                            self.add_image(
                                    "Abnormality", image_id=i,
                                    path=coco.imgs[i]['File_path'],
                                    width=coco.imgs[i]["width"],
                                    height=coco.imgs[i]["height"],
                                    annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)),
                                    #annotations=coco.anns[i]['segmentation'],
                                    win=coco.anns[i]['Windowing'],
                                    class_id=class_id
                                    )
            if subset == "train" and coco.imgs[i]['Train_Val_Test'] == 1:
                            self.add_image(
                                    "Abnormality", image_id=i,
                                    path=coco.imgs[i]['File_path'],
                                    width=coco.imgs[i]["width"],
                                    height=coco.imgs[i]["height"],
                                    annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)),
                                    #annotations=coco.anns[i]['segmentation'],
                                    win=coco.anns[i]['Windowing'],
                                    class_id=class_id
                                    )           
                
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

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


    #def annToMask(self, ann,image_id):
    #    """
    #    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    #    :return: binary mask (numpy 2D array)
    #    """
    #    rle = self.annToRLE(ann,image_id)
    #    m = maskUtils.decode(rle)
    #    return m
    
    #def annToRLE(self, ann,image_id):
    #    """
    #    Convert annotation which can be polygons, uncompressed RLE to RLE.
    #    :return: binary mask (numpy 2D array)
    #    """
    #    image_info = self.image_info[image_id]
    #    #t = self.imgs[ann['image_id']]
    #    t = image_info
    #    h, w = t['height'], t['width']
    #    #segm = ann['segmentation']
    #    segm = ann
    #    segm = [round(x) for x in segm]
    #    if type(segm) == list:
    #        # polygon -- a single object might consist of multiple parts
    #        # we merge all parts into one mask rle code
    #        rles = maskUtils.frPyObjects([segm], h, w)
    #        rle = maskUtils.merge(rles)
    #    elif type(segm['counts']) == list:
    #        # uncompressed RLE
    #        rle = maskUtils.frPyObjects(segm, h, w)
    #    else:
    #        # rle
    #        rle = ann
    #        #rle = ann['segmentation']
    #    return rle
 
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Abnormality":
            return super(dataset_test, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            
#            class_id = self.map_source_class_id(
#                "coco.{}".format(annotation['category_id']))
            #class_id = self.map_source_class_id(annotation['category_id'])
            class_id = annotation['category_id']

            

            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(dataset_test, self).load_mask(image_id)

    

    
    
    
#    def load_mask(self, image_id):
#        """Load instance masks for the given image.
#        Different datasets use different ways to store masks. This
#        function converts the different mask format to one format
#        in the form of a bitmap [height, width, instances].
#        Returns:
#        masks: A bool array of shape [height, width, instance count] with
#            one mask per instance.
#        class_ids: a 1D array of class IDs of the instance masks.
#        """
#        image_info = self.image_info[image_id]
#        if image_info["source"] != "Abnormality":
#            return super(Deep_Lesion_Dataset, self).load_mask(image_id)
#        instance_masks = []
#        class_ids = []
#        annotations = self.image_info[image_id]["annotations"]
#
#        for annotation in annotations:
#            class_id = image_info["class_id"]
#            if class_id:
#                m = self.annToMask(annotation,image_id)
#                if m.max() < 1:
#                    continue
#                instance_masks.append(m)
#                print('instance_masks')
#                print(instance_masks)
#                class_ids.append(class_id)
#        if class_ids:
#            mask = np.stack(instance_masks, axis=2).astype(np.bool)
#            class_ids = np.array(class_ids, dtype=np.int32)
#            return mask, class_ids
#        else:
#            return super(Deep_Lesion_Dataset, self).load_mask(image_id)
        
        
        
        
        
        
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]    
        return info["path"]
    
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
 
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str: #or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()

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

print("Test Images: {}\nClasses: {}".format(len(dataset_test.image_ids), dataset_test.class_names))

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

##################################################################################### LOAD LAST WEIGHTs
weights_path = model.find_last()
########## Load weights
#print("Loading weights ", WEIGHT_DIR)
#model.load_weights(WEIGHT_DIR, by_name=True)
########################################################################################
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]
            contours = measure.find_contours(mask, 0.5)
            #if contours == []:
            #    print('mask')
            #    print(mask)

            if len(contours) >= 2:
                def PolyArea(x,y): #calculate area given series of x.y coordinates.
                    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
                def column(matrix, i): #pull out x,y values from contours numpy format
                    return [row[i] for row in matrix]
            
                # contours should be a list of 2 or more numpy arrays
                # loop over it. 
                out_var = 0

                for b in range(0,len(contours)):
                    x_y=contours[b] # was x_y=contours[0], but that don't make no sense. 
                    all_points_x = column(x_y, 0) #first column should be x
                    all_points_x_int = []
                    for a in all_points_x:
                        int(a)
                        all_points_x_int.append(int(a))
                        all_points_y = column(x_y, 1) #second column should be y
                        all_points_y_int = []
                    for a in all_points_y:
                        int(a)
                        all_points_y_int.append(int(a))
                    if PolyArea(all_points_x,all_points_y) >= out_var:
                        out_var = PolyArea(all_points_x,all_points_y)
                        number = b
                    
            
                contours = [contours[number]] #assist conoutrs to whatever contour had the highest area.
            
            # save segmentation as polygon
            #segmentation = [] ## Added this because there was a rare instance when contour was empty and the following lines that included segmentation threw an error. 
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                 #save segmentation data as polygon 
            
            #Round to eliminate floats

            segmentation = [round(x) for x in segmentation]
            
            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "Abnormality"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": [segmentation] #added [] to help with maskUtils.frPyObjects below. It was throwing an error suggesting this segmentation should be within a list. However, this is probably in case there are multiple predictions. I am not 100 % we are handling the multiple prediction/multiple ground truth scenario correctly. 
            }
            
            results.append(result)


    return results

def loadNumpyAnnotations(data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    # coco = json file loaded as is. It should be in coco format. 
    """
    # Pick COCO images from the dataset
    #image_ids = image_ids or dataset.image_ids
    #image_ids = dataset_train.image_ids
    image_ids = dataset.image_ids

    
    
    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]


    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)


        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)


    # Load results. This modifies results with additional attributes.
    #print(results)
    #coco_results = loadRes(results)

    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    
coco = COCO(os.path.join(DEXTR_DIR, json_file))

## HOW MANY FILES DO YOU WANT TO ANALZYE?
#limit=None


            

print("Running bbox COCO evaluation on {} images.".format(len(dataset_test.image_ids)))
#evaluate_coco(model, dataset_val, coco, "bbox", limit=int(limit))
evaluate_coco(model, dataset_test, coco, "bbox", limit=None)

print("Running segmentation COCO evaluation on {} images.".format(len(dataset_test.image_ids)))
#evaluate_coco(model, dataset_val, coco, "segm", limit=int(limit))
evaluate_coco(model, dataset_test, coco, "segm", limit=None)






