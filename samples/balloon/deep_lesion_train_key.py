epoch = 100
layers = 'all' #'all'  or 'heads' # PREVIOUSLY JUST DID HEADS
patience1=2

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    
     
    # Train a new model starting from pre-trained COCO weights
    python3 deep_lesion_train_key.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 deep_lesion_train_key.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 deep_lesion_train_key.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 deep_lesion_train_key.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 deep_lesion_train_key.py splash --weights=last --video=<URL or path to file>
"""
from random import randint
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
from imgaug import augmenters as iaa
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils_DL
import utilities as util
import cv2
import pickle



# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

### Path to DEXTR
BASE_DIR = os.path.abspath("../../../")

DEXTR_DIR = os.path.join(BASE_DIR,"DEXTR-PyTorch_p")


############################################################
#  Configurations
############################################################


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


############################################################
#  Dataset
############################################################

class Deep_Lesion_Dataset(utils.Dataset):

    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Lesion", 1, "1") # "Bone"
        self.add_class("Lesion", 2, "2") # "Abdomen_notLiver_notKidney"
        self.add_class("Lesion", 3, "3") # "Mediastinum"
        self.add_class("Lesion", 4, "4") # "Liver"
        self.add_class("Lesion", 5, "5") #"Lung"
        self.add_class("Lesion", 6, "6") #"Kidney"
        self.add_class("Lesion", 7, "7") #Soft tissue: miscellaneous lesions in the body wall, muscle, skin, fat, limbs, head, and neck
        self.add_class("Lesion", 8, "8") #"Pelvis"
        
        
        
        
        
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
        
        #annotations = list(annotations.values())  # don't need the dict keys
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
            image_cat_name = "blank"
            
            if image_cat == 1:
                image_cat_name = "Bone"
            elif image_cat == 2:
                image_cat_name = "Abdomen_notLiver_notKidney"
            elif image_cat == 3:
                image_cat_name = "Mediastinum"
            elif image_cat == 4:
                image_cat_name = "Liver"
            elif image_cat == 5:
                image_cat_name = "Kidney"
            elif image_cat == 6:
                image_cat_name = "Lung"                
            elif image_cat == 7:
                image_cat_name = "Soft tissue"
            elif image_cat == 8:
                image_cat_name = "Pelvis"
                
                
                
            
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
            objects = [s['region_attributes'] for s in a['regions'].values()]
            num_ids = [int(n['class']) for n in objects]
            
            
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
 
            if subset == 'train' and train_valid_test==1 and int(image_cat) >= 0: #Last checks that there are no unknowns.
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        #image_cat,
                        image_cat_name,
                        ############### Replace balloon with CLASSES ABOVE,take from category.
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,num_ids=num_ids)

            if subset == 'val' and train_valid_test==2 and int(image_cat) >= 0: #Last checks that there are no unknowns.
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        #image_cat,
                        image_cat_name,
                        ############### Replace balloon with CLASSES ABOVE,take from category.
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,num_ids=num_ids)
               
             

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
        if image_info["source"] != "Lesion":
            return super(self.__class__, self).load_mask(image_id)

        
         
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        annotations = info["polygons"]
        count = len(annotations[0]['all_points_x'])
        if count == 0:
            mask = np.zeros((info["height"], info["width"], 1), dtype=np.uint8)
            #class_ids = np.zeros((1,), dtype=np.int32)            
        else:
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
        
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
        return mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        
        #################### NEED TO SET THIS equal to something else. or jsut return all. 
        
        return info["path"]
        
        if info["source"] == "Lesion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

         ################## Not sure about this... 

#####################################################################
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])
    ########################################################################################

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=patience1, verbose=1, mode='min', cooldown=0, min_lr=0)
callbacks=[reduce_lr] 

def train(model):
    """Train the model."""
    # Training dataset.
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
    
    
    dataset_train.load_deep_lesion(args.dataset, "train")
    
    
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Deep_Lesion_Dataset()
    
    dataset_val.load_deep_lesion(args.dataset, "val")
    
    
    
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                #epochs=30,
                #layers='heads')
                epochs=epoch,
                layers=layers,
                custom_callbacks=callbacks,#############
                augmentation=augmentation)######################

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
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


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

  ############################
  ########   ******************!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ########### NEED TO MAKE SURE IT IS A TEST IMAGE (see above, 3)
  ##########    ******************!!!!!!!!!!!!!!!!!!!!!!!
    #################################
    ######## LOAD RANDOM IMAGE
    annotations = json.load(open(os.path.join(DEXTR_DIR, "data.json")))
    annotations_seg = annotations['annotations']
    b = randint(0,len(annotations_seg))
    image_info = annotations['images'][b]

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
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
        
        # Read image
        #image = skimage.io.imread(args.image)
        
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
