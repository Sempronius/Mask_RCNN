



############## This information is hard coded below in training model, First 5 epoch heads , then 10 epochs of all training.
#epoch = 5
#layers = 'heads' #'all'  or 'heads' # PREVIOUSLY JUST DID HEADS
######################



patience1=3

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    
     


    # Train a new model starting from ImageNet weights
    python3 deep_lesion_train_one_lesion_test.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Train a new model starting from pre-trained COCO weights
    python3 deep_lesion_train_one_lesion_test.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 deep_lesion_train_one_lesion_test.py train --dataset=/path/to/balloon/dataset --weights=last
    
    # Apply color splash to an image
    python3 ddeep_lesion_train_one_lesion_test.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 deep_lesion_train_one_lesion_test.py splash --weights=last --video=<URL or path to file>
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
import os, random




# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

### Path to DEXTR
BASE_DIR = os.path.abspath("../../../")

DEXTR_DIR = os.path.join(BASE_DIR,"DEXTR-PyTorch_p")
with open (os.path.join(DEXTR_DIR,'outfile_bg'), 'rb') as fp:
    files_bg = pickle.load(fp)

############################################################
#  Configurations
############################################################


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
    VALIDATION_STEPS = 5/BATCH_SIZE 
    STEPS_PER_EPOCH = 22/BATCH_SIZE 
    #VALIDATION_STEPS = 9586/BATCH_SIZE 
    #STEPS_PER_EPOCH = 44990/BATCH_SIZE 
    # Number of classes (including background)
    LOSS_WEIGHTS = {'mrcnn_mask_loss': 1.0, 'rpn_class_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 3.0, 'rpn_bbox_loss': 1.0}
    
    
    ######################################### UNKNOWNS REMOVED, so only 8 classes. 
    
    
    NUM_CLASSES = 1 + 1  # Background + ONE CLASS


    #IMAGE_RESIZE_MODE = "square"
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM = 512
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #TRAIN_ROIS_PER_IMAGE = 256
    #MAX_GT_INSTANCES = 3
    #DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    #DETECTION_NMS_THRESHOLD = 0.3
    
    #LEARNING_RATE = 0.00001
    #LEARNING_MOMENTUM = 0.9
    

    # Weight decay regularization
    #WEIGHT_DECAY = 0.0001

    #USE_MINI_MASK = False
    ########################## make this false?
    #USE_MINI_MASK = True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    ##########################


############################################################
#  Dataset
############################################################

class Deep_Lesion_Dataset(utils_DL.Dataset):

    def load_deep_lesion(self, dataset_dir, subset): #I don't think we need dataset directory.
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Abnormality", 1, "Lesion") # "Bone"
        annotations = json.load(open(os.path.join(DEXTR_DIR, "data_including_unknown.json")))
        annotations_seg = annotations['annotations']
        b=0
        for a in annotations_seg:
            image_info = annotations['images'][b]
            win = annotations_seg[b]['Windowing']
            image_id = annotations['images'][b]['id']
            polygons=[]
            objects=[]
            #for r in a['regions'].values():
            for r in a['regions']:
                polygons.append(r['shape_attributes'])
            # print("polygons=", polygons)
                #objects.append(r['region_attributes'])
                objects.append({"Abnormality":1}) #All images have labels in format {Lesion:int}, but we want to change them all to one type, "Lesion".
            
            class_ids = [int(n['Abnormality']) for n in objects]
            train_valid_test = annotations['images'][b]['Train_Val_Test']
            b=b+1
            image_path = os.path.join(DEXTR_DIR,image_info['File_path'])
            image = cv2.imread(image_path, -1) #This isn't for reading, it's just to get height and width, see next line. 
            height, width = image.shape[:2]
######################################################### ADD RANDOM BACKGROUND IMAGE.

            dir_name = os.path.dirname(image_path)
            
            random_choice = random.choice(os.listdir(dir_name)) #change dir name to whatever
            while random_choice == os.path.basename(image_path):
                random_choice = random.choice(os.listdir(dir_name))
                #if random choice is equal to the key image, it will pick again.
            
            #Random background image path
            bg_path = os.path.join(dir_name,random_choice)
             
            #filename
            os.path.basename(os.path.splitext(bg_path)[0])
            #directory name
            os.path.basename(os.path.dirname(bg_path)) 
            
            line = os.path.basename(os.path.dirname(bg_path)) + os.path.basename(os.path.splitext(bg_path)[0])
            
            bg_image_id =  line.translate({ord(i):None for i in '_'}) #get rid of the "_"
                         
            if subset == 'train' and train_valid_test==1: #Last checks that there are no unknowns.
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
                ##### Add that random background image
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "BG",
                        image_id=bg_image_id,  # id is set above. 
                        path=bg_path,
                        width=width, height=height,
                        polygons={'all_points_x':[],'all_points_y':[],'name':'polygon'}, #Not sure if this is the right way to do it. 
                        win=win,#using same windowing as keyimage. 
                        class_ids=[0])#it's just a background image, adding it as a zero. 
                
            elif subset == 'val' and train_valid_test==2: #Last checks that there are no unknowns.
                print("Image Added for validation")
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "Abnormality",
                        image_id=image_id,  # id is set above. 
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        win=win,
                        class_ids=class_ids)
                
                self.add_image(
                        ############ Replace balloon with CLASSES ABOVE, take from category. 
                        "BG",
                        image_id=bg_image_id,  # id is set above. 
                        path=bg_path,
                        width=width, height=height,
                        polygons={'all_points_x':[],'all_points_y':[],'name':'polygon'}, #Not sure if this is the right way to do it. 
                        win=win,#using same windowing as keyimage. 
                        class_ids=[0])#it's just a background image, adding it as a zero. 
                
            else:
                print("No image added...")
                
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
        info = self.image_info[image_id]
        
        #if info["source"] != "Abnormality":
        #    return super(self.__class__, self).load_mask(image_id)
        
        
        
        
        class_ids = info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], int(len(info["polygons"])/3)], #This is supposed to be a list of instances. So, when you look at one instance it is {'all_points_x': [], 'all_points_y': [], 'name': 'polygon'}, but len results in '3'. So we divide by three. That way each 'instance' is equal to one. not 3. 
                        dtype=np.uint8)
        ##########################################
        ########################################## NEED TO SKIP CODE BELOW IF THERE IS NO MASK IMAGE (ie, all_points_x = [])
        
        if info["source"] == "Abnormality":
            for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                rr, cc = skimage.draw.polygon(p['all_points_x'],p['all_points_y'])
                mask[rr, cc, i] = 1
        ########################################## OTHERWISE IT SHOULD JUST BE A BLANK MASK...

        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids#[mask.shape[-1]] #np.ones([mask.shape[-1]], dtype=np.int32)#class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Abnormality":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#####################################################################
#augmentation = iaa.SomeOf((0, 1), [
    #iaa.Fliplr(0.5),
 #   iaa.Affine(
 #       scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#        rotate=(-5, 5),
#        shear=(-1, 1)
#    ),
#    iaa.Multiply((0.9, 1.1))
#])
    ########################################################################################

#from keras.callbacks import ReduceLROnPlateau
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=patience1, verbose=1, mode='min', cooldown=0, min_lr=0)
#callbacks=[reduce_lr] 

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
    augmentation = iaa.SomeOf((0, 1), [
    #iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-5, 5),
        shear=(-1, 1)
    ),
    iaa.Multiply((0.9, 1.1))
    ])
    ########################################################################################

    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=patience1, verbose=1, mode='min', cooldown=0, min_lr=0)
    callbacks=[reduce_lr] 
    
    print("Training Images: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))
    print("Validations Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                #epochs=30,
                #layers='heads')
                epochs=20,
                layers='all',
                custom_callbacks=callbacks,#############
                augmentation=augmentation)######################
    print("finised training network")


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
            utils_DL.download_trained_weights(weights_path)
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
