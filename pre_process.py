from config import Input_shape, channels, path
from network_function import YOLOv3

from loss_function import compute_loss
from utils.yolo_utils import get_training_data, read_anchors, read_classes, pre_process_data

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

np.random.seed(101)

# Get Data #############################################################################################################
PATH = path
classes_paths = PATH + '/model/bdd_classes.txt'
classes_data = read_classes(classes_paths)
anchors_paths = PATH + '/model/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)
print ("Number of classes: ", len(classes_data))

annotation_path_train = PATH + '/model/bdd_train.txt'
annotation_path_valid = PATH + '/model/bdd_train.txt'
annotation_path_test = PATH + '/model/bdd_test.txt'

data_path_train = PATH + '/data/bdd_train/'
data_path_valid = PATH + '/data/bdd_valid/'
data_path_test = PATH + '/data/bdd_test/'


input_shape = (Input_shape, Input_shape)  # multiple of 32
pre_process_data(annotation_path_train, data_path_train, input_shape, anchors, "train", num_classes=len(classes_data), max_boxes=40, load_previous=True)
pre_process_data(annotation_path_valid, data_path_valid, input_shape, anchors, "val", num_classes=len(classes_data), max_boxes=40, load_previous=True)