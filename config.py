# CONFIGURE
# change the path to your project
path = '/home/liu/Desktop/YOLOv3D-tensorflow'
data_path = '/media/liu/Liu Drive/BDD'
# image pre-processing
Input_width = 1216  # width=height # 608 or 416 or 320
Input_height = 320
channels = 3  # RBG
angle = 0
saturation = 1.5
exposure = 1.5
hue = 0.1
jitter = 0.3
random = 1

# training
# score = 0.3
# iou = 0.7

epochs = 50
batch_size = 24
threshold = 0.6
ignore_thresh = 0.4
truth_thresh = 1
momentum = 0.9
decay = 0.0005
learning_rate = 0.001
burn_in = 1000
max_batches = 500200

# policy=steps
# learning_rate_steps = [40000, 45000]  # steps=400000,450000
# learning_rate_scales = [0.1, 0.1]  # scales=.1,.1
# anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
# num = 9 
# NumClasses = 80


