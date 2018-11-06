import random
import cv2
import numpy as np
from PIL import Image
import os


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def online_process(line, input_shape, anchors, num_classes, max_boxes=100):
    line = line.split(' ')
    filename = line[0]
    if filename[-1] == '\n':
        filename = filename[:-1]
    image = Image.open(filename)

    # resize the image to input shape
    boxed_image, shape_image = letterbox_image(image, tuple(reversed(input_shape)))
    
    image_data = np.array(boxed_image, dtype=np.uint8)  # pixel: [0:255] uint8:[-128, 127]
    box_data = []
    boxes = np.zeros((max_boxes, 9), dtype=np.float)

    # correct the BBs to the image resize
    if len(line)==1:  # if there is no object in this image
        box_data.append(boxes)
    for i, box in enumerate(line[1:]):
        if i < max_boxes:
            # for 3D
            boxes[i] = np.array(list(map(float, box.split(','))))
            # boxes[i, 0:5] = boxes[i, 0:5].astype(int)
        else:
            break
        image_size = np.array(image.size)
        input_size = np.array(input_shape[::-1])  # hw -> wh
        
        boxes[i, 0] = (boxes[i, 0]*input_size[0]/image_size[0]).astype(np.int32)
        boxes[i, 2] = (boxes[i, 2]*input_size[0]/image_size[0]).astype(np.int32)
        boxes[i, 1] = (boxes[i, 1]*input_size[1]/image_size[1]).astype(np.int32)
        boxes[i, 3] = (boxes[i, 3]*input_size[1]/image_size[1]).astype(np.int32)

    box_data.append(boxes)
    # correct the BBs to the format of YOLO output
    y_true = preprocess_true_boxes(np.array(box_data), input_size, anchors, num_classes)
    # save the processed label into file
    # np.savez(data_path + filename.split('/')[-1].split('.')[0] + '.npz', image_data=image_data, box_data=boxes, image_shape=image_shape, y_true0=y_true[0], y_true1=y_true[1], y_true2=y_true[2])
    return image_data, y_true

def load_training_data(data_path):
    """
    processes the data into standard shape
    :param annotation_path: path_to_image box1,box2,...,boxN with boxX: x_min,y_min,x_max,y_max,class_index
    :param data_path: saver at "/home/minh/stage/train.npz"
    :param input_shape: (416, 416)
    :param max_boxes: 100: maximum number objects of an image
    :param load_previous: for 2nd, 3th, .. using
    :return: image_data [N, 416, 416, 3] not yet normalized, N: number of image
             box_data: box format: [N, 100, 6], 100: maximum number of an image
                                                6: top_left{x_min,y_min},bottom_right{x_max,y_max},class_index (no space)
                                                /home/minh/keras-yolo3/VOCdevkit/VOC2007/JPEGImages/000012.jpg 156,97,351,270,6
    """
    if os.path.isfile(data_path):
        data = np.load(data_path)
        return data['image_data'], data['box_data'], data['image_shape'], [data['y_true0'], data['y_true1'], data['y_true2']]

# Old letter_box, which resize image with unchanged aspect ratio using padding

# def letterbox_image(image, size):
#     """resize image with unchanged aspect ratio using padding
#     :param: size: input_shape
#     :return:boxed_image:
#             image_shape: original shape (h, w)
#     """
#     image_w, image_h = image.size
#     image_shape = np.array([image_h, image_w])
#     w, h = size
#     new_w = int(image_w * min(w/image_w, h/image_h))
#     new_h = int(image_h * min(w/image_w, h/image_h))
#     resized_image = image.resize((new_w, new_h), Image.BICUBIC)

#     boxed_image = Image.new('RGB', size, (128, 128, 128))
#     boxed_image.paste(resized_image, ((w-new_w)//2, (h-new_h)//2))
#     return boxed_image, image_shape

def letterbox_image(image, size):
    """resize image with changing aspect ratio without padding
    :param: size: input_shape
    :return:boxed_image:
            image_shape: original shape (h, w)
    """
    image_w, image_h = image.size
    image_shape = np.array([image_h, image_w])
    w, h = size
    resized_image = image.resize((w, h), Image.BICUBIC)
    
    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image)

    return boxed_image, image_shape

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    # input_shape -> wh (xy)
    """
    Preprocess true boxes to training input format
    :param true_boxes: array, shape=(N, 100, 10) N:batch size, 100: max number of bounding boxes,
                                                5: x_min,y_min,x_max,y_max,score,cos(alpha),sin(alpha),height,width,length,class_id
                    Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    :param input_shape: array-like, hw, multiples of 32, shape = (2,)
    :param anchors: array, shape=(9, 2), wh
    :param num_classes: integer
    :return: y_true: list(3 array), shape like yolo_outputs, xywh are reletive value 3 array [N, h//32, w//32, 3, 10+num_class]
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] # different anchors are assigned to different scales
    true_boxes = np.array(true_boxes, dtype=np.float32)

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # [m, T, 2]  (x, y) center point of BB
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w = x_max - x_min  [m, T, 2]
                                                            # h = y_max - y_min
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape

    N = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]
    y_true = [np.zeros((N, grid_shapes[l][1], grid_shapes[l][0], len(anchor_mask[l]), 10 + int(num_classes)),
                       dtype=np.float32) for l in range(3)]  # (m, h//32, w//32, 3, 10+num_class)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # [1, 3, 2]
    anchor_maxes = anchors / 2.  # w/2, h/2  [1, 3, 2]
    anchor_mins = -anchor_maxes   # -w/2, -h/2  [1, 3, 2]
    valid_mask = boxes_wh[..., 0] > 0  # w>0 True, w=0 False

    for b in (range(N)):  # for all of N image
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # image 0: wh [[[163., 144.]]]
        # Expand dim to apply broadcasting.
        if len(wh)==0:
            continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        # print(true_boxes[b][:10])
        for t, n in enumerate(best_anchor):
            for l in range(3):  # 1 in 3 scale
                if n in anchor_mask[l]:  # choose the corresponding mask: best_anchor in [6, 7, 8]or[3, 4, 5]or[0, 1, 2]
                    # i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(np.int32)
                    # j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(np.int32)
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][1]).astype(np.int32)
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][0]).astype(np.int32)

                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype(np.int32)  # idx classes in voc classes
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]  # l: scale; b; idx image; grid(i:y , j:x); k: best anchor; 0:4: (x,y,w,h)/input_shape
                    y_true[l][b, j, i, k, 4] = 1  # score = 1
                    # for 3D
                    y_true[l][b, j, i, k, 5] = np.cos(true_boxes[b, t, 5])
                    y_true[l][b, j, i, k, 6] = np.sin(true_boxes[b, t, 5])
                    y_true[l][b, j, i, k, 7:10] = true_boxes[b, t, 6:9]
                    y_true[l][b, j, i, k, 10 + c] = 1  # classes = 1, the others =0
                    # print(y_true[l][b, j, i, k])
                    # print("  y_true[l][b, j, i, k, :] with l:%s, b:%s, j:%s, i:%s, k:%s, c:%s" %(l,b,j,i,k,c))
                    # print("  with l:", l, "b:", b, "j:", j, "i:", i, "k:", k, "c:", c)
                    # print(y_true[l][b, j, i, k, :])
                    break
    return y_true