import keras.backend as K
import tensorflow as tf
import math
from config import ignore_thresh

from detect_function import yolo_head


def compute_loss(yolo_outputs, y_true, anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=False):
    """
        Return yolo_loss tensor
    :param YOLO_outputs: list of 3 sortie of yolo_neural_network
    :param Y_true: list(3 array) [(N,h//32,w//32,3,10+numclass), (N,h//16,w//16,3,10+numclass), (N,h//8,w//8,3,10+numclass)]
    :param anchors: array, shape=(T, 2), wh
    :param num_classes: 80
    :param ignore_thresh:float, the iou threshold whether to ignore object confidence loss
    :return: loss
    """
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(3)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(3):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 10:]
        
        # get YOLO output
        grid, raw_pred, pred_xy, pred_wh, pred_cs, pred_hwl = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log((y_true[l][..., 2:4] + 1e-9) / anchors[anchor_mask[l]] * input_shape[::-1])
        # for 3D
        raw_true_cs = y_true[l][..., 5:7]
        raw_true_hwl = y_true[l][..., 7:10]

        # raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        extended_mask = tf.tile(object_mask, [1,1,1,1,2])
        raw_true_wh = tf.where(extended_mask > 0, raw_true_wh, K.ones_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_IoU(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # set some weights for different losses
        lambda_xy = 0.5
        lambda_wh = 10.0
        lambda_obj = 0.5
        lambda_alpha = 6

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # In YOLOv1, sum of square loss is used, you can easily replace it and remember to change the weight
        xy_loss = lambda_xy * object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        # In YOLOv1, sum of square of sqrt loss is used, but sqrt has nan problem in backprop, need some trick if you really want to use it
        wh_loss = lambda_wh * object_mask * box_loss_scale * K.square(raw_true_wh - raw_pred[..., 2:4])
        # cs means cos,sin
        # loss for alpha angle, simply use sum of square loss here, you can replace by the arccos loss
        pred_cs = tf.nn.l2_normalize(pred_cs, axis=-1)
        alpha_loss = lambda_alpha * object_mask * K.square(raw_true_cs - pred_cs)
        # raw_true_cs = tf.nn.l2_normalize(raw_true_cs, axis=-1)
        # cos_val = tf.multiply(pred_cs, raw_true_cs)
        # cos_val = tf.clip_by_value(cos_val, -1, 1)
        # alpha_loss = lambda_alpha * object_mask * (3.1415926/2 - (cos_val + K.pow(cos_val,3)/6 + K.pow(cos_val,5)*3/40 + K.pow(cos_val,7)*15/336))

        hwl_loss = object_mask * K.square(raw_true_hwl - pred_hwl)
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          lambda_obj * (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 10:], from_logits=True)
        
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf / 5
        class_loss = K.sum(class_loss) / mf
        alpha_loss = K.sum(alpha_loss) / mf
        hwl_loss = K.sum(hwl_loss) / mf

        loss += (xy_loss + wh_loss + confidence_loss + class_loss + alpha_loss + hwl_loss)

        print_loss = True
        if print_loss and l == 2:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, alpha_loss, hwl_loss, K.sum(object_mask)],
                            message='loss: ')
    return loss

def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
    :param b2: true box, shape=[None, 13, 13, 3, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3, 1] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b1 = tf.expand_dims(b1, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b1_xy = b1[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2.  # w/2, h/2 shape= (None, 13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half  # x,y: left bottom corner of BB
        b1_maxes = b1_xy + b1_wh_half  # x,y: right top corner of BB
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # w1 * h1 (None, 13, 13, 3, 1)

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        # b2 = tf.expand_dims(b2, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b2 = tf.expand_dims(b2, 0)  # shape= (1, None, 13, 13, 3, 4)  # TODO 0?
        b2_xy = b2[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b2_wh = b2[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b2_wh_half = b2_wh / 2.  # w/2, h/2 shape=(None, 13, 13, 3, 1, 2)
        b2_mins = b2_xy - b2_wh_half  # x,y: left bottom corner of BB
        b2_maxes = b2_xy + b2_wh_half  # x,y: right top corner of BB
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # w2 * h2

    with tf.name_scope('Intersection'):
        """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
        # intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
        intersect_mins = K.maximum(b1_mins, b2_mins)  # (None, 13, 13, 3, 1, 2)
        # intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # intersection: wi * hi (None, 13, 13, 3, 1)

    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (None, 13, 13, 3, 1)

    return IoU



