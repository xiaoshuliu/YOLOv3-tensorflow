from config import Input_width, Input_height, channels, path, data_path, learning_rate, batch_size
from network_function import YOLOv3

from loss_function import compute_loss
from yolo_utils import read_anchors, read_classes, load_training_data, online_process

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# might need to set random seed the same in debug
# np.random.seed(101) 

# Get Data #############################################################################################################
PATH = path
DATA_PATH = data_path
classes_paths = PATH + '/model/kitti_classes.txt'
classes_data = read_classes(classes_paths)
anchors_paths = PATH + '/model/kitti_anchors.txt'
anchors = read_anchors(anchors_paths)
print ("Number of classes: ", len(classes_data))

annotation_path_train = PATH + '/model/kitti_train.txt'
annotation_path_valid = PATH + '/model/kitti_train.txt'
annotation_path_test = PATH + '/model/kitti_train.txt'

label_train = []
label_valid = []
with open(annotation_path_train) as f:
        label_train = f.readlines()
with open(annotation_path_valid) as f:
        label_valid = f.readlines()

input_shape = (Input_height, Input_width)  # multiple of 32
print("Number_image_train", len(label_train))

########################################################################################################################

# Explicitly create a Graph object
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Start running operations on the Graph.
    # STEP 1: Input data ###############################################################################################
    X = tf.placeholder(tf.float32, shape=[None, Input_height, Input_width, channels], name='Input')  # for image_data
    with tf.name_scope("Target"):
        # for 3D
        Y1 = tf.placeholder(tf.float32, shape=[None, Input_height/32, Input_width/32, 3, (5+5+len(classes_data))], name='target_S1')
        Y2 = tf.placeholder(tf.float32, shape=[None, Input_height/16, Input_width/16, 3, (5+5+len(classes_data))], name='target_S2')
        Y3 = tf.placeholder(tf.float32, shape=[None, Input_height/8, Input_width/8, 3, (5+5+len(classes_data))], name='target_S3')

    # STEP 2: Building the graph #######################################################################################
    # Building the graph
    # Generate output tensor targets for filtered bounding boxes.
    scale1, scale2, scale3 = YOLOv3(X, len(classes_data)).feature_extractor()
    y_pred = [scale1, scale2, scale3]

    with tf.name_scope("Loss_and_Detect"):
        # Label
        y_gt = [Y1, Y2, Y3]
        # Calculate loss
        loss = compute_loss(y_pred, y_gt, anchors, len(classes_data), print_loss=False)
        tf.summary.scalar("Loss", loss)
        
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # STEP 3: Merge all summaries for Tensorboard generation ###########################################################
    # create a saver
    saver = tf.train.Saver()
    # Build the summary operation based on the TF collection of Summaries
    # summary_op = tf.summary.merge_all()

    # STEP 4: Train the model, and write summaries #####################################################################
    # The Graph to be launched (described above)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=graph) as sess:
        # Merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # Summary Writers
        # tensorboard --logdir='./graphs/' --port 6005
        # train_summary_writer = tf.summary.FileWriter(PATH + '/graphs_boat10/train', sess.graph)
        # validation_summary_writer = tf.summary.FileWriter(PATH + '/graphs_boat10/validation', sess.graph)
        # summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())

        # If you want to continue training from check point
        checkpoint = "save_model/kitti/kitti_l2loss.ckpt-" + "48"
        saver.restore(sess, checkpoint)

        best_loss_valid = 10e6
        number_image_train = len(label_train)
        number_image_valid = len(label_valid)
        num_classes = len(classes_data)
        print(number_image_train, "images for training.")

        for epoch in range(epochs):
            start_time = time.time()
            ## Training #################################################################################################
            mean_loss_train = []
            random_indices = np.random.permutation(number_image_train)
            for start in (range(0, number_image_train, batch_size)):
                end = start + batch_size
                x_list = []
                y1_list = []
                y2_list = []
                y3_list = []
                for bb in range(batch_size):
                    if start + bb == number_image_train:
                        break
                    ind = random_indices[start + bb]
                    label_line = label_train[ind]
                    # process the label matrix from label file online
                    # if the disk is large enough, could process all labels and save them, and load them in training
                    # this is more efficient but lost some flexibility 
                    x_, y_ = online_process(label_line, input_shape, anchors, num_classes)
                    x_list.append(x_)
                    y1_list.append(y_[0][0])
                    y2_list.append(y_[1][0])
                    y3_list.append(y_[2][0])

                x_train = np.array(x_list)
                y1_train = np.array(y1_list)
                y2_train = np.array(y2_list)
                y3_train = np.array(y3_list)
                
                summary_train, loss_train, _ = sess.run([summary_op, loss, optimizer],
                                                        feed_dict={X: (x_train/255.),
                                                                   Y1: y1_train,
                                                                   Y2: y2_train,
                                                                   Y3: y3_train})

                # train_summary_writer.add_summary(summary_train, epoch)
                # Flushes the event file to disk
                # train_summary_writer.flush()
                # summary_writer.add_summary(summary_train, counter)
                mean_loss_train.append(loss_train)
                print("Epoch:", epoch + 1, "Image:", end, "Loss: ", loss_train)

            # summary_writer.add_summary(summary_train, global_step=epoch)
            mean_loss_train = np.mean(mean_loss_train)
            duration = time.time() - start_time
            examples_per_sec = number_image_train / duration
            sec_per_batch = float(duration)

            # Validation ###############################################################################################
            # not used for now

            # mean_loss_valid = []
            # for start in (range(0, number_image_valid, batch_size)):
            #     end = start + batch_size
            #     # Run summaries and measure accuracy on validation set
            #     x_list = []
            #     y1_list = []
            #     y2_list = []
            #     y3_list = []
            #     for bb in range(batch_size):
            #         if start + bb == number_image_valid:
            #             break
            #         ind = start + bb
            #         # data_name = train_list[ind]

            #         label_line = label_valid[ind]
            #         x_, y_ = online_process(label_line, input_shape, anchors, num_classes)
            #         # x_, _, _, y_ = load_training_data(data_name)
            #         x_list.append(x_)
            #         y1_list.append(y_[0][0])
            #         y2_list.append(y_[1][0])
            #         y3_list.append(y_[2][0])

            #     x_valid = np.array(x_list)
            #     y1_valid = np.array(y1_list)
            #     y2_valid = np.array(y2_list)
            #     y3_valid = np.array(y3_list)

            #     summary_valid, loss_valid = sess.run([summary_op, loss],
            #                                         feed_dict={X: (x_valid/255.),
            #                                                    Y1: y1_valid,
            #                                                    Y2: y2_valid,
            #                                                    Y3: y3_valid})  # ,options=run_options)

            #     # validation_summary_writer.add_summary(summary_valid, epoch)
            #     # Flushes the event file to disk
            #     # validation_summary_writer.flush()
            #     mean_loss_valid.append(loss_valid)
            # mean_loss_valid = np.mean(mean_loss_valid)

            # print("epoch %s / %s \ttrain_loss: %s,\tvalid_loss: %s" %(epoch+1, epochs, mean_loss_train, mean_loss_valid))

            # if best_loss_valid > mean_loss_valid:
            #     best_loss_valid = mean_loss_valid

            # store the model every 3 epochs
            if epoch % 3 == 0:
                create_new_folder = PATH + "/save_model/kitti"
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass
                checkpoint_path = create_new_folder + "/kitti_l2loss" + ".ckpt"
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % checkpoint_path)

        print("Tuning completed!")


