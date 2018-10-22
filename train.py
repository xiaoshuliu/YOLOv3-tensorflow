from config import Input_shape, channels, path, data_path
from network_function import YOLOv3

from loss_function import compute_loss
from yolo_utils import get_training_data, read_anchors, read_classes, load_training_data, online_process

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

np.random.seed(101)

# Add argument
def argument():
    parser = argparse.ArgumentParser(description='COCO or VOC or boat')
    parser.add_argument('--COCO', action='store_true', help='COCO flag')
    parser.add_argument('--VOC', action='store_true', help='VOC flag')
    parser.add_argument('--boat', action='store_true', help='boat flag')
    args = parser.parse_args()
    return args
# Get Data #############################################################################################################
PATH = path
DATA_PATH = data_path
classes_paths = PATH + '/model/bdd_classes.txt'
classes_data = read_classes(classes_paths)
anchors_paths = PATH + '/model/bdd_anchors.txt'
anchors = read_anchors(anchors_paths)
print ("Number of classes: ", len(classes_data))

annotation_path_train = PATH + '/model/bdd_train.txt'
annotation_path_valid = PATH + '/model/bdd_val.txt'
annotation_path_test = PATH + '/model/bdd_test.txt'

# data_path_train = PATH + '/model/bdd_train.npz'
# data_path_valid = PATH + '/model/bdd_valid.npz'
# data_path_test = PATH + '/model/bdd_test.npz'

data_path_train = DATA_PATH + '/bdd_train/'
data_path_valid = DATA_PATH + '/bdd_valid/'
data_path_test = DATA_PATH + '/bdd_test/'

label_train = []
label_valid = []
with open(annotation_path_train) as f:
        label_train = f.readlines()
with open(annotation_path_valid) as f:
        label_valid = f.readlines()


input_shape = (Input_shape, Input_shape)  # multiple of 32
# x_train, box_data_train, image_shape_train, y_train = get_training_data(annotation_path_train, data_path_train,
#                                                                         input_shape, anchors, "train", num_classes=len(classes_data), max_boxes=40, load_previous=True)
# x_valid, box_data_valid, image_shape_valid, y_valid = get_training_data(annotation_path_valid, data_path_valid,
#                                                                         input_shape, anchors, "val", num_classes=len(classes_data), max_boxes=40, load_previous=True)
# # x_test, box_data_test, image_shape_test, y_test = get_training_data(annotation_path_test, data_path_test,
# #                                                                     input_shape, anchors, num_classes=len(classes_data), max_boxes=20, load_previous=True)
print("number_image_train", len(label_train))
print("number_image_valid", len(label_valid))

########################################################################################################################
"""
# Clear the current graph in each run, to avoid variable duplication
# tf.reset_default_graph()
"""
print("Starting 1st session...")
# Explicitly create a Graph object
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Start running operations on the Graph.
    # STEP 1: Input data ###############################################################################################

    X = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels], name='Input')  # for image_data
    with tf.name_scope("Target"):
        Y1 = tf.placeholder(tf.float32, shape=[None, Input_shape/32, Input_shape/32, 3, (5+len(classes_data))], name='target_S1')
        Y2 = tf.placeholder(tf.float32, shape=[None, Input_shape/16, Input_shape/16, 3, (5+len(classes_data))], name='target_S2')
        Y3 = tf.placeholder(tf.float32, shape=[None, Input_shape/8, Input_shape/8, 3, (5+len(classes_data))], name='target_S3')
        # Y = tf.placeholder(tf.float32, shape=[None, 100, 5])  # for box_data
    # Reshape images for visualization
    x_reshape = tf.reshape(X, [-1, Input_shape, Input_shape, 1])
    tf.summary.image("input", x_reshape)
    # STEP 2: Building the graph #######################################################################################
    # Building the graph
    # Generate output tensor targets for filtered bounding boxes.
    scale1, scale2, scale3 = YOLOv3(X, len(classes_data)).feature_extractor()
    scale_total = [scale1, scale2, scale3]

    with tf.name_scope("Loss_and_Detect"):
        # Label
        y_predict = [Y1, Y2, Y3]
        # Calculate loss
        loss = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
        tf.summary.scalar("Loss", loss)
        
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

    # STEP 3: Build the evaluation step ################################################################################
    # with tf.name_scope("Accuracy"):
    #     # Model evaluation
    #     accuracy = 1  #

    # STEP 4: Merge all summaries for Tensorboard generation ###########################################################
    # create a saver
    # saver = tf.train.Saver(tf.global_variables())
    # Returns all variables created with trainable=True
    # saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver = tf.train.Saver()
    # Build the summary operation based on the TF collection of Summaries
    # summary_op = tf.summary.merge_all()

    # STEP 5: Train the model, and write summaries #####################################################################
    # The Graph to be launched (described above)
    # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True) #, gpu_options.allow_growth = False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
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
        checkpoint = "save_model/bdd10/epoch_new.ckpt-" + "4"
        saver.restore(sess, checkpoint)

        epochs = 50  #
        batch_size = 24  # consider
        best_loss_valid = 10e6
        number_image_train = len(label_train)
        number_image_valid = len(label_valid)
        num_classes = len(classes_data)
        print(number_image_train, "images for training.")
        for epoch in range(epochs):
            start_time = time.time()

            ## Training#################################################################################################
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
                    x_, y_ = online_process(label_line, input_shape, anchors, num_classes)
                    # x_, _, _, y_ = load_training_data(data_name)
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

                # summary_train, loss_train, _ = sess.run([summary_op, loss, optimizer],
                #                                         feed_dict={X: (x_train[random_indices[start:end]]/255.),
                #                                                    Y1: y_train[0][random_indices[start:end]],
                #                                                    Y2: y_train[1][random_indices[start:end]],
                #                                                    Y3: y_train[2][random_indices[start:end]]})

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
            if epoch % 5 == 0:
                create_new_folder = PATH + "/save_model/bdd10"
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass
                checkpoint_path = create_new_folder + "/epoch_new" + ".ckpt"
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % checkpoint_path)

        print("Tuning completed!")

        # Testing ######################################################################################################
        # mean_loss_test = []
        # for start in (range(0, 128, batch_size)):
        #     end = start + batch_size
        #     if end > number_image_train:
        #         end = number_image_train
        #     # Loss in test data set
        #     summary_test, loss_test = sess.run([summary_op, loss],
        #                                        feed_dict={X: (x_test[start:end]/255.),
        #                                                   Y1: y_test[0][start:end],
        #                                                   Y2: y_test[1][start:end],
        #                                                   Y3: y_test[2][start:end]})
        #     mean_loss_test.append(mean_loss_test)
        #     # print("Loss on test set: ", (loss_test))
        # mean_loss_test = np.mean(mean_loss_test)
        # print("Mean loss in all of test set: ", mean_loss_test)
        # summary_writer.flush()
        # train_summary_writer.close()
        # validation_summary_writer.close()
        # summary_writer.close()



