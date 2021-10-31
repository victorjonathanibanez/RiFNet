#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# RiFNet training code 
# Author: Victor Ibanez, University of Zurich, Institute of Forensic Medicine
# Contact: victor.ibanez@uzh.ch
# -------------------------------------------------------------------------------------

#----------------------------------------------------------------------------
# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import sys
import random
import math
import gc
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

#----------------------------------------------------------------------------
# load helper functions


# creates directory if it doesn't already exist
def createdir(path): # create directory
    if not os.path.isdir(path):
        os.makedirs(path)


# stores training images and labels to array and saves testimages to test dir
# input: list of training-files, list of test-files, directory of current iter-
# ation, image-label dictionary, label names
def run_split(train_list, test_list, test_dir, lab_dict, lab_names):

    # append training-val-set images to image-array and labels to label-array
    for i in range(len(train_list)):
        image = Image.open(train_list[i])
        image = np.asarray(image)
        X[i, :] = image # save image to array
        if lab_dict[train_list[i]] == lab_names[0]: # save label to array
            y[i] = 0 # class one
        else:
            y[i] = 1 # class two

    # save testset to test folder
    for i in range(len(test_list)):
        image = Image.open(test_list[i])
        filename = os.path.basename(test_list[i])
        file_label = lab_dict[test_list[i]]
        image_path = os.path.join(test_dir, file_label, filename)
        image.save(image_path)

    return X, y


# appends dataframe with accuracy and loss values of model
# input: training history, # of k-fold, # of iteration, dataframe to append to
def store_acc_loss(training_history, split_nr, it_nr, data_frame):
    h_a = training_history.history['acc'][0]
    h_va = training_history.history['val_acc'][0]
    h_l = training_history.history['loss'][0]
    h_vl = training_history.history['val_loss'][0]
    iteration = split_nr*it_nr
    dat = np.array([[iteration, h_a, h_va, h_l, h_vl]])
    new_df = pd.DataFrame(dat, columns=list(data_frame.columns))
    data_frame = data_frame.append(new_df, ignore_index=True)
    return(data_frame)


# saves plots of accuary and loss values
# input: training dataframe, # of k-fold, # of iteration, output directory
# note: returns None
def plot_acc_loss(dframe, split_nr, it_nr, out_dir):
    # plot acc
    plt.plot(dframe['acc'])
    plt.plot(dframe['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plot_name1 = os.path.join(out_dir, 'it{}_fold{}_acc.png'.format(str(it), str(n_split)))
    plt.savefig(plot_name1)
    plt.close()
    # plot loss
    plt.plot(dframe['loss'], color='C2')
    plt.plot(dframe['val_loss'], color='C3')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plot_name2 = os.path.join(out_dir, 'it{}_fold{}_loss.png'.format(str(it), str(n_split)))
    plt.savefig(plot_name2)
    plt.close()


#----------------------------------------------------------------------------
# set directories, image parameters and seeds

# get input directories and split percentage
in_dir_one = sys.argv[1] # class one (label = 0)
in_dir_two = sys.argv[2] # class two (label = 1)
data_split = sys.argv[3] # training testing percentage

# set image filetype and image size
img_type = 'png'
img_height = 1024
img_width = img_height
img_depth = 3 # rgb, 3 channels

# set 5 seeds, one for each iteration
# note: seed for kfold split can stay the same since imgs change each iteration
seeds = list(range(61, 11, -10))

#----------------------------------------------------------------------------
# initialize dependencies for training, image access and statistics

# get classes [0, 1]
label_names = [os.path.basename(in_dir_one), os.path.basename(in_dir_two)]

# create dataframe to store cv data
col_names = ['iter','acc','val_acc','loss','val_loss']
data = np.zeros(shape=(1,5))
df = pd.DataFrame(data, columns=col_names)

label_dict = {}
# fill dictionary with filepaths as keys and labels as values
for f in glob.glob(os.path.join(in_dir_one, '*.{}'.format(img_type))):
    label_dict[f] = label_names[0]
for f in glob.glob(os.path.join(in_dir_two, '*.{}'.format(img_type))):
    label_dict[f] = label_names[1]

# initiate list to store cv stats
stats_list = []

#----------------------------------------------------------------------------
# set initial-model parameters

# create directory to save initial model and run-files
run_dir = './CV_run_dir'
createdir(run_dir)

# settings
input_shape = (img_height, img_width, img_depth)
nclasses = 1
epochs = 30
batch_size = 15
loss_type = 'binary_crossentropy'
class_type = 'binary'
activation_fun = 'sigmoid'
learning_rate = 0.00015
dropout_rate = 0.5
reduce_lr = ReduceLROnPlateau(monitor='acc',
                            factor=0.2,
                            patience=5,
                            min_lr=0.00005)
opt = optimizers.Adam(lr=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=False)

#----------------------------------------------------------------------------
# build initial-model

# conv layer 1
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 2
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 3
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 5
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
model.add(Flatten())

# dense layer 1
model.add(Dense(500))
model.add(Activation('relu'))

# dense layer 2
model.add(Dense(500))
model.add(Activation('relu'))

# dropout layer
model.add(Dropout(dropout_rate))

# output layer
model.add(Dense(nclasses))
model.add(Activation(activation_fun))

# Compile model
model.compile(loss=loss_type,
            optimizer=opt,
            metrics=['acc'])

model.save(os.path.join(run_dir, 'base_model.h5'))

#----------------------------------------------------------------------------
# run 5 times k-fold cross validation

for it in range(1,6):

    # define paths for current iteration
    current_test_dir = 'CV_test_dir_{}'.format(str(it))
    test_class_one = os.path.join(current_test_dir, label_names[0])
    test_class_two = os.path.join(current_test_dir, label_names[1])
    #it_dir = os.path.join(run_dir, 'cv_iteration_' + str(it))
    # create directories
    createdir(current_test_dir)
    createdir(test_class_one)
    createdir(test_class_two)
    #createdir(it_dir)

    # shuffle image list
    img_names = list(label_dict.keys())
    random.seed(seeds[it-1])
    random.shuffle(img_names)

    # split image list into training and testing list
    training = img_names[0:math.floor(len(img_names)/100*data_split)]
    testing = img_names[len(training):]

    # initiate arrays for images and labels
    X = np.zeros((len(training), img_height, img_width, img_depth)) # samples, rows, columns, channels
    y = np.zeros((len(training)))

    # load arrays with images and labels
    X, y = run_split(training, testing, current_test_dir, label_dict, label_names)

    # show an image
    #plt.title(y[256])
    #plt.imshow(X[256]/256.)
    #plt.show()

    # initiate 5-fold cross validation (4 parts training 1 part validation)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seeds[0])

    # apply model cross validation
    n_split = 1
    for train, val in kfold.split(X, y):

        print('\ntrain/test split:', it)
        print('k =', n_split)

        n_split_train_samples = train*nclasses
        n_split_validation_samples = val*nclasses

        # load initial-model
        model = load_model(os.path.join(run_dir, 'base_model.h5'))

        # Fit the model
        history = model.fit(X[train], y[train],
                            epochs=epochs,
                            callbacks=[reduce_lr],
                            batch_size=batch_size,
                            validation_data=(X[val],y[val]))

        # append acc and loss values to dataframe
        df = store_acc_loss(history, n_split, it, df)
        print(df.round(2))

        # plot acc and loss values
        plot_acc_loss(df, n_split, it, run_dir)

        # evaluate the model
        scores = model.evaluate(X[val], y[val], verbose=0)
        #print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

        # save evaluation
        cvscores = list()
        cvscores.append(scores[1] * 100)

        # save the model
        model_name = 'it{}_fold{}_model.h5'.format(str(it), str(n_split))
        model.save(os.path.join(run_dir, model_name))

        # clear memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        n_split += 1

    # calculate mean and st deviation of CAs for each CV cycle
    stats = ("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    stats_list.append(stats)
    print(stats)

#----------------------------------------------------------------------------
# write output files

# write text file
f = open(os.path.join(run_dir, 'CV_stats.txt'), 'w+')
for s in range(len(stats_list)):
    f.write('stats_run_#_{}'.format(s+1) + '\n' + stats_list[s] + '\n')
f.close()

# write a csv with all values
df.to_csv(os.path.join(run_dir, 'values.csv', index=False)

#----------------------------------------------------------------------------
