#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:50:08 2019

@author: simen
"""

import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer

num_filters = 32

def createModel(opt, loss):
    '''sets up the model architeture and returns the model.
    '''
    # init model.
    model = keras.Sequential()

    # 4 Conv. layers with num_filters in each layer
    model.add(Conv2D(num_filters, kernel_size = 5, padding = "same", activation=tf.nn.relu, input_shape = (300,300,1)))
    model.add(Conv2D(num_filters, kernel_size = 5, padding = "same", activation=tf.nn.relu))
    model.add(Conv2D(num_filters, kernel_size = 5, padding = "same", activation=tf.nn.relu))
    model.add(Conv2D(num_filters, kernel_size = 5, padding = "same", activation=tf.nn.relu))
    
    # Flatten and fully connected layer
    model.add(Flatten())
    
    # 4 Fully Connected layers
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))

    # output.
    model.add(keras.layers.Dense(20, activation=tf.nn.softmax))

    
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model
