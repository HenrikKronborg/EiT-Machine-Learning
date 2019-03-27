#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:50:08 2019

@author: simen
"""

import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer
from keras.applications.vgg16 import VGG16

def createModel(opt, loss, in_shape, num_classes):
    # Create an instance of the VGG_model without FC network
    vgg_model = VGG16(include_top = False, weights = 'imagenet', input_shape = in_shape)
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])    

    x = layer_dict['block5_pool'].output

    # Stacking a new fully connected network on top of it    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    custom_model = keras.Model(input = vgg_model.input, output = x)

    # Only the last layers should be retrained
    for layer in custom_model.layers[:-4]:
        layer.trainable = False
    
    custom_model.compile(opt, loss, metrics = ['accuracy'])

    return custom_model
