from random import randint
from VGG import createModel

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# load training data and convert to RGB
train_data = np.load("/lustre1/work/johnew/EiT/data/training_set.npy")
train_labels = np.load("/lustre1/work/johnew/EiT/data/training_labels.npy")

train_data = np.array(train_data)
train_data = np.resize(train_data, (*train_data.shape, 1))
train_data /= 255

rgb_train = np.zeros((*train_data.shape[0:3],3))
for i in range(train_data.shape[0]):
    rgb_train[i] = np.concatenate([train_data[i],train_data[i],train_data[i]], axis=2)

# convert labels to years and one hot encode
ezyConvert = lambda x: x // 12
train_labels = ezyConvert(train_labels)
oneHot_labels = keras.utils.to_categorical(train_labels)

# hyper parameters
EPOCHS = 100
BATCH_SIZE = 64
IN_SHAPE = rgb_train[0].shape
NUM_CLASSES = 20
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 0.01
LAMBDA = 0

loss = keras.losses.MSE
opt = keras.optimizers.Adam(lr = LEARNING_RATE, decay = LAMBDA)

# creates a model with the structure defined in VGG.py
model = createModel(opt, loss, IN_SHAPE, NUM_CLASSES)
early_stop = keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

# start training
hist = model.fit(rgb_train, oneHot_labels, validation_split=0.1, epochs=EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[early_stop])
          
print("-----------    Training finished.    -----------")

# save model.
model.save("new_model.h5")



np.save("histTest.npy", hist)
