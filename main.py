import os
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras import losses

with open("training_set.P", mode="rb") as pickle_file:
    train_data = np.load(pickle_file)

with open("training_labels.P", mode="rb") as pickle_file:
    train_labels = np.load(pickle_file)

print(train_data.shape)

plt.imshow(train_data[0])


train_data = np.array(train_data)
train_data = np.resize(train_data, (500, 300, 300, 1))

print(train_data.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(300, 300, 1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(528, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])

print("dfsdfs", train_labels[0])

model.compile(optimizer='adam',
            loss="mean_squared_error",
            metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=2)

print(model.predict(np.array([train_data[0]])))


print(train_labels[0])