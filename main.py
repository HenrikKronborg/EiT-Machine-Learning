import os
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.optimizers import adam
from keras.utils import to_categorical

## hyper parameters adjust these:

EPOCHS = 5
BATCH_SIZE = 8

opt = adam(lr=0.001)


with open("training_set.P", mode="rb") as pickle_file:
    train_data = np.load(pickle_file)

with open("training_labels.P", mode="rb") as pickle_file:
    train_labels = np.load(pickle_file)

print("input data shape:", train_data.shape)
print("inpu fdsfsdfs ", train_labels)

plt.imshow(train_data[0])
print(train_data[0])

train_data = np.array(train_data)
train_data = np.resize(train_data, (500, 300, 300, 1))

# convert labels to years:
print("before convert.")
print(train_labels[1])

ezyConvert = lambda x: x // 12
train_labels = ezyConvert(train_labels)
print("after..")
print(train_labels.shape)
print(train_labels[1])

oneHot_labels = to_categorical(train_labels)
print(oneHot_labels.shape)
print(oneHot_labels[1])

print(train_data.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(300, 300, 1)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(19, activation=tf.nn.softmax)
])

print("dfsdfs", train_labels[0])

# opt = optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

model.compile(optimizer=opt,
              loss="mean_squared_error",
              metrics=['accuracy'])

print("Starting training...")
model.fit(train_data, oneHot_labels, epochs=EPOCHS,
          batch_size=BATCH_SIZE)
print("-----------    Training finished.    -----------")


# ------------- testing --------------
for i in range(0, 10):
    pred = model.predict(np.array([train_data[i]]))

    print("predicted: ", np.argmax(pred))
    print("actual answer: ", np.argmax(oneHot_labels[i]))

    plt.imshow(np.resize(train_data[i], (300, 300)))
    plt.show()
