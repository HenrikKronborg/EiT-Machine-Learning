from random import randint
from dataset_loader import data
from model import createModel

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


### hyper parameters adjust these:

in_shape = (100, 100)

NUM_CLASSES = 20
EPOCHS = 20
BATCH_SIZE = 32

LEARNING_RATE = 0.001

loss = keras.losses.mean_squared_error
opt = keras.optimizers.Adam(lr=LEARNING_RATE)

### ----------------------------------------

# load data
train_data = data["training_set"]
train_labels = data["training_labels"]

# squeeze pixel value between 0 and 1
train_data /= 255

# reshape data
train_data = np.array(train_data)
train_data = np.resize(train_data, (len(train_data), in_shape[0], in_shape[1], 1))

# convert labels to years and one hot encode
ezyConvert = lambda x: x // 12
train_labels = ezyConvert(train_labels)

# one hot encode labels
train_labels = keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
print("one hot: ", train_labels[0]); print("one hot: ", train_labels[1])


# creates a model with the structure defined in model.py
model = createModel(opt, loss, in_shape, NUM_CLASSES)


# start training.
hist = model.fit(train_data, train_labels, epochs=EPOCHS,
          batch_size=BATCH_SIZE)
          
print("-----------    Training finished.    -----------")
print(hist.history)

# save model.
model.save("new_model.h5")



## testing < --------

# load data
test_data = np.array(data["test_set"])
test_data = np.resize(test_data, (test_data.shape[0], in_shape[0], in_shape[1], 1))

# normalize!
test_data /= 255

# convert labels
test_labels = ezyConvert(data["test_labels"])
test_labels = keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

# test.
score = model.evaluate(test_data, test_labels)
print("Test set:")
print("loss:", score[0])
print("accuracy:", score[1])

'''
# ---------------- yala testing --------------
for i in range(0, 10):
    pred = model.predict(np.array([train_data[i]]))

    print("pred output: ", pred)
    print("predicted: ", np.argmax(pred))
    print("actual answer: ", np.argmax(train_labels[i]))

    plt.imshow(np.resize(train_data[i], (100, 100)))
    plt.show()
'''




'''
def mmnist example():
    from keras.datasets import mnist

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = train_data.reshape(train_data.shape[0], in_shape[0], in_shape[1], 1)

    train_data = train_data.astype('float32')

    train_data /= 255

    train_labels = keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)

'''