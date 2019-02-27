from random import randint
from dataset_loader import data
from model import createModel

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras



### hyper parameters adjust these:

EPOCHS = 1
BATCH_SIZE = 1

LEARNING_RATE = 0.001

loss = keras.losses.mean_squared_error
opt = keras.optimizers.adadelta(lr=LEARNING_RATE)

### ----------------------------------------


train_data = data["training_set"]
train_labels = data["training_labels"]

train_data = np.array(train_data)
train_data = np.resize(train_data, (500, 300, 300, 1))


# convert labels to years and one hot encode
ezyConvert = lambda x: x // 12
train_labels = ezyConvert(train_labels)
oneHot_labels = keras.utils.to_categorical(train_labels)

# creates a model with the structure defined in model.py
model = createModel(opt, loss)


# start training.
hist = model.fit(train_data, oneHot_labels, epochs=EPOCHS,
          batch_size=BATCH_SIZE)
          
print("-----------    Training finished.    -----------")
print(hist.history)

# save model.
model.save("new_model.h5")


# ---------------- yala testing --------------
for i in range(0, 5):
    pred = model.predict(np.array([train_data[i]]))

    print("pred output: ", pred)
    print("predicted: ", np.argmax(pred))
    print("actual answer: ", np.argmax(oneHot_labels[i]))

    plt.imshow(np.resize(train_data[i], (300, 300)))
    plt.show()
