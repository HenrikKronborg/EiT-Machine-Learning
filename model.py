import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer

def createModel(opt, loss):
    '''sets up the model architeture and returns the model.
    '''
    # init model.
    model = keras.Sequential()

    model.add(InputLayer(input_shape=(300, 300, 1)))
    model.add(Conv2D(24, (5, 5), activation="relu", name="conv2d_1"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(24, (4, 4), activation="relu", name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))

    # output.
    model.add(keras.layers.Dense(20, activation=tf.nn.softmax))

    
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model
