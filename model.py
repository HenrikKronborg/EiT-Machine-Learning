import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer

def createModel(opt, loss, in_shape, num_classes):
    '''sets up the model architeture and returns the model.
    '''
    # init model.
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2),
                         activation='relu',
                         input_shape=(in_shape[0], in_shape[1], 1)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    

    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
