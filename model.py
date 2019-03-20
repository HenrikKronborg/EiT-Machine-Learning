import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer

def createModel(opt, loss, in_shape, num_classes):
    '''sets up the model architeture and returns the model.
    '''
    # init model.
    model = keras.Sequential()

<<<<<<< HEAD
    model.add(Conv2D(128, kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(in_shape[0], in_shape[1], 1)))
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
=======
    model.add(Conv2D(256, kernel_size=(2, 2),
                         activation='relu',
                         input_shape=(in_shape[0], in_shape[1], 1)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (2, 2), activation='relu')) 
    model.add(Conv2D(32, (2, 2), activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
>>>>>>> bf309e7ebfd30a6c3cfec70ba7ff139e5fc59826

    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
<<<<<<< HEAD
    model.add(Dense(512, activation='relu'))
=======
    model.add(Dense(256, activation='relu'))
>>>>>>> bf309e7ebfd30a6c3cfec70ba7ff139e5fc59826
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
