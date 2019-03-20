import tensorflow
import keras

from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix





if __name__ == "__main__":

    MODEL_NAME = "new_model.h5"

    model = load_model(MODEL_NAME)

    DATA_PATH = "test_set.npy"
    LABEL_PATH = "test_labels.npy"

    test = np.load(DATA_PATH)
    test_labels = np.load(LABEL_PATH)

    pred = model.predict(test)

    cm = confusion_matrix(test, pred)
    

    ## predict the given data.
    # predicted = model.predict()