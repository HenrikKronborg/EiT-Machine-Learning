import tensorflow
import keras

from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix





if __name__ == "__main__":

    MODEL_NAME = "subSet-model.h5"

    model = load_model(MODEL_NAME)
    print("data loaded.")

    DATA_PATH = "2test_set.npy"
    LABEL_PATH = "2test_labels.npy"

    test = np.load(DATA_PATH)[100:]
    test = np.resize(test, (len(test), 100, 100, 1))


    test_labels = np.load(LABEL_PATH)[100:]
    ezyConvert = lambda x: x // 12
    test_labels = ezyConvert(test_labels)


    pred = model.predict(test)

    pred = np.argmax(pred, axis=1)

    print("shape: ", pred.shape)
    print("test shape: ", test_labels.shape)
    print(test_labels)


    #cm = confusion_matrix(test_labels, pred)

    from pandas_ml import ConfusionMatrix
    import matplotlib.pyplot as plt

    print(test_labels[:10])
    print(pred[:10])


  
    cm = ConfusionMatrix(test_labels, pred)
    cm.plot()
    plt.show()

    cm.plot()
    

    ## predict the given data.
    # predicted = model.predict()