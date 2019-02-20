from pathlib import Path
from image_reader import ARRAY_FROM_PATH
from image_reader import MAX_HEIGHT, MAX_WIDTH
import csv
import numpy as np
import matplotlib.pyplot as plt


# --- Parameters -------------------------------------------------------------

TRAINING_SET     = "boneage-training-dataset"
TRAINING_COUNT   = 500
TEST_SET         = "boneage-test-dataset"
TEST_COUNT       = 200
SAMPLES_METADATA = "boneage-training-dataset.csv"


# ----------------------------------------------------------------------------

# Go through the csv file and complete the dictionary maps
with open(SAMPLES_METADATA, mode='r') as csv_file:
    csv_data = csv.reader(csv_file, delimiter=',')
    
    # The first line is the header, which we ignore
    csv_data.__next__()
    
    # Fetch the data
    data = [(id_, int(age), is_male=="True") for id_, age, is_male in csv_data]
    
    # Transpose the dataset
    ids, ages, is_males = zip(*data)
    

id_age = {id_: age for id_, age in zip(ids, ages)}

# Initialize numpy array (3-tensor) for the training arrays & age labels
training_arrays = np.empty((TRAINING_COUNT, MAX_HEIGHT, MAX_WIDTH))
training_age = np.empty(TRAINING_COUNT)

# Iterate through the training set folder and add to image_arrays
for i, image_file in enumerate(Path(TRAINING_SET).iterdir()):
    print(f"{i+1}/{TRAINING_COUNT}")
    
    training_id = image_file.name.replace('.png', '')
    
    training_age[i] = id_age[training_id]
    training_arrays[i] = ARRAY_FROM_PATH(str(image_file))

# Initialize numpy array (3-tensor) for the training arrays & age labels
test_arrays = np.empty((TRAINING_COUNT, MAX_HEIGHT, MAX_WIDTH))
test_age = np.empty(TRAINING_COUNT)

# Iterate through the training set folder and add to image_arrays
for i, image_file in enumerate(Path(TEST_SET).iterdir()):
    print(f"{i+1}/{TEST_COUNT}")
    
    test_id = image_file.name.replace('.png', '')
    
    test_age[i] = id_age[test_id]
    test_arrays[i] = ARRAY_FROM_PATH(str(image_file))
    
file_object = {
    "training_set"    : training_arrays,
    "training_labels" : training_age,
    "test_set"        : test_arrays,
    "test_labels"     : test_age,
}
    
# Pickle dump the image arrays
for file, obj in file_object.items():
    with open(f"{file}.P", mode='wb') as pickle_file:
        obj.dump(pickle_file)

if __name__ == "__main__":
    mean_image = np.mean(training_arrays, axis=0)
    plt.imshow(mean_image); plt.show()