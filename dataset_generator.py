from pathlib import Path
from image_reader import ARRAY_FROM_PATH
from image_reader import MAX_HEIGHT, MAX_WIDTH
import csv
import numpy as np
import matplotlib.pyplot as plt


# --- Parameters -------------------------------------------------------------

TRAINING_DIR     = Path("/lustre1/work/johnew/EiT/data/boneage-training-dataset")
TEST_DIR         = Path("/lustre1/work/johnew/EiT/data/test-dataset")
SAMPLES_METADATA = Path("/lustre1/work/johnew/EiT/data/boneage-training-dataset.csv")

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
    

# id->age mapping
id_age = {id_: age for id_, age in zip(ids, ages)}

def obtain_data(data_path, id_label):
    # Objects in dataset
    obj_count = sum(1 for obj in data_path.iterdir())
    
    # Initialize numpy array (3-tensor) for the training arrays & age labels
    dataset = np.empty((obj_count, MAX_HEIGHT, MAX_WIDTH))
    label = np.empty(obj_count)
    for i, image_file in enumerate(data_path.iterdir()):
        print(f"Reading file {i+1} of {obj_count}.")
        
        # gets array from image path
        dataset[i] = ARRAY_FROM_PATH(image_file.__str__())
        
        # gets label from id (file name)
        label[i] = id_label[image_file.stem]
    
    return dataset, label

# Iterate through the training set folder and add to image_arrays
training_arrays, training_age = obtain_data(TRAINING_DIR, id_age)

# Iterate through the training set folder and add to image_arrays
test_arrays, test_age = obtain_data(TEST_DIR, id_age)

# Pickle file names and their objects
file_object = {
    "training_set"    : training_arrays,
    "training_labels" : training_age,
    "test_set"        : test_arrays,
    "test_labels"     : test_age,
}

# Pickle dump the image arrays
for file, obj in file_object.items():
    with open(f"/lustre1/work/johnew/EiT/data/2{file}.npy", mode='wb') as pickle_file:
        print(f"Writing {file}.P.")
        np.save(pickle_file, obj)

if __name__ == "__main__":
    mean_image = np.mean(training_arrays, axis=0)
    plt.imshow(mean_image); plt.show()
