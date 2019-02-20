import numpy as np
file_object = {
    "training_set"    : None,
    "training_labels" : None,
    "test_set"        : None,
    "test_labels"     : None,
}
    
# Pickle dump the image arrays
for file in file_object:
    with open(f"{file}.P", mode="rb") as pickle_file:
        print(f"reading {file}")
        file_object[file] = np.load(pickle_file)

if __name__ == "__main__":
    print(file_object["training_set"].shape)